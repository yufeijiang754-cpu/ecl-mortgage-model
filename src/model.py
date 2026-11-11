"""IFRS 9 / CECL expected credit loss model on Freddie Mac single-family loans.

1. PD + prepayment: multinomial logistic regression on annual snapshots (performing loan at
   December t -> default / prepay / stay over the next 12 months), with loan, behavioral and
   macro drivers. Gradient boosting is fitted as a challenger.
2. LGD: loss severity of resolved defaults (liquidations at actual loss, cures at zero),
   gradient boosting on current LTV at default, forward house prices, MI, judicial state, etc.
3. EAD: scheduled amortization of the current balance.
4. ECL engine: year-by-year projection with competing risks (default vs prepayment),
   scenario macro paths, discounting at the note rate; IFRS 9 staging (12-month vs lifetime).
5. Validation: in-time hold-out loans, out-of-time years (2016+, incl. COVID), and an
   out-of-sample vintage backtest: 2007 loans as of Dec 2007, predicted vs realized losses.

The 2007 vintage is excluded from all model fitting.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
BACKTEST_VINTAGE = 2007
SPLINE = ["fico", "cur_ltv", "age_y", "incentive", "dti"]
LINEAR = ["orig_ltv", "unemp", "d_unemp", "hpi_g", "log_upb"]
FLAGS = ["dlq1", "dlq2", "forborne", "investor", "second_home", "cashout", "refi", "single_borrower",
         "first_time", "condo", "manufactured", "judicial"]
PD_FEATURES = SPLINE + LINEAR + FLAGS
LGD_FEATURES = ["cur_ltv", "hpi_g24", "unemp", "mi_pct", "judicial", "log_upb", "fico", "investor", "condo",
                "manufactured", "single_borrower"]
WEIGHTS = {"Baseline": 0.5, "Upside": 0.2, "Severe": 0.3}


# ----------------------------------------------------------------------------- features
def add_months(yyyymm, n):
    y, m = np.divmod(np.asarray(yyyymm, dtype=np.int64), 100)
    t = y * 12 + (m - 1) + n
    return (t // 12) * 100 + t % 12 + 1


def load_state_hpi(macro):
    """FHFA all-transactions state HPI (quarterly) -> monthly, interpolated at mid-quarter.
    Loans outside the 50 states + DC fall back to the national Case-Shiller index."""
    months = pd.period_range("1975-01", "2026-12", freq="M")
    cols = {}
    for f in sorted((ROOT / "data/macro/state_hpi").glob("*.csv")):
        q = pd.read_csv(f, header=0, names=["date", "v"], parse_dates=["date"]).dropna()
        q.index = (q.date + pd.DateOffset(months=1)).dt.to_period("M")
        cols[f.stem] = q.v.reindex(months).interpolate(limit_area="inside").ffill().bfill()
    nat = macro.set_index("period").hpi
    nat.index = pd.PeriodIndex([pd.Period(f"{p // 100}-{p % 100:02d}", "M") for p in nat.index])
    cols["NAT"] = nat.reindex(months).ffill().bfill()
    df = pd.DataFrame(cols)
    df.index = months.year * 100 + months.month
    return df


def hpi_at(shpi, states, periods):
    states = pd.Series(states).where(pd.Series(states).isin(shpi.columns), "NAT").values
    periods = np.clip(np.asarray(periods, dtype=np.int64), shpi.index.min(), shpi.index.max())
    rows = shpi.index.get_indexer(periods)
    return shpi.values[rows, shpi.columns.get_indexer(states)]


def static_features(loans, shpi):
    f = pd.DataFrame(index=loans.index)
    f["fico"] = loans.credit_score.where(loans.credit_score.between(300, 850))
    f["fico_na"] = f.fico.isna().astype(int)
    f["fico"] = f.fico.fillna(f.fico.median())
    f["dti"] = loans.dti.where(loans.dti.between(0, 65)).fillna(loans.dti[loans.dti <= 65].median())
    f["orig_ltv"] = loans.ltv.where(loans.ltv.between(1, 200)).fillna(80)
    f["investor"] = loans.occupancy.eq("I").astype(int)
    f["second_home"] = loans.occupancy.eq("S").astype(int)
    f["cashout"] = loans.purpose.eq("C").astype(int)
    f["refi"] = loans.purpose.eq("N").astype(int)
    f["single_borrower"] = loans.n_borrowers.eq(1).astype(int)
    f["first_time"] = loans.first_time.eq("Y").astype(int)
    f["condo"] = loans.prop_type.eq("CO").astype(int)
    f["manufactured"] = loans.prop_type.eq("MH").astype(int)
    f["judicial"] = loans.judicial
    f["mi_pct"] = loans.mi_pct.where(loans.mi_pct.between(0, 55)).fillna(0)
    f["prop_value"] = loans.prop_value.fillna(loans.orig_upb / 0.8)
    f["state"] = loans.state
    f["hpi_orig"] = hpi_at(shpi, loans.state.values, loans.orig_period.values)
    f["unemp_orig"] = loans.unemp_orig
    f["orig_upb"] = loans.orig_upb
    return f


def dynamic_features(st, upb, hpi_now, age_m, dlq, forborne, rate, unemp, d_unemp, hpi_g, mort_rate):
    """Combine static loan attributes with the loan state and macro conditions at a point in time."""
    f = st.copy()
    f["cur_ltv"] = np.clip(100 * upb / (st.prop_value * hpi_now / st.hpi_orig), 1, 250)
    f["age_y"] = np.asarray(age_m) / 12
    f["dlq1"] = (np.asarray(dlq) == 1).astype(int)
    f["dlq2"] = (np.asarray(dlq) == 2).astype(int)
    f["forborne"] = np.asarray(forborne).astype(int)
    f["incentive"] = np.asarray(rate) - np.asarray(mort_rate)
    f["log_upb"] = np.log1p(np.asarray(upb))
    f["unemp"], f["d_unemp"], f["hpi_g"] = unemp, d_unemp, hpi_g
    return f


def snapshot_frame(snaps, st, macro, shpi):
    m = macro.set_index("period")
    t, t12 = snaps.period.values, add_months(snaps.period.values, 12)
    s = st.loc[snaps.loan_id.values].reset_index(drop=True)
    h_t, h_12 = hpi_at(shpi, s.state.values, t), hpi_at(shpi, s.state.values, t12)
    f = dynamic_features(s, snaps.upb.values, h_t, snaps.age.values, snaps.dlq_n.values,
                         snaps.forborne.values, snaps.rate.values, m.unemp.reindex(t).values,
                         m.unemp.reindex(t12).values - m.unemp.reindex(t).values,
                         h_12 / h_t - 1, m.mort_rate.reindex(t).values)
    f["y"] = np.select([snaps.y_default.values == 1, snaps.y_prepay.values == 1], [1, 2], 0)
    f["year"], f["vintage"], f["loan_id"] = t // 100, snaps.vintage.values, snaps.loan_id.values
    return f


# ----------------------------------------------------------------------------- models
def pd_model():
    pre = ColumnTransformer([("spline", SplineTransformer(n_knots=5, degree=3), SPLINE),
                             ("lin", "passthrough", LINEAR + FLAGS)])
    return make_pipeline(pre, StandardScaler(), LogisticRegression(C=1.0, max_iter=400))


def lgd_frame(loans, st, macro, shpi):
    """Resolved defaults: liquidations at actual loss, cures (paid off or performing) at zero."""
    m = macro.set_index("period")
    d = loans[loans.default_month.notna() & (loans.vintage != BACKTEST_VINTAGE)].copy()
    liquidated = d.exit_code.isin([2, 3, 9]) | (d.exit_code.isin([15, 16]) & d.actual_loss.notna())
    cured = (d.exit_code == 1) | (d.exit_code.isna() & (d.last_dlq < 3))
    d = d[(liquidated | cured) & d.upb_at_default.gt(0)]
    loss = np.where(liquidated.loc[d.index], d.actual_loss.fillna(0), 0.0)
    d["lgd"] = np.clip(loss / d.upb_at_default, 0, 1)
    dm = d.default_month.astype(int).values
    s = st.loc[d.index]
    h_d = hpi_at(shpi, s.state.values, dm)
    f = dynamic_features(s, d.upb_at_default.values, h_d, np.zeros(len(d)), np.zeros(len(d)),
                         np.zeros(len(d)), np.zeros(len(d)), m.unemp.reindex(dm).values, 0, 0, 0)
    ok = add_months(dm, 24) <= 202606
    f["hpi_g24"] = np.where(ok, hpi_at(shpi, s.state.values, add_months(dm, 24)) / h_d - 1, np.nan)
    f["lgd"], f["loan_id"] = d.lgd.values, d.loan_id.values
    return f.dropna(subset=["hpi_g24"])


def split_by_loan(ids, share=0.3, seed=0):
    u = pd.Series(ids).drop_duplicates()
    test = u.sample(frac=share, random_state=seed)
    return pd.Series(ids).isin(test).values


# ----------------------------------------------------------------------------- ECL engine
def amortize(upb, rate, rem, months):
    r = np.asarray(rate) / 1200
    n = np.maximum(np.asarray(rem), 1)
    pay = np.where(r > 0, upb * r / (1 - (1 + r) ** -n), upb / n)
    k = np.minimum(months, n)
    bal = np.where(r > 0, upb * (1 + r) ** k - pay * ((1 + r) ** k - 1) / np.where(r > 0, r, 1), upb - pay * k)
    return np.where(months >= n, 0.0, np.maximum(bal, 0))


def project(port, st, path, pd_mdl, lgd_mdl, discount=True):
    """Year-by-year expected loss per loan under one macro path.

    path: dict of arrays indexed by year k = 0..H, start-of-year unemployment (u), national house
    price index (hpi) and mortgage rate (mr). Each loan's state HPI (port.hpi_now) moves with the
    national scenario growth, unless the path carries a loan-level matrix `hpi_loan` (n x H+1).
    Returns 12-month ECL, lifetime ECL, 12-month PD, lifetime PD and expected loss by year.
    """
    n, H = len(port), int(np.ceil(port.rem.max() / 12))
    H = min(H, len(path["u"]) - 1)
    surv = np.ones(n)
    el_by_year = np.zeros((n, H))
    pd12 = pd_life = None
    pd_cum = np.zeros(n)
    s = st.loc[port.loan_id.values].reset_index(drop=True)
    lvl = loan_hpi(port, path)
    for k in range(H):
        upb_k = amortize(port.upb.values, port.rate.values, port.rem.values, 12 * k)
        alive = upb_k > 0
        first = k == 0
        f = dynamic_features(s, upb_k, lvl[:, k], port.age.values + 12 * k,
                             port.dlq.values if first else np.zeros(n), port.forborne.values if first else np.zeros(n),
                             port.rate.values, path["u"][k], path["u"][k + 1] - path["u"][k],
                             lvl[:, k + 1] / lvl[:, k] - 1, path["mr"][k])
        p = pd_mdl.predict_proba(f[PD_FEATURES])
        p_def, p_pre = p[:, 1] * alive, p[:, 2] * alive
        lf = f.copy()
        k24 = min(k + 2, lvl.shape[1] - 1)
        lf["hpi_g24"] = lvl[:, k24] / lvl[:, k] - 1
        lgd = np.clip(lgd_mdl.predict(lf[LGD_FEATURES]), 0, 1)
        disc = (1 + port.rate.values / 100) ** -(k + 0.5) if discount else 1.0
        marg = surv * p_def
        el_by_year[:, k] = marg * lgd * upb_k * disc
        pd_cum += marg
        if first:
            pd12 = p_def.copy()
        surv = surv * (1 - p_def - p_pre)
    return {"ecl12": el_by_year[:, 0], "ecl_life": el_by_year.sum(1), "pd12": pd12, "pd_life": pd_cum,
            "el_by_year": el_by_year}


def loan_hpi(port, path):
    if "hpi_loan" in path:
        return path["hpi_loan"]
    rel = np.asarray(path["hpi"]) / path["hpi"][0]
    return port.hpi_now.values[:, None] * rel[None, :]


def stage3_lgd(port, st, path, lgd_mdl):
    s = st.loc[port.loan_id.values].reset_index(drop=True)
    lvl = loan_hpi(port, path)
    f = dynamic_features(s, port.upb.values, lvl[:, 0], port.age.values, 0, 0, port.rate.values,
                         path["u"][0], 0, 0, path["mr"][0])
    f["hpi_g24"] = lvl[:, 2] / lvl[:, 0] - 1
    return np.clip(lgd_mdl.predict(f[LGD_FEATURES]), 0, 1)


def make_path(u0, hpi0, mr0, u_years, hpi_growth, mr_years, H=31, u_long=4.5, g_long=0.03, mr_long=6.0):
    u = np.r_[u0, u_years, np.full(H, u_long)][: H + 1]
    g = np.r_[hpi_growth, np.full(H, g_long)][:H]
    mr = np.r_[mr0, mr_years, np.full(H, mr_long)][: H + 1]
    return {"u": u, "hpi": hpi0 * np.r_[1, np.cumprod(1 + g)], "mr": mr}


def realized_path(macro, start, H=31):
    """Actual December-to-December macro path from `start`, extended with long-run assumptions."""
    m = macro.set_index("period")
    pts = [p for p in add_months(start, 12 * np.arange(H + 1)) if p in m.index]
    u, hpi, mr = m.unemp.reindex(pts).values, m.hpi.reindex(pts).values, m.mort_rate.reindex(pts).values
    extra = H + 1 - len(pts)
    return {"u": np.r_[u, np.full(extra, 4.5)], "hpi": np.r_[hpi, hpi[-1] * 1.03 ** np.arange(1, extra + 1)],
            "mr": np.r_[mr, np.full(extra, 6.0)]}


def run_ecl(port, st, paths, pd_mdl, lgd_mdl, pd12_orig):
    """Scenario-weighted IFRS 9 ECL with staging. Returns loan-level frame."""
    res = {name: project(port, st, p, pd_mdl, lgd_mdl) for name, p in paths.items()}
    pd12 = res["Baseline"]["pd12"]  # SICR compares like with like: baseline today vs neutral at origination
    in_default = (port.dlq.values >= 3) & ~port.forborne.values.astype(bool)
    sicr = (pd12 > 3 * pd12_orig) & (pd12 - pd12_orig > 0.005)
    backstop = (port.dlq.values >= 1) | port.forborne.values.astype(bool)
    stage = np.where(in_default, 3, np.where(sicr | backstop, 2, 1))
    out = pd.DataFrame({"loan_id": port.loan_id.values, "vintage": port.vintage.values, "ead": port.upb.values,
                        "stage": stage, "pd12_baseline": pd12})
    for name, r in res.items():
        lgd3 = stage3_lgd(port, st, paths[name], lgd_mdl)
        out[f"ecl_{name}"] = np.select([stage == 1, stage == 2], [r["ecl12"], r["ecl_life"]], lgd3 * port.upb.values)
        out[f"life_{name}"] = r["ecl_life"]
    out["ecl"] = sum(WEIGHTS[n] * out[f"ecl_{n}"] for n in paths)
    return out, res


# ----------------------------------------------------------------------------- main
def main():
    loans = pd.read_parquet(PROC / "loans.parquet").set_index("loan_id", drop=False)
    snaps = pd.read_parquet(PROC / "snapshots.parquet")
    macro = pd.read_parquet(PROC / "macro.parquet")
    shpi = load_state_hpi(macro)
    st = static_features(loans, shpi)
    metrics = {}

    # ---- PD / prepayment model
    df = snapshot_frame(snaps, st, macro, shpi).dropna(subset=PD_FEATURES)
    dev = df[df.vintage != BACKTEST_VINTAGE]
    hold = split_by_loan(dev.loan_id.values)
    fit_sample = split_by_loan(dev.loan_id.values, share=0.5, seed=1)  # 50% of loans keeps fitting fast
    train = dev[~hold & (dev.year <= 2015) & fit_sample]
    test_in = dev[hold & (dev.year <= 2015)]
    test_oot = dev[~hold & (dev.year >= 2016)]
    champ = pd_model().fit(train[PD_FEATURES], train.y)
    chall = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(
        train[PD_FEATURES], train.y)

    def evaluate(mdl, d):
        p = mdl.predict_proba(d[PD_FEATURES])
        return {"auc_default": round(roc_auc_score(d.y == 1, p[:, 1]), 4),
                "auc_prepay": round(roc_auc_score(d.y == 2, p[:, 2]), 4),
                "pred_default_rate": round(float(p[:, 1].mean()), 4),
                "actual_default_rate": round(float((d.y == 1).mean()), 4), "n": int(len(d))}

    metrics["pd"] = {"train_rows": int(len(train)),
                     "logistic": {"in_time_holdout": evaluate(champ, test_in), "out_of_time_2016plus": evaluate(champ, test_oot)},
                     "gbm_challenger": {"in_time_holdout": evaluate(chall, test_in),
                                        "out_of_time_2016plus": evaluate(chall, test_oot)}}
    p_oot = champ.predict_proba(test_oot[PD_FEATURES])[:, 1]
    cal = (pd.DataFrame({"year": test_oot.year, "pred": p_oot, "act": test_oot.y == 1})
           .groupby("year").mean())
    p_in = champ.predict_proba(test_in[PD_FEATURES])[:, 1]
    cal = pd.concat([pd.DataFrame({"year": test_in.year, "pred": p_in, "act": test_in.y == 1}).groupby("year").mean(),
                     cal])
    cal.to_csv(OUT / "pd_calibration_by_year.csv")
    # production PD model: all development data (still excluding the backtest vintage)
    pd_final = pd_model().fit(dev.loc[fit_sample, PD_FEATURES], dev.loc[fit_sample, "y"])
    print("PD models fitted", flush=True)

    # ---- LGD model
    lg = lgd_frame(loans, st, macro, shpi).dropna(subset=LGD_FEATURES)
    lh = split_by_loan(lg.loan_id.values)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(
        lg.loc[~lh, LGD_FEATURES], lg.loc[~lh, "lgd"])
    lin = LinearRegression().fit(lg.loc[~lh, LGD_FEATURES], lg.loc[~lh, "lgd"])
    yt = lg.loc[lh, "lgd"]
    metrics["lgd"] = {"resolved_defaults": int(len(lg)), "mean_lgd": round(float(lg.lgd.mean()), 4),
                      "cure_share": round(float((lg.lgd == 0).mean()), 4)}
    for name, mdl in [("gbm", gbm), ("linear", lin)]:
        pr = np.clip(mdl.predict(lg.loc[lh, LGD_FEATURES]), 0, 1)
        metrics["lgd"][name] = {"rmse": round(float(np.sqrt(mean_squared_error(yt, pr))), 4),
                                "mae": round(float(mean_absolute_error(yt, pr)), 4),
                                "mean_pred": round(float(pr.mean()), 4), "mean_actual": round(float(yt.mean()), 4)}
    lgd_final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(
        lg[LGD_FEATURES], lg.lgd)
    lb = pd.cut(lg.loc[lh, "cur_ltv"], [0, 60, 80, 90, 100, 120, 250])
    lgd_curve = pd.DataFrame({"actual": yt.values, "pred": np.clip(gbm.predict(lg.loc[lh, LGD_FEATURES]), 0, 1),
                              "band": lb.values}).groupby("band", observed=True).mean()

    print("LGD models fitted", flush=True)

    # ---- origination PD for staging (12-month PD at origination under neutral macro)
    def pd12_at_origination(ids):
        s = st.loc[ids].reset_index(drop=True)
        f = dynamic_features(s, s.orig_upb.values, s.hpi_orig.values, np.zeros(len(s)), 0, 0, 0,
                             s.unemp_orig.values, 0, 0.03, 0)
        f["incentive"] = 0
        return pd_final.predict_proba(f[PD_FEATURES])[:, 1]

    m = macro.set_index("period")

    # ---- Reporting date June 2026: active portfolio, three scenarios
    R = 202606
    act = loans[loans.exit_code.isna() & (loans.last_period == R)]
    port = pd.DataFrame({"loan_id": act.loan_id.values, "vintage": act.vintage.values, "upb": act.cur_upb.values,
                         "rate": act.cur_rate.values, "age": act.cur_age.values,
                         "rem": act.cur_rem_months.fillna(act.orig_term - act.cur_age).values,
                         "dlq": act.cur_dlq_n.fillna(0).values, "forborne": act.cur_forborne.fillna(False).values,
                         "hpi_now": hpi_at(shpi, act.state.values, np.full(len(act), R))})
    u0, h0, r0 = m.unemp[R], m.hpi[R], m.mort_rate[R]
    paths26 = {
        "Baseline": make_path(u0, h0, r0, [4.4, 4.5, 4.5, 4.4], [0.02, 0.025, 0.03], [6.6, 6.3, 6.1]),
        "Upside": make_path(u0, h0, r0, [3.9, 3.8, 4.0, 4.2], [0.05, 0.04, 0.035], [6.2, 5.8, 5.8]),
        "Severe": make_path(u0, h0, r0, [8.0, 10.0, 9.0, 7.5, 6.0, 5.0], [-0.20, -0.12, 0.0, 0.02],
                            [5.8, 5.5, 5.5, 5.8]),
    }
    ecl26, _ = run_ecl(port, st, paths26, pd_final, lgd_final, pd12_at_origination(port.loan_id.values))
    ecl26.to_csv(OUT / "ecl_2026_loan_level.csv", index=False)
    by_stage = ecl26.groupby("stage").agg(loans=("ead", "size"), ead=("ead", "sum"), ecl=("ecl", "sum"))
    by_stage["coverage_pct"] = 100 * by_stage.ecl / by_stage.ead
    scen = {n: float(ecl26[f"ecl_{n}"].sum()) for n in paths26}
    metrics["ecl_2026_06"] = {
        "loans": int(len(ecl26)), "ead_musd": round(ecl26.ead.sum() / 1e6, 1),
        "ecl_weighted_musd": round(ecl26.ecl.sum() / 1e6, 3),
        "coverage_bps": round(1e4 * ecl26.ecl.sum() / ecl26.ead.sum(), 1),
        "coverage_bps_by_scenario": {n: round(1e4 * v / ecl26.ead.sum(), 1) for n, v in scen.items()},
        "stage_mix": {int(k): {"loans": int(r.loans), "ead_share": round(r.ead / ecl26.ead.sum(), 4),
                               "coverage_pct": round(r.coverage_pct, 3)} for k, r in by_stage.iterrows()},
        "weights": WEIGHTS,
    }

    print("2026 ECL done", flush=True)

    # ---- Backtest: 2007 vintage performing at Dec 2007 (never used in fitting)
    B = 200712
    bt = snaps[(snaps.vintage == BACKTEST_VINTAGE) & (snaps.period == B)]
    bport = pd.DataFrame({"loan_id": bt.loan_id.values, "vintage": BACKTEST_VINTAGE, "upb": bt.upb.values,
                          "rate": bt.rate.values, "age": bt.age.values,
                          "rem": bt.rem_months.fillna(loans.loc[bt.loan_id, "orig_term"].values - bt.age).values,
                          "dlq": bt.dlq_n.values, "forborne": bt.forborne.values,
                          "hpi_now": hpi_at(shpi, loans.loc[bt.loan_id, "state"].values, np.full(len(bt), B))})
    ub, hb, rb = m.unemp[B], m.hpi[B], m.mort_rate[B]
    realized = realized_path(macro, B)
    # realized house prices loan by loan (state index), extended at 3% a year after the data end
    bstate = loans.loc[bport.loan_id, "state"].values
    ks = np.arange(len(realized["u"]))
    lvl = np.column_stack([hpi_at(shpi, bstate, np.full(len(bport), add_months(B, 12 * k))) for k in ks])
    last_k = int(np.searchsorted(add_months(B, 12 * ks), 202606, side="right")) - 1
    lvl[:, last_k + 1:] = lvl[:, [last_k]] * 1.03 ** (ks[last_k + 1:] - last_k)
    realized["hpi_loan"] = lvl
    bpaths = {
        "Dec-2007 baseline (illustrative)": make_path(ub, hb, rb, [5.1, 5.2, 5.0], [-0.05, 0.0, 0.02], [6.0, 6.0]),
        "Realized macro": realized,
        "Severe": make_path(ub, hb, rb, [8.0, 10.0, 9.0, 7.5, 6.0], [-0.20, -0.12, 0.0, 0.02], [5.5, 5.0]),
    }
    bres = {n: project(bport, st, p, pd_final, lgd_final, discount=False) for n, p in bpaths.items()}
    bl = loans.loc[bport.loan_id.values]
    actual_loss = bl.actual_loss.where(bl.default_month > B).fillna(0).clip(lower=0)
    loss_year = (bl.default_month // 100).where(actual_loss > 0)  # losses dated at default, like the projection
    years = np.arange(2008, 2008 + 19)
    act_cum = np.array([actual_loss[loss_year <= y].sum() for y in years]) / bport.upb.sum()
    pred_cum = {n: np.cumsum(r["el_by_year"].sum(0))[: len(years)] / bport.upb.sum() for n, r in bres.items()}
    actual_default = float(((bl.default_month > B)).mean())
    metrics["backtest_2007_vintage"] = {
        "loans": int(len(bport)), "upb_musd": round(bport.upb.sum() / 1e6, 1),
        "actual_cum_loss_rate_pct": round(100 * act_cum[-1], 3),
        "actual_lifetime_default_rate_pct": round(100 * actual_default, 2),
        "pred_lifetime_loss_rate_pct": {n: round(100 * float(v[-1]), 3) for n, v in pred_cum.items()},
        "pred_lifetime_pd_pct": {n: round(100 * float(r["pd_life"].mean()), 2) for n, r in bres.items()},
    }
    # one-year-ahead check on the same vintage: actual December states, model PD for the next 12 months
    v07 = df[df.vintage == BACKTEST_VINTAGE]
    one_yr = pd.DataFrame({"year": v07.year, "pred": pd_final.predict_proba(v07[PD_FEATURES])[:, 1],
                           "act": v07.y == 1}).groupby("year").mean()
    one_yr.to_csv(OUT / "backtest_2007_one_year_pd.csv")
    metrics["backtest_2007_vintage"]["one_year_pd_2008_2012"] = {
        int(y): {"pred_pct": round(100 * r.pred, 2), "actual_pct": round(100 * r.act, 2)}
        for y, r in one_yr.loc[2007:2011].iterrows()}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- chart
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    a = ax[0, 0]
    a.plot(cal.index, 100 * cal.act, "o-", label="Actual")
    a.plot(cal.index, 100 * cal.pred, "s--", label="Predicted (logistic)")
    a.axvspan(2015.5, cal.index.max() + 0.5, color="grey", alpha=0.12, label="Out-of-time")
    a.set(title="PD: 12-month default rate by snapshot year", ylabel="%")
    a.legend(fontsize=8)
    a = ax[0, 1]
    x = np.arange(len(lgd_curve))
    a.bar(x - 0.2, 100 * lgd_curve.actual, 0.4, label="Actual")
    a.bar(x + 0.2, 100 * lgd_curve.pred, 0.4, label="Predicted (GBM)")
    a.set_xticks(x, [str(b) for b in lgd_curve.index], fontsize=8)
    a.set(title="LGD by current LTV at default (hold-out)", ylabel="Loss given default, %", xlabel="Current LTV")
    a.legend(fontsize=8)
    a = ax[1, 0]
    names = list(paths26)
    vals = [1e4 * scen[n] / ecl26.ead.sum() for n in names] + [metrics["ecl_2026_06"]["coverage_bps"]]
    a.bar(names + ["Weighted"], vals, color=["#4C72B0", "#55A868", "#C44E52", "#333333"])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    a.set(title="ECL coverage, June 2026 portfolio (bps of EAD)", ylabel="bps")
    a = ax[1, 1]
    a.plot(years, 100 * act_cum, "k-", lw=2.5, label="Actual losses")
    for n, v in pred_cum.items():
        a.plot(years, 100 * v, "--", label=n)
    a.set(title="Backtest: 2007 vintage as of Dec 2007 (out of sample)", ylabel="Cumulative loss, % of UPB")
    a.legend(fontsize=8)
    from matplotlib.ticker import MaxNLocator
    for a in (ax[0, 0], ax[1, 1]):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(OUT / "ecl_model.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
