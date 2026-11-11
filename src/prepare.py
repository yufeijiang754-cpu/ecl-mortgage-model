"""Build model-ready tables from the Freddie Mac Single-Family Loan-Level Dataset (sample files).

Outputs (data/processed/, parquet):
  loans.parquet      one row per loan: origination attributes + outcome (default / prepay / loss)
  snapshots.parquet  one row per loan per December while the loan is performing:
                     state at the snapshot + what happened in the next 12 months (default, prepay)
  macro.parquet      monthly unemployment rate and national house price index

Definitions
  Default   first month with 90+ days delinquent (status >= 3 or REO acquisition) outside a
            forbearance plan, or a credit-event zero balance (02 third-party sale, 03 short sale /
            charge-off, 09 REO).
  Prepay    zero balance code 01 (prepaid or matured) with no earlier default.
  Censored  other exits (15 whole-loan sale, 16 reperforming sale, 96 defect) or less than
            12 months of data after the snapshot.

Field positions follow the July 2026 General User Guide (Release 47): 31 origination fields,
35 monthly performance fields.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
VINTAGES = [2005, 2006, 2007, 2010, 2014, 2018]
DATA_END = 202606

ORIG_COLS = ["credit_score", "first_pay", "first_time", "maturity", "msa", "mi_pct", "units", "occupancy",
             "cltv", "dti", "orig_upb", "ltv", "orig_rate", "channel", "ppm", "amort", "state", "prop_type",
             "zip3", "loan_id", "purpose", "orig_term", "n_borrowers", "seller", "super_conf", "pre_harp",
             "program", "harp", "valuation", "io", "vantage"]
# performance file: 0-based positions of the fields we use
PERF_USE = {0: "loan_id", 1: "period", 2: "upb", 3: "dlq", 4: "age", 5: "rem_months", 8: "zb_code",
            10: "rate", 13: "mi_rec", 14: "net_sales", 15: "non_mi_rec", 16: "expenses", 21: "actual_loss",
            26: "zb_upb", 29: "assist"}
JUDICIAL = {"CT", "DE", "FL", "HI", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "NJ", "NM", "NY", "ND", "OH",
            "OK", "PA", "SC", "SD", "VT", "WI"}


def add_months(yyyymm, n):
    y, m = np.divmod(np.asarray(yyyymm, dtype=np.int64), 100)
    t = y * 12 + (m - 1) + n
    return (t // 12) * 100 + t % 12 + 1


def load_macro():
    u = pd.read_csv(ROOT / "data/macro/UNRATE.csv", names=["date", "unemp"], header=0, parse_dates=["date"])
    h = pd.read_csv(ROOT / "data/macro/CSUSHPINSA.csv", names=["date", "hpi"], header=0, parse_dates=["date"])
    r = pd.read_csv(ROOT / "data/macro/MORTGAGE30US.csv", names=["date", "mort_rate"], header=0,
                    parse_dates=["date"])
    r = r.set_index("date").mort_rate.resample("MS").mean().reset_index()
    m = u.merge(h, on="date", how="outer").merge(r, on="date", how="outer").sort_values("date")
    m["period"] = m.date.dt.year * 100 + m.date.dt.month
    m[["hpi", "mort_rate"]] = m[["hpi", "mort_rate"]].ffill()
    return m.set_index("period")[["unemp", "hpi", "mort_rate"]].dropna()


def vintage(year, macro):
    o = pd.read_csv(RAW / f"sample_orig_{year}.txt", sep="|", header=None, names=ORIG_COLS,
                    dtype={"loan_id": str, "zip3": str, "msa": str, "seller": str})
    o["vintage"] = year
    p = pd.read_csv(RAW / f"sample_perf_{year}.txt", sep="|", header=None, usecols=list(PERF_USE),
                    dtype={0: str, 3: str, 8: str, 29: str})
    p.columns = [PERF_USE[c] for c in sorted(PERF_USE)]
    p["dlq_n"] = pd.to_numeric(p.dlq.replace("RA", "99"), errors="coerce")
    p["zb"] = pd.to_numeric(p.zb_code, errors="coerce")
    p["forborne"] = p.assist.eq("F")

    # outcome per loan; delinquency while in a forbearance plan (COVID-19) is not treated as default
    is_def = ((p.dlq_n >= 3) & ~p.forborne) | p.zb.isin([2, 3, 9])
    default_month = p.loc[is_def].groupby("loan_id").period.min()
    zb_rows = p.loc[p.zb.notna()].groupby("loan_id").last()
    loans = o.set_index("loan_id")
    loans["default_month"] = default_month
    loans["exit_month"] = zb_rows.period
    loans["exit_code"] = zb_rows.zb
    loans["last_period"] = p.groupby("loan_id").period.max()
    loans["last_dlq"] = p.groupby("loan_id").dlq_n.last()
    for c in ["actual_loss", "zb_upb", "net_sales", "mi_rec", "non_mi_rec", "expenses"]:
        loans[c] = zb_rows[c]
    last = p.groupby("loan_id").last()
    for c in ["upb", "dlq_n", "forborne", "age", "rem_months", "rate"]:
        loans[f"cur_{c}"] = last[c]
    loans["prepay_month"] =np.where((loans.exit_code == 1) & loans.default_month.isna(), loans.exit_month, np.nan)
    upb_at_def = p.merge(default_month.rename("dm"), left_on="loan_id", right_index=True)
    upb_at_def = upb_at_def[upb_at_def.period == add_months(upb_at_def.dm, -1)].set_index("loan_id").upb
    loans["upb_at_default"] = upb_at_def
    loans["upb_at_default"] = loans.upb_at_default.fillna(loans.zb_upb)

    # December snapshots of performing loans
    s = p.loc[(p.period % 100 == 12) & ((p.dlq_n < 3) | p.forborne) & p.upb.gt(0)].copy()
    s = s.merge(loans[["default_month", "prepay_month", "exit_month", "exit_code"]], left_on="loan_id",
                right_index=True)
    s = s[s.default_month.isna() | (s.period < s.default_month)]
    s = s[s.exit_month.isna() | (s.period < s.exit_month)]
    horizon = add_months(s.period, 12)
    s["y_default"] = (s.default_month <= horizon).astype(int)
    s["y_prepay"] = ((s.prepay_month <= horizon) & (s.y_default == 0)).astype(int)
    other_exit = s.exit_code.isin([15, 16, 96]) & (s.exit_month <= horizon) & (s.y_default == 0)
    s = s[(horizon <= DATA_END) & ~other_exit]
    s["vintage"] = year
    keep = ["loan_id", "vintage", "period", "upb", "dlq_n", "forborne", "age", "rem_months", "rate",
            "y_default", "y_prepay"]
    return loans.reset_index(), s[keep]


def main():
    macro = load_macro()
    loans, snaps = zip(*(vintage(y, macro) for y in VINTAGES))
    loans, snaps = pd.concat(loans, ignore_index=True), pd.concat(snaps, ignore_index=True)

    # origination property value and house price index at origination (month before first payment)
    loans["orig_period"] = add_months(loans.first_pay, -2)
    loans["hpi_orig"] = macro.hpi.reindex(loans.orig_period).values
    loans["unemp_orig"] = macro.unemp.reindex(loans.orig_period).values
    loans["prop_value"] = loans.orig_upb / (loans.ltv.where(loans.ltv.between(1, 200)) / 100)
    loans["judicial"] = loans.state.isin(JUDICIAL).astype(int)
    loans.to_parquet(OUT / "loans.parquet", index=False)
    snaps.to_parquet(OUT / "snapshots.parquet", index=False)
    macro.reset_index().to_parquet(OUT / "macro.parquet", index=False)

    print(f"loans {len(loans):,} | snapshots {len(snaps):,} | defaults {loans.default_month.notna().sum():,}")
    print(snaps.groupby("vintage")[["y_default", "y_prepay"]].mean().round(4))


if __name__ == "__main__":
    main()
