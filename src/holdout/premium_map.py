"""Frozen holdout test V2 (agent premium_map): March-April drift from 9 months out to expiry.

Spec (frozen in reports/agents/premium_map.md section 6; not changed after seeing results):
  * Pairs: Mar/Apr 2023, 2024, 2025, 2026 (primary); Mar/Apr 2022 reported separately (entry in 2021).
  * Entry: settle on the last trading day on or before (March-contract expiry - 9 calendar months).
  * D = (Mar - Apr at entry) - (Mar final settle - Apr settle on the March expiry day), $/MMBtu,
    and D / mean(Mar, Apr at entry) in %.
  * Short 1 Mar-Apr spread at entry, hold to expiry. Net = D * 10,000 - $160 (4 ticks/contract/side).
  * PASS iff mean net > 0 AND D > 0 in >= 3 of 4 primary winters; else FAIL.
  * Also reported (descriptive, requested by the coordinator): worst mark-to-market P&L of the short
    vs entry during the hold, and the worst peak-to-trough drawdown, both before costs.

Usage:
  python -m src.holdout.premium_map --end 2021-12-31   # development check (Mar 2011..2021)
  python -m src.holdout.premium_map --holdout          # the one holdout run
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "research_premium_map"
MULT, COST = 10_000, 160.0
PRIMARY = [2023, 2024, 2025, 2026]


def load(end):
    d = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    d = d[(d.date <= pd.Timestamp(end)) & d.settle.notna() & (d.settle > 0)]
    d = d[d.contract_month.isin([3, 4])]
    W = d.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle", aggfunc="last")
    return W.sort_index()


def winter(W, y, end):
    if (y, 3) not in W.columns or (y, 4) not in W.columns:
        return None
    mar = W[(y, 3)].dropna()
    T = mar.index.max()
    # expired inside the data window: delivery month starts after T and data continues beyond T
    if T >= pd.Timestamp(end) - pd.Timedelta(days=3) or T < pd.Timestamp(y, 2, 15):
        return None
    target = T - pd.DateOffset(months=9)
    days = W.index[(W.index <= target)]
    days = [d for d in days[::-1] if not np.isnan(W.at[d, (y, 3)]) and not np.isnan(W.at[d, (y, 4)])]
    if not days:
        return None
    t0 = days[0]
    S = (W[(y, 3)] - W[(y, 4)]).loc[t0:T].dropna()
    D = S.iloc[0] - S.iloc[-1]
    px = (W.at[t0, (y, 3)] + W.at[t0, (y, 4)]) / 2
    mtm = (S.iloc[0] - S) * MULT                      # short P&L vs entry, before costs
    dd = mtm - mtm.cummax()
    return dict(winter=f"{y - 1}-{str(y)[2:]}", mar_year=y, entry=t0.date(), expiry=T.date(),
                S_entry=S.iloc[0], S_expiry=S.iloc[-1], D=D, D_pct=100 * D / px, net=D * MULT - COST,
                worst_mtm=mtm.min(), worst_mtm_date=mtm.idxmin().date(),
                max_drawdown=dd.min(), days=len(S))


def run(end, years):
    W = load(end)
    rows = [r for y in years if (r := winter(W, y, end)) is not None]
    return pd.DataFrame(rows)


def verdict(df):
    p = df[df.mar_year.isin(PRIMARY)]
    ok = len(p) == 4
    passed = ok and p.net.mean() > 0 and (p.D > 0).sum() >= 3
    return p, ("PASS" if passed else "FAIL") if ok else "INCOMPLETE"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    if not a.holdout:
        assert pd.Timestamp(a.end) < pd.Timestamp("2022-01-01")
        df = run(a.end, range(2011, 2023))
        print(df.round(4).to_string(index=False))
        print(f"\nmean D {df.D.mean() * 100:+.1f}c ({df.D_pct.mean():+.1f}%), D>0 in {(df.D > 0).sum()}/{len(df)}, "
              f"mean net ${df.net.mean():,.0f}")
        df.to_csv(OUT / "holdout_v2_devcheck.csv", index=False)
    else:
        df = run("2026-11-17", range(2022, 2027))
        df.to_csv(OUT / "holdout_v2_results.csv", index=False)
        print(df.round(4).to_string(index=False))
        p, v = verdict(df)
        print(f"\nPRIMARY (Mar 2023-2026): mean D {p.D.mean() * 100:+.1f}c ({p.D_pct.mean():+.1f}%), "
              f"D>0 in {(p.D > 0).sum()}/{len(p)}, mean net ${p.net.mean():,.0f}, total ${p.net.sum():,.0f} -> {v}")
