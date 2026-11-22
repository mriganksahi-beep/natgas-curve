"""Frozen test V3 (storage_space_option, price only): autumn Nov-Oct contango drift.

Spec (frozen in reports/agents/storage_space_option.md, section 6):
  * C = Nov(Y) - Oct(Y) settle ($/MMBtu); D_Y = C at October's last trading day - mean C over
    trading days 1 Apr - 31 Aug of Y.
  * Net P&L: short one contango spread at each of those days' settles, hold to October's last
    trade, 4 ticks/contract/side ($160 per spread round trip); averaged within each year.
  * Pass: D_Y < 0 in >= 4 of 5 years AND mean yearly net > 0.
    Fail: < 3 negative years OR mean yearly net < -$500.  Otherwise inconclusive.
  * Flag any year with D_Y > +$0.15 as a tail event.
Prices only (no storage, no Model 2).

Run:
  python -m src.holdout.storage_space_option --start 2010 --end 2021-12-31   # dev check
  python -m src.holdout.storage_space_option --start 2022 --end 2026-11-17   # holdout, once
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MULT, COST = 10_000, 2 * 2 * 4 * 10.0
TAIL = 0.15


def run(start_year, end):
    end = pd.Timestamp(end)
    d = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    d = d[(d.date <= end) & d.settle.notna() & (d.settle > 0)]
    d = d[d.contract_month.isin([10, 11]) & (d.contract_year >= start_year)]
    rows = []
    for y in sorted(d.contract_year.unique()):
        v = d[(d.contract_year == y) & (d.contract_month == 10)].set_index("date").settle
        x = d[(d.contract_year == y) & (d.contract_month == 11)].set_index("date").settle
        if v.empty or x.empty:
            continue
        last = v.index.max()
        # October must have expired inside the data (last trade in September of Y, before end)
        if not (last.year == y and last.month == 9 and last < end - pd.Timedelta(days=5)):
            continue
        c = (x - v).dropna()
        c = c[c.index <= last]
        win = c[(c.index >= pd.Timestamp(y, 4, 1)) & (c.index <= pd.Timestamp(y, 8, 31))]
        if win.empty or last not in c.index:
            continue
        exp = c[last]
        pnl = -(exp - win) * MULT - COST
        rows.append({"year": y, "last_trade": last.date(), "n_entry_days": len(win),
                     "summer_avg_c": 100 * win.mean(), "expiry_c": 100 * exp,
                     "D_c": 100 * (exp - win.mean()), "net_pnl_$": pnl.mean(),
                     "tail": bool(exp - win.mean() > TAIL)})
    return pd.DataFrame(rows)


def verdict(r):
    n_neg, mean_net = int((r.D_c < 0).sum()), r["net_pnl_$"].mean()
    if n_neg >= 4 and mean_net > 0:
        v = "PASS"
    elif n_neg < 3 or mean_net < -500:
        v = "FAIL"
    else:
        v = "INCONCLUSIVE"
    return {"years": len(r), "n_D_neg": n_neg, "mean_D_c": r.D_c.mean(), "median_D_c": r.D_c.median(),
            "mean_yearly_net_$": mean_net, "tail_years": r.year[r["tail"]].tolist(), "verdict": v}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    r = run(a.start, a.end)
    pd.set_option("display.width", 200)
    print(r.round(2).to_string(index=False))
    print(verdict(r))
