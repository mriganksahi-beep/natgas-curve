"""Test 1 (+ test 4 slopes): does premium_part predict later GEFS revisions and realised weather
beyond the GEFS forecast at d?  Descriptive pooled fits (HAC + winter bootstrap) and walk-forward
OOS incremental R^2 (expanding by winter). Writes t1_daily.csv, t1_weekly.csv, t4_slopes.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CACHE, INS, daily, weekly, incr_test

LVL = ["lvl_w1", "lvl_w2", "lvl_w3"]
NEWS_D = ["news_w0", "news_w1", "news_w2", "news_w3", "news_tot_l1"]
FN_T = ["FN1", "FN3", "FN5", "FN5_w1", "FN5_w3"]
REAL_T = ["real_w1", "real_w2", "real_w3", "real_tot"]


def run_daily(p: pd.DataFrame, sub=None, tag="all") -> pd.DataFrame:
    rows = []
    for name in INS:
        d = p[p[f"in_core_{name}"] == True]  # noqa: E712
        if sub is not None:
            d = d[sub(d)]
        for x in (f"P_{name}", f"P5_{name}"):
            for y in FN_T:
                rows.append({"ins": name, "kind": "future_gefs", **incr_test(d, y, LVL + NEWS_D, [x], 5)})
            for y in REAL_T:
                rows.append({"ins": name, "kind": "realised", **incr_test(d, y, LVL, [x], 15)})
            # beyond the NEXT day's 00Z run: realised over the d+1 run's windows, given that run's levels
            rows.append({"ins": name, "kind": "realised_vs_next_run",
                         **incr_test(d, "real_tot_n1", ["lvl_w1_n1", "lvl_w2_n1", "lvl_w3_n1"], [x], 15)})
    out = pd.DataFrame(rows)
    out["sample"] = tag
    return out


def run_weekly(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ins in INS.items():
        d = w[(w.instrument == ins) & (w.in_core == True)]  # noqa: E712
        for y in ["FN5", "FN5_w1", "FN5_w3"]:
            rows.append({"ins": name, "kind": "future_gefs",
                         **incr_test(d, y, LVL + ["news_tot", "news_week"], ["premium_part"], 2)})
        for y in REAL_T:
            rows.append({"ins": name, "kind": "realised", **incr_test(d, y, LVL, ["premium_part"], 3)})
    return pd.DataFrame(rows)


def run_asym(p: pd.DataFrame) -> pd.DataFrame:
    """Separate slopes for up (P>0) and down (P<0) premium days."""
    rows = []
    for name in INS:
        d = p[p[f"in_core_{name}"] == True].copy()  # noqa: E712
        d["Pup"] = d[f"P_{name}"].clip(lower=0)
        d["Pdn"] = d[f"P_{name}"].clip(upper=0)
        for y, base, lag in [("FN5", LVL + NEWS_D, 5), ("real_tot", LVL, 15)]:
            for a, b in [("Pup", "Pdn"), ("Pdn", "Pup")]:
                r = incr_test(d, y, base + [b], [a], lag)
                rows.append({"ins": name, "side": "up" if a == "Pup" else "down", **r})
    return pd.DataFrame(rows)


def main():
    p = daily()
    t = run_daily(p)
    t2 = run_daily(p, sub=lambda d: d.gefs_system == "reforecast_v12", tag="reforecast_only")
    pd.concat([t, t2]).to_csv(CACHE / "t1_daily.csv", index=False, float_format="%.5g")
    tw = run_weekly(weekly())
    tw.to_csv(CACHE / "t1_weekly.csv", index=False, float_format="%.5g")
    ta = run_asym(p)
    ta.to_csv(CACHE / "t4_slopes.csv", index=False, float_format="%.5g")
    cols = ["ins", "x", "y", "n", "coef_sd", "t_hac", "ci_lo", "ci_hi", "dR2_in", "oos_dR2_choose", "oos_dR2_check"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(t[cols].round(3)); print(t2[cols].round(3).to_string())
        print(tw[cols].round(3)); print(ta[["ins", "side"] + cols[2:]].round(3))


if __name__ == "__main__":
    main()
