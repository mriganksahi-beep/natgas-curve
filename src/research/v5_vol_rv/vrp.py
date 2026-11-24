"""Step 1: the VRP map (descriptive; no fitting).  gap = atm_iv - realised vol of the same contract
over (a) the option's remaining life and (b) the next 20 trading days; and delta-hedged straddle
P&L (dh_straddle) per unit vega and per premium.  CIs by gas-year block bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import base, data as D
from .stats import table, block_mean_ci

TEN_BINS = [1, 21, 63, 126, 252, 400]
TEN_LAB = ["<1m", "1-3m", "3-6m", "6-12m", "12-18m"]


def window(d):
    return np.where(pd.DatetimeIndex(d).year <= 2016, "choose", "check")


def prep_base():
    b = base.build()
    b["ten"] = pd.cut(b.tdays, TEN_BINS, labels=TEN_LAB)
    b["gap_life"] = b.atm_iv - b.rv_life
    b["gap20"] = b.atm_iv - b.rv20f
    b["lratio_life"] = np.log(b.atm_iv / b.rv_life)
    b["tseason"] = np.where(b.date.dt.month.isin([11, 12, 1, 2, 3]), "Nov-Mar", "Apr-Oct")
    b["stor"] = pd.cut(b.sdev, [-1e9, -150, 150, 1e9], labels=["tight(<-150)", "normal", "loose(>150)"])
    b["pxlvl"] = pd.cut(b.F, [0, 3, 4, 99], labels=["F<=3", "3-4", ">4"])
    b["win"] = window(b.date)
    return b


def prep_dh(b):
    d = D.dh()
    d = d.merge(b[["date", "cid", "tdays", "ten", "dclass", "dm", "tseason", "stor", "pxlvl", "gy", "win"]],
                on=["date", "cid"], how="inner")
    d["pv"] = d.pnl / d.vega0 * 100               # vol points per day per unit vega
    d["carry_v"] = (d.gamma_pnl + d.theta_pnl) / d.vega0 * 100
    d["vega_v"] = d.vega_pnl / d.vega0 * 100
    d["pp"] = d.pnl / d.P0 * 100                  # % of premium per day
    return d


def run():
    b = prep_base()
    d = prep_dh(b)
    out = {}
    # (a) gap over life and next-20, by delivery class x tenor (vol points)
    for col in ["gap_life", "gap20"]:
        t = table(b, col, ["dclass", "ten"], scale=100)
        t["col"] = col
        out[f"map_{col}"] = t
    out["by_month_life"] = table(b, "gap_life", "dm", scale=100)
    out["by_month_ten_life"] = table(b, "gap_life", ["dm", "ten"], scale=100)
    for by in ["tseason", "stor", "pxlvl", "win"]:
        out[f"life_{by}"] = table(b, "gap_life", by, scale=100)
        out[f"life_{by}_cls"] = table(b, "gap_life", [by, "dclass"], scale=100)
    out["ratio_cls_ten"] = table(b, "lratio_life", ["dclass", "ten"])
    # (b) delta-hedged straddle P&L, per unit vega (vol pts/day) and per premium (%/day)
    for col in ["pv", "carry_v", "vega_v", "pp"]:
        out[f"dh_{col}_cls_ten"] = table(d, col, ["dclass", "ten"])
    for by in ["tseason", "stor", "pxlvl", "win"]:
        out[f"dh_pv_{by}"] = table(d, "pv", [by, "dclass"])
        out[f"dh_pp_{by}"] = table(d, "pp", [by, "dclass"])
    out["dh_pv_month"] = table(d, "pv", "dm")
    out["dh_overall"] = pd.DataFrame([dict(zip(["mean", "lo", "hi", "k"], block_mean_ci(d[c], d.gy)), col=c)
                                      for c in ["pv", "carry_v", "vega_v", "pp"]])
    # per-gas-year mean gap (life) for winter contracts <3m: how many years positive
    gy = b[b.tdays <= 63].groupby(["gy", "dclass"]).gap_life.mean().unstack() * 100
    out["gy_short_tenor"] = gy.reset_index()
    for k, v in out.items():
        v.to_csv(D.CACHE / f"s1_{k}.csv", index=False)
    return out, b, d


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out, b, d = run()
    for k in ["map_gap_life", "map_gap20", "by_month_life", "life_tseason_cls", "life_stor_cls", "life_pxlvl_cls",
              "life_win", "dh_overall", "dh_pv_cls_ten", "dh_carry_v_cls_ten", "dh_vega_v_cls_ten",
              "dh_pp_cls_ten", "dh_pv_tseason", "dh_pv_stor", "dh_pv_pxlvl", "dh_pv_win", "dh_pv_month", "gy_short_tenor"]:
        print("==", k)
        print(out[k].round(3).to_string(index=False))
