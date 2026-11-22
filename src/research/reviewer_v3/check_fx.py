"""Check 4: FX >= 0.10 rule -- dependence on 2018-19, comparison with S > 0.15, multiple testing."""
import json

import numpy as np
import pandas as pd

from src.research.reviewer_v3.common import ROOT, CUT, MULT, RT, wboot
from src.research.fear_vs_physics import analysis as A

FV = ROOT / "data/processed/research_fear_vs_physics"
r = pd.read_pickle(FV / "panel_signals.pkl")
assert r.obs.max() <= CUT
r["pnl4"] = -(r.S_4w - r.S) * MULT - RT
r["pnlx"] = -(r.S_exp - r.S) * MULT - RT
chk = r[r.Y.isin(A.CHECK_Y)]


def stats(t, col, label):
    t = t.dropna(subset=[col])
    g = {w: d[col].values for w, d in t.groupby("winter")}
    lo, hi = wboot(g, B=4000, seed=3) if len(g) > 1 else (np.nan, np.nan)
    # one-sided bootstrap p for mean <= 0
    rng = np.random.default_rng(5)
    ws = list(g)
    bm = [np.concatenate([g[ws[i]] for i in rng.choice(len(ws), len(ws))]).mean() for _ in range(4000)] if len(ws) > 1 else [np.nan]
    pval = np.mean(np.array(bm) <= 0)
    print(f"  {label:42s} n {len(t):3d} winters {len(g)} mean {t[col].mean():7.0f} CI [{lo:6.0f},{hi:6.0f}] "
          f"p(<=0) {pval:.3f} worst {t[col].min():6.0f}  by winter sum {t.groupby('winter')[col].sum().round(0).to_dict()}")
    return t


for hold, col in (("4w", "pnl4"), ("exp", "pnlx")):
    print(f"== hold {hold}, check winters 2017-18..Dec 2021")
    fx = chk[chk.FX >= 0.10]
    s15 = chk[chk.S > 0.15]
    stats(fx, col, "FX>=0.10")
    stats(fx[fx.winter != "2018-19"], col, "FX>=0.10 ex 2018-19")
    stats(fx[~fx.winter.isin(["2018-19", "2021-22"])], col, "FX>=0.10 ex 2018-19 & truncated 2021-22")
    stats(s15, col, "S>0.15")
    stats(s15[s15.winter != "2018-19"], col, "S>0.15 ex 2018-19")
    # paired by winter: total P&L difference per winter
    a = fx.groupby("winter")[col].sum()
    b = s15.groupby("winter")[col].sum()
    d = (a.reindex(b.index.union(a.index)).fillna(0) - b.reindex(b.index.union(a.index)).fillna(0))
    print("  per-winter total FX - S>0.15:", d.round(0).to_dict())
    only_s = s15[~s15.origin.isin(fx.origin)]
    only_f = fx[~fx.origin.isin(s15.origin)]
    print(f"  weeks only in S>0.15: {len(only_s)} (sum {only_s[col].sum():.0f}; dates {list(only_s.obs.dt.date.astype(str))[:8]})")
    print(f"  weeks only in FX:     {len(only_f)} (sum {only_f[col].sum():.0f})")

# train window too (2012-13..2016-17)
tr = r[r.Y.isin(range(2013, 2018))]
print("== train 4w FX>=0.10 by winter:", tr[tr.FX >= 0.10].groupby("winter").pnl4.agg(["count", "sum"]).to_dict())

# frozen full-sample PP applied to the whole dev panel (in-sample; shows the 2014 fragility)
fz = json.load(open(FV / "frozen_holdout.json"))
q = r.assign(neg_dev_now=-r.dev_now, neg_x_an=-r.x_an)
ph = pd.concat([(q[c] - fz["z"][c]["mean"]) / fz["z"][c]["sd"] for c in A.PHYS], axis=1).mean(axis=1)
q["FXf"] = q.S - (fz["const"] + fz["wte"] * q.wte + fz["Phys"] * ph)
for Ys, lab in ((range(2011, 2018), "2010-11..2016-17"), (A.CHECK_Y, "check")):
    t = q[q.Y.isin(Ys) & (q.FXf >= 0.10)].dropna(subset=["pnl4"])
    print(f"== frozen-PP FX>=0.10, 4w, {lab}: n {len(t)} mean {t.pnl4.mean():.0f} worst {t.pnl4.min():.0f} "
          f"on {t.loc[t.pnl4.idxmin(), 'obs'].date()}")

# rough multiple-testing count: check-period cells inspected before the FX family was declared
grids = {k: pd.read_csv(FV / f) for k, f in (("A", "grid_A_rv4_ptail.csv"), ("B", "grid_B_FP.csv"),
                                              ("C", "grid_C_FX.csv"), ("T", "grid_T_TX.csv"))}
n = {k: int((g.period == "check").sum()) for k, g in grids.items()}
print("check cells per grid:", n, "total", sum(n.values()))
fxr = pd.read_csv(FV / "fx_robustness.csv")
print("fx_robustness rows:", len(fxr))
