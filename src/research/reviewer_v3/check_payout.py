"""Check 5: robustness of widowmaker's actuarial value (AV) to the 11-point payout curve."""
import numpy as np
import pandas as pd

from src.research.widowmaker_insurance import data as D, payout as P, analysis as A

sp = D.spreads()
panel = D.weekly_panel(sp)
m2 = D.load_m2()
assert m2.origin.max() <= D.CUTOFF and panel.obs.max() <= D.CUTOFF
tab = A.payout_table()
print(tab[["x", "S_exp", "S_exp5"]].round(3).to_string())

variants = {}
cd, t = P.curves(tab, "S_exp", False, 1, B=200)
variants["primary convex (WF)"] = cd
# drop 2014 from every fit
tab_no14 = tab.drop(2014)
cd2, _ = P.curves(tab_no14, "S_exp", False, 1, B=200)
variants["convex, 2014 excluded"] = cd2
# near-linear: tau fixed at 3000
g0 = P.TAU_GRID
P.TAU_GRID = np.array([3000.0])
variants["near-linear (tau=3000)"] = P.curves(tab, "S_exp", False, 1, B=200)[0]
# strongly convex: tau fixed at the lower bound
P.TAU_GRID = np.array([300.0])
variants["very convex (tau=300)"] = P.curves(tab, "S_exp", False, 1, B=200)[0]
P.TAU_GRID = g0
# 2014 payout measured as last-5-day average (0.78) instead of expiry day
tab5 = tab.copy()
tab5.loc[2014, "S_exp"] = tab.loc[2014, "S_exp5"]
variants["convex, 2014 payout = 5-day avg"] = P.curves(tab5, "S_exp", False, 1, B=200)[0]

rows = []
for nm, c in variants.items():
    r = A.actuarial(panel, m2, c, "5y", "wf")
    rows.append({"variant": nm, "mean_AV": r.av.mean(), "sd_AV": r.av.std(), "max_AV": r.av.max(),
                 "mean_prem": r.prem.mean(), "share_prem>0": (r.prem > 0).mean(),
                 "mean_p_large": r.p_large.mean()})
    if nm.startswith("primary"):
        base = r
print(pd.DataFrame(rows).round(3).to_string())
# constant benchmark: mean of earlier expiries
cm = {Y: tab.S_exp[tab.index < Y].mean() for Y in base.Y.unique()}
print("constant benchmark AV (mean past expiries): mean %.3f; mean premium vs it %.3f" %
      (base.Y.map(cm).mean(), (base.S - base.Y.map(cm)).mean()))
print("mean S_exp 2011-21 %.3f, ex-2014 %.3f" % (tab.S_exp.loc[2011:2021].mean(), tab.S_exp.loc[2011:2021].drop(2014).mean()))
print("corr(AV, S) over weeks: %.2f; AV range by winter:" % np.corrcoef(base.av, base.S)[0, 1],
      base.groupby("winter").av.mean().round(3).to_dict())
# break-even disaster probability: premium / (2014 payout - typical payout)
pr = base.prem.mean()
print("break-even probability of a 2014-size expiry (0.31 vs -0.05): %.2f" % (pr / (0.314 + 0.05)))
