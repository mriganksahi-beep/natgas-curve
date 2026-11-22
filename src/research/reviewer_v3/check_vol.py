"""Check 3: is 'fear' (trailing spread vol) mechanically the spread level?  Uses cached dev panels only."""
import numpy as np
import pandas as pd

from src.research.reviewer_v3.common import ROOT, CUT
from src.research.fear_vs_physics import analysis as A

FV = ROOT / "data/processed/research_fear_vs_physics"
raw = pd.read_pickle(FV / "panel.pkl")
assert raw.obs.max() <= CUT
sp = pd.read_pickle(FV / "spreads.pkl")
fv = pd.read_pickle(FV / "fvol.pkl")
assert max(s.index.max() for s in sp.values()) <= CUT and fv.index.max() <= CUT
FLOOR = -0.15  # approx. ample-storage floor of H-J (widowmaker a=-0.13, scenario_risk a in [-0.2, 0.1])

# level-normalised spread vol: sd of dS_t / (S_{t-1} - floor) over trailing 20 days
rvn = []
for _, r in raw.iterrows():
    S = sp[r.Y]
    S = S[S.index <= r.obs]
    z = (S.diff() / np.maximum(S.shift(1) - FLOOR, 0.02)).iloc[-20:]
    rvn.append(z.std())
raw = raw.assign(rv4n=rvn)
p = raw.assign(neg_dev_now=-raw.dev_now, neg_x_an=-raw.x_an)
print("corr with S:", p[["rv4", "rv4n", "fvol", "maxrise", "p_tail"]].corrwith(p.S).round(2).to_dict())


def z(x):
    return (x - x.mean()) / x.std()


p["Phys"] = pd.concat([z(p[c]) for c in A.PHYS], axis=1).mean(axis=1)
fear_defs = {"Fear (5 measures, as reported)": A.FEAR, "rv4 only": ["rv4"], "fvol only (front-month %)": ["fvol"],
             "rv4 normalised by (S - floor)": ["rv4n"], "fvol + rv4n": ["fvol", "rv4n"]}
print("\nLevel R2 of S ~ wte + X  (n=%d)" % len(p))
(b, r2), _ = A.fit(p, "S", ["wte", "Phys"])
print(f"  Phys: R2 {r2:.2f}")
for nm, cols in fear_defs.items():
    p["Fear"] = pd.concat([z(p[c]) for c in cols], axis=1).mean(axis=1)
    (b1, r1), _ = A.fit(p, "S", ["wte", "Fear"])
    (b2, r2b), _ = A.fit(p, "S", ["wte", "Fear", "Phys"])
    dec, bl = A.decompose(p)
    res = {}
    for y in ("dS_exp", "dS_4w"):
        t = A.boot(dec, y, ["S_fear", "S_phys", "S_time", "S_res"], B=500)
        res[y] = (t.loc["S_fear", ["coef", "lo", "hi"]].values, t.loc["S_phys", ["coef", "lo", "hi"]].values)
    g = A.boot(dec, "dS_exp", ["S", "Fear", "Phys"], B=500)
    print(f"  {nm:34s} R2 fear-only {r1:.2f}; +Phys {r2b:.2f}; level coef fear {bl[2]:.3f} phys {bl[3]:.3f}; "
          f"beta_F(exp) {res['dS_exp'][0].round(2)} beta_P(exp) {res['dS_exp'][1].round(2)}; "
          f"given S: Fear {g.loc['Fear', ['coef','lo','hi']].values.round(3)} Phys {g.loc['Phys', ['coef','lo','hi']].values.round(3)}")

# all components decay ~ -1? (reported decomposition)
p["Fear"] = pd.concat([z(p[c]) for c in A.FEAR], axis=1).mean(axis=1)
dec, _ = A.decompose(p)
t = A.boot(dec, "dS_exp", ["S"], B=500)
print("\nS alone slope on dS_exp:", t.loc["S", ["coef", "lo", "hi"]].values.round(2))

# ---------------- premium_map: Mar-Apr L3-12 drift (% of price) on log rv13, with level / normalised vol
from src.research.premium_map.stats import ols_cluster
P = pd.read_pickle(ROOT / "data/processed/research_premium_map/drift_panel.pkl")
assert P.t.max() <= CUT
g = P[(P.pair == "Mar-Apr") & P.L.isin(range(3, 13)) & P.dev.notna() & P.rv13.notna()].copy()
fvw = fv.reindex(pd.to_datetime(g.t), method="ffill").values
g["fvol"] = fvw
g["rvn"] = g.rv13 / np.maximum(g.V - FLOOR, 0.02)
print("\npremium_map Mar-Apr L3-12: n=%d, years %d; corr(log rv13, V) = %.2f, corr(log rvn, V) = %.2f, corr(fvol,V)=%.2f"
      % (len(g), g.gy.nunique(), np.corrcoef(np.log(g.rv13 + 1e-4), g.V)[0, 1],
         np.corrcoef(np.log(g.rvn), g.V)[0, 1], np.corrcoef(g.fvol, g.V)[0, 1]))
fe = [(g.L == l).astype(float) for l in sorted(g.L.unique())[1:]]
specs = {"as reported: dev, aprpx, log rv13": [z(g.dev), z(g.apr_px), z(np.log(g.rv13 + 1e-4))],
         "+ spread level V": [z(g.dev), z(g.apr_px), z(np.log(g.rv13 + 1e-4)), z(g.V)],
         "log rv13/(V-floor) instead": [z(g.dev), z(g.apr_px), z(np.log(g.rvn))],
         "front-month vol instead": [z(g.dev), z(g.apr_px), z(g.fvol)],
         "spread level V only (no vol)": [z(g.dev), z(g.apr_px), z(g.V)]}
for nm, xs in specs.items():
    b, lo, hi, _ = ols_cluster(g.pct.values, np.column_stack(xs + fe), g.gy.values, nboot=1000)
    print(f"  {nm:40s} " + "  ".join(f"x{i}: {b[i]:+.1f} [{lo[i]:+.1f},{hi[i]:+.1f}]" for i in range(1, len(xs) + 1)))
