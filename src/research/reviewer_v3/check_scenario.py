"""Check 6: PHYS-M2 calibration claims vs a level-only twin; Kupiec power at n=79."""
import numpy as np
import pandas as pd
from scipy import stats

from src.research.reviewer_v3.common import ROOT, CUT, MULT

fc = pd.read_pickle(ROOT / "data/processed/research_scenario_risk/forecasts.pkl")
assert fc.date.max() <= CUT
h1 = fc[fc.h == 1]
m2dates = set(h1[h1.model == "PHYS-M2"].date)
d = h1[h1.date.isin(m2dates)].copy()
print("M2 sample weeks:", len(m2dates))


def es(q):
    return float(np.mean(q[975:]))


def kupiec(x, n, p=0.01):
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return 1 - stats.chi2.cdf(lr, 1)


print("Kupiec p at n=79 for 0..5 breaches:", [round(kupiec(x, 79), 3) for x in range(6)])

phys = d[d.model == "PHYS-M2"].sort_values("date").reset_index(drop=True)
phys["k"] = np.maximum(phys.S - phys.a, 0.005)
# LEVEL twin: PHYS-M2's own earlier forecasts, normalised by (S - a), averaged, rescaled by today's (S - a).
# Model 2's week-specific dispersion is thrown away; only the spread level moves the forecast.
Q = np.stack(phys.q.values)
lev = []
for i in range(len(phys)):
    if i < 10:
        lev.append(None)
        continue
    norm = Q[:i] / phys.k.values[:i, None]
    lev.append(np.median(norm, axis=0) * phys.k.values[i])
phys["q_lev"] = lev


def evaluate(qs, loss):
    br99 = sum(l > q[989] for q, l in zip(qs, loss))
    br95 = sum(l > q[949] for q, l in zip(qs, loss))
    v975 = np.array([q[974] for q in qs])
    E = np.array([es(q) for q in qs])
    z2 = np.mean(loss * (loss > v975) / (0.025 * E)) - 1
    return br95, br99, z2, E.mean() * MULT


sub = phys[phys.q_lev.notna()]
ew = d[d.model == "EWMA"].set_index("date").q
hy = d[d.model == "HYB"].set_index("date").q
for nm, qs in (("PHYS-M2", sub.q.values), ("LEVEL twin", sub.q_lev.values),
               ("EWMA", ew.reindex(sub.date).values), ("HYB", hy.reindex(sub.date).values)):
    b95, b99, z2, me = evaluate(list(qs), sub.loss.values)
    print(f"{nm:10s} n={len(sub)} breaches95 {b95} breaches99 {b99} Z2 {z2:.2f} meanES ${me:,.0f}")

# 9 Nov 2018: ES and alarm (ES / median earlier ES) for PHYS-M2, LEVEL twin, and S - a itself
t = pd.Timestamp("2018-11-09")
i = phys.index[phys.date == t][0]
E = np.array([es(q) for q in phys.q])
El = np.array([es(q) if q is not None else np.nan for q in phys.q_lev])
print(f"9 Nov 2018: loss next week ${phys.loss[i] * MULT:,.0f}; PHYS-M2 ES ${E[i] * MULT:,.0f} (loss/ES {phys.loss[i] / E[i]:.2f}, "
      f"alarm {E[i] / np.median(E[:i]):.2f}); LEVEL ES ${El[i] * MULT:,.0f} (loss/ES {phys.loss[i] / El[i]:.2f}); "
      f"(S-a) alarm {phys.k[i] / np.median(phys.k[:i]):.2f}")
ratio = E / phys.k
print("PHYS-M2 ES/(S-a): median %.3f, 9 Nov 2018 %.3f (pct rank %.2f); corr(ES, S-a) = %.2f" %
      (np.median(ratio[:i]), ratio[i], (ratio[:i] < ratio[i]).mean(), np.corrcoef(E, phys.k)[0, 1]))
