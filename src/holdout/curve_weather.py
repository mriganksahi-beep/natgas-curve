"""Frozen holdout test V1 (curve_weather): weather skill of the curve over persistence.

Spec (reports/agents/curve_weather.md, section 6; frozen before any 2022+ data was seen):
  sample : Nov-Feb Fridays with F >= START (2022-01-07 for the holdout), panel built by
           src/research/curve_weather/data.py with only the end date changed
  coefs  : data/processed/research_curve_weather/frozen_holdout_coefs.json (not refitted)
           front+pers: w = c + b1*r_front + b2*a0 ; pers: w = c + b*a0
  target : wx2 = a1 + a2 (HDD anomaly vs prior-10-year same-week mean)
  stat   : dSS = SS(front+pers) - SS(pers), SS = 1 - sum(y-p)^2 / sum(y^2);
           one-sided Diebold-Mariano on d = (y-p_fp)^2 - (y-p_pers)^2 (HAC, 2 lags; H1: mean d < 0)
  verdict: PASS if dSS > 0 and p < 0.05; INCONCLUSIVE if dSS > 0 and p >= 0.05; FAIL if dSS <= 0.

Usage:
  python -m src.holdout.curve_weather --end 2021-12-31 --start 2010-01-01   # validation reproduction
  python -m src.holdout.curve_weather --end 2026-11-17 --start 2022-01-07   # the holdout (run once)

The panel builder is reused unchanged: its module-level CUTOFF and CACHE are re-pointed at run
time, so the development cache (panel.csv) is never overwritten.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.research.curve_weather import data

ROOT = Path(__file__).resolve().parents[2]
COEFS = ROOT / "data/processed/research_curve_weather/frozen_holdout_coefs.json"


def ss(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


def predict(df, spec):
    c = spec["coef"]
    return c["const"] + sum(c[x] * df[x] for x in spec["features"])


def dm(y, p1, p2, lags=2):
    d = (y - p1) ** 2 - (y - p2) ** 2
    m = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(m.tvalues[0])
    from scipy.stats import norm
    return float(m.params[0]), t, float(norm.cdf(t))  # one-sided: H1 mean d < 0


def winter_boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = {k: v for k, v in d.groupby("winter")}
    keys = list(g)
    out = []
    for _ in range(n):
        s = pd.concat([g[k] for k in rng.choice(keys, len(keys))])
        out.append(ss(s.wx2, s.p_fp) - ss(s.wx2, s.p_pers))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def run(end, start):
    end, start = pd.Timestamp(end), pd.Timestamp(start)
    data.CUTOFF = end
    data.CACHE = ROOT / f"data/processed/research_curve_weather/holdout_end_{end.date()}"
    data.CACHE.mkdir(parents=True, exist_ok=True)
    df = data.build(force=True)
    assert df.d0.max() <= end
    coefs = json.load(open(COEFS))
    d = df[df.F >= start].dropna(subset=["r_front", "a0", "wx2"]).copy()
    d["p_fp"] = predict(d, coefs["front+pers"])
    d["p_pers"] = predict(d, coefs["pers"])
    d["p_front"] = predict(d, coefs["front"])
    y = d.wx2
    res = {"end": str(end.date()), "start": str(start.date()), "n": int(len(d)),
           "winters": sorted(d.winter.unique().tolist()),
           "first_F": str(d.F.min().date()), "last_F": str(d.F.max().date()),
           "SS_front+pers": float(ss(y, d.p_fp)), "SS_pers": float(ss(y, d.p_pers)),
           "SS_front": float(ss(y, d.p_front))}
    res["dSS"] = res["SS_front+pers"] - res["SS_pers"]
    res["dm_mean_d"], res["dm_t"], res["dm_p_one_sided"] = dm(y, d.p_fp, d.p_pers)
    res["dSS_winter_boot_ci"] = winter_boot(d)
    for k, p in (("front+pers", d.p_fp), ("pers", d.p_pers), ("front", d.p_front)):
        nz = (y != 0) & (p != 0)
        res[f"corr_{k}"] = float(np.corrcoef(y, p)[0, 1])
        res[f"hit_{k}"] = float((np.sign(y[nz]) == np.sign(p[nz])).mean())
    if res["dSS"] <= 0:
        res["verdict"] = "FAIL"
    elif res["dm_p_one_sided"] < 0.05:
        res["verdict"] = "PASS"
    else:
        res["verdict"] = "INCONCLUSIVE"
    bw = []
    for w, g in d.groupby("winter"):
        bw.append({"winter": w, "n": len(g), "sd_wx2": float(g.wx2.std()),
                   "SS_front+pers": float(ss(g.wx2, g.p_fp)), "SS_pers": float(ss(g.wx2, g.p_pers)),
                   "SS_front": float(ss(g.wx2, g.p_front)),
                   "dSS": float(ss(g.wx2, g.p_fp) - ss(g.wx2, g.p_pers)),
                   "corr_front": float(np.corrcoef(g.wx2, g.r_front)[0, 1])})
    res["by_winter"] = bw
    out = d[["F", "winter", "r_front", "a0", "wx2", "p_fp", "p_pers", "p_front"]]
    out.to_csv(data.CACHE / "v1_weekly.csv", index=False)
    json.dump(res, open(data.CACHE / "v1_results.json", "w"), indent=1)
    return res, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    ap.add_argument("--start", required=True)
    a = ap.parse_args()
    res, _ = run(a.end, a.start)
    print(json.dumps({k: v for k, v in res.items() if k != "by_winter"}, indent=1))
    print(pd.DataFrame(res["by_winter"]).round(3).to_string())
