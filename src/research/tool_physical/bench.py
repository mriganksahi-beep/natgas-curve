"""Walk-forward benchmark of physical as-of distributions of end-March storage deviation x.

Development only: every as-of date <= 2021-12-31; storage, weather, forecasts read through
src.tool.physical, which filters everything to <= the as-of date. Realised outcomes x(Y) are read
with as_of = 2021-12-31, so end-March 2022 is never read (2021-22 rows are unscored).

As-of grid: for each EIA storage week F, d = F + 7 (the Friday after the Thursday release), with
d in Nov 1 .. Feb 28. Models that run from the storage week (P-lite, Model 2, AN_F) use origin F.

    python -m src.research.tool_physical.bench grid            # as-of grid
    python -m src.research.tool_physical.bench run <model>     # one model -> cache (each < 10 min)
    python -m src.research.tool_physical.bench score           # scores, tables
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.tool import physical as P

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_tool_physical"
CACHE.mkdir(parents=True, exist_ok=True)
GRIDQ = (np.arange(1000) + 0.5) / 1000          # stored quantile grid (1000 equal-weight points)
X0 = (-200, -400, -600)

# analog-model variants (choose on winters 2012..2016, check 2017..2021)
AN = {
    "an_w_l3": dict(prior="last3", balance="winter"),
    "an_w_tr": dict(prior="trend", balance="winter"),
    "an_f_l3": dict(prior="last3", balance="fall"),
    "an_f_tr": dict(prior="trend", balance="fall"),
    "an_w_l3_ht": dict(prior="last3", balance="winter", hdd_trend=True),
    "an_f_l3_ht": dict(prior="last3", balance="fall", hdd_trend=True),
    # added after diagnosing (on 2012-16 winters) that the weekly HDD slope overstates season-level draws
    "an_f_tr_mu": dict(prior="trend", balance="fall", mundlak=True),
    "an_f_l3_mu": dict(prior="last3", balance="fall", mundlak=True),
}


def grid():
    p = CACHE / "grid.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["F", "asof"])
    s = P.storage_asof(CUT)
    rows = []
    for F in s.index:
        d = F + pd.Timedelta(days=7)
        if d > CUT or d.month not in (11, 12, 1, 2):
            continue
        Y = P.winter_of(d)
        if Y < 2012:
            continue
        rows.append({"F": F, "asof": d, "Y": Y, "month": d.month,
                     "weeks_to_go": (P.target_week(Y) - F).days // 7,
                     "baseline": P.baseline(Y, d), "x_real": P.x_realised(Y, CUT), "bench5": P.bench5(d)})
    g = pd.DataFrame(rows)
    assert g["asof"].max() <= CUT and g[g.Y == 2022].x_real.isna().all()
    g.to_csv(p, index=False)
    return g


def _q(x):
    return np.quantile(np.asarray(x, float), GRIDQ)


def run(model):
    g = grid()
    out, t0 = {}, time.time()
    if model in AN or model == "an_F":
        opts = AN.get(model, {})
        for r in g.itertuples():
            if model == "an_F":        # ablation: storage-week information only (no forecast, no extra days)
                base = AN[open(CACHE / "chosen.txt").read().strip()]
                d = P.physical_dist(r.asof - pd.Timedelta(days=1), use_forecast=False,
                                    weather_asof=r.F + pd.Timedelta(days=1), **base)
            else:
                d = P.physical_dist(r.asof, **opts)
            assert abs(d["baseline"] - r.baseline) < 1e-6 and d["storage_week"] == r.F
            out[r.asof] = _q(d["samples"])
    elif model == "clim":
        for r in g.itertuples():
            out[r.asof] = _q(P.physical_dist(r.asof, model="climatology")["samples"])
    elif model == "plite":
        import src.research.v5_implied_storage.physical as VP
        # P-lite's _season_end maps an October origin to the PAST March; use the coming March
        # (v5 only ran it from November origins). No other change to P-lite.
        VP._season_end = lambda F: P.target_week(P.winter_of(F))
        pl = VP.PLite()
        for r in g.itertuples():
            out[r.asof] = _q(pl.simulate(r.F)["levels"] - r.baseline)
    elif model == "m2":
        m = pd.read_csv(ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv", parse_dates=["origin"])
        m = m[m.origin <= CUT].set_index("origin")
        pc = [f"p{i}" for i in range(1, 100)]
        for r in g.itertuples():
            if r.F in m.index:
                lv = np.sort(m.loc[r.F, pc].values.astype(float))
                # 99 percentiles -> 1000-point grid by linear interpolation (tails held at p1/p99)
                out[r.asof] = np.interp(GRIDQ, np.arange(1, 100) / 100, lv) - r.baseline
    else:
        raise ValueError(model)
    pd.to_pickle(out, CACHE / f"q_{model}.pkl")
    print(model, len(out), f"{time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- scoring
def crps(q, y):
    """CRPS of an equal-weight sample q (sorted) at outcome y."""
    q = np.sort(q)
    n = len(q)
    e1 = np.abs(q - y).mean()
    e2 = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * q) / n ** 2
    return e1 - 0.5 * e2


def load_all(models):
    g = grid().set_index("asof")
    rows = []
    for m in models:
        p = CACHE / f"q_{m}.pkl"
        if not p.exists():
            continue
        Q = pd.read_pickle(p)
        for d, q in Q.items():
            r = g.loc[d]
            o = {"asof": d, "model": m, "Y": r.Y, "month": r.month, "weeks_to_go": r.weeks_to_go,
                 "x_real": r.x_real, "med": np.median(q), "p10": np.quantile(q, .1), "p90": np.quantile(q, .9),
                 "p25": np.quantile(q, .25), "p75": np.quantile(q, .75), "p05": np.quantile(q, .05),
                 "p95": np.quantile(q, .95)}
            for x0 in X0:
                o[f"P{-x0}"] = float((q < x0).mean())
            if np.isfinite(r.x_real):
                y = r.x_real
                o["crps"] = crps(q, y)
                o["pit"] = float((q < y).mean())
                o["cov50"] = float(o["p25"] <= y <= o["p75"])
                o["cov80"] = float(o["p10"] <= y <= o["p90"])
                o["cov90"] = float(o["p05"] <= y <= o["p95"])
                for x0 in X0:
                    o[f"B{-x0}"] = (o[f"P{-x0}"] - float(y < x0)) ** 2
            rows.append(o)
    return pd.DataFrame(rows)


def centred(models):
    """Recentred versions (median moved to the 5-year path, v5 convention)."""
    g = grid().set_index("asof")
    for m in models:
        p = CACHE / f"q_{m}.pkl"
        if p.exists():
            Q = pd.read_pickle(p)
            pd.to_pickle({d: q - np.median(q) + (g.loc[d, "bench5"] - g.loc[d, "baseline"]) for d, q in Q.items()},
                         CACHE / f"q_{m}_rc.pkl")


def block_ci(df, col, ref=None, B=4000, seed=0):
    """Winter-weighted mean of col (each winter's mean counts once) and winter-block bootstrap CI.
    With ref: mean of (model - ref) differences on common dates."""
    d = df.groupby("Y")[col].mean()
    rng = np.random.default_rng(seed)
    w = d.index.values
    bs = [d.loc[rng.choice(w, len(w))].mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "grid":
        print(grid().groupby("Y").agg(n=("asof", "size"), x=("x_real", "first"), base=("baseline", "first")))
    elif cmd == "run":
        for m in sys.argv[2:]:
            run(m)
