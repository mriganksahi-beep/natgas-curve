"""Evaluation harness for src/tool/rnd_direct.py (investigation rnd_engine; report reports/tool/rnd_engine.md).

All data are read through rnd_direct (options/futures filtered to date <= 2021-12-31 on load).
Realised S_T (H-J settle on the option-expiry day) is used only for winters Y <= 2021 (expiry <= Feb 2021);
the 2021-22 winter (Nov-Dec 2021 rows) is scored for coverage/stability/LOSO only.

Steps (each < 10 min; cached per winter in data/processed/research_tool_rnd/):
    python -m src.tool.rnd_engine_eval daily  [Y ...]   # all methods, every winter day -> daily_{Y}.pkl
    python -m src.tool.rnd_engine_eval loso   [Y ...]   # leave-one-strike-out (every 3rd day)   -> loso_{Y}.pkl
    python -m src.tool.rnd_engine_eval thin   [Y ...]   # dense days refit on the thin strike lattice -> thin_{Y}.pkl
    python -m src.tool.rnd_engine_eval oiw    [Y ...]   # zero-OI weight variants (LOSO on OI>0 strikes) -> oiw_{Y}.pkl
    python -m src.tool.rnd_engine_eval final  [Y ...]   # shipped config (drop zero-OI marks) -> final_{Y}.pkl
    python -m src.tool.rnd_engine_eval losofinal [Y ...] # LOSO under the shipped quote rule -> losofinal_{Y}.pkl
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from src.tool import rnd_direct as R

warnings.filterwarnings("ignore")
C = R.CACHE
C.mkdir(parents=True, exist_ok=True)
METHODS = ["mixture", "quad", "svi", "convex", "sln"]
DELTAS = [-0.10, 0.0, 0.10, 0.25, 0.50, 1.00]
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
AGRID = np.round(np.arange(-0.5, 3.0001, 0.01), 2)
WINTERS = list(range(2013, 2023))
THIN_LATTICE = np.array([-0.10, -0.05, 0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00])


def winter_days(Y):
    """Option quote days and the futures-trading-day denominator for winter Y (Nov 1 .. day before expiry)."""
    o = R.load_options()
    fut = R.load_futures()
    oe = R.option_expiry(Y)
    lo = pd.Timestamp(Y - 1, 11, 1)
    hi = min(oe - pd.Timedelta(days=1), R.DATA_CUTOFF)
    fd = fut.loc[Y].index if Y in fut.index.get_level_values(0) else pd.DatetimeIndex([])
    fd = fd[(fd >= lo) & (fd <= hi)]
    oy = o[(o.Y == Y) & (o.date <= R.DATA_CUTOFF)]
    return oy, fd


def realised(Y):
    fut = R.load_futures()
    oe = R.option_expiry(Y)
    if oe > R.DATA_CUTOFF:
        return np.nan
    try:
        return float(fut.loc[(Y, oe), "S"])
    except KeyError:
        return np.nan


def _prev_map(oy):
    ds = np.sort(oy.date.unique())
    g = dict(tuple(oy.groupby("date")))
    return ds, g


def crps(grid, cdf, y):
    ind = (grid >= y).astype(float)
    return float(np.sum((cdf - ind) ** 2) * (grid[1] - grid[0]))


# ------------------------------------------------------------------------------------------- daily
def run_daily(Y, force=False, methods=None, zw=1.0, tag="daily"):
    """Comparison runs use zero_oi_weight = 1 (all quotes) for every method; tag 'final' = shipped defaults."""
    methods = METHODS if methods is None else methods
    p = C / f"{tag}_{Y}.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    oy, fd = winter_days(Y)
    ds, g = _prev_map(oy)
    ST = realised(Y)
    rows = []
    t0 = time.time()
    for d in fd:
        if d not in g:
            for m in methods:
                rows.append({"date": d, "Y": Y, "method": m, "has_quotes": False, "usable": False})
            continue
        i = np.searchsorted(ds, np.datetime64(d))
        prev = g[pd.Timestamp(ds[i - 1])] if i > 0 else None
        for m in methods:
            try:
                r = R.spread_density(d, Y, quotes=g[d], prev_quotes=prev, method=m, zero_oi_weight=zw)
            except Exception as e:                       # noqa
                rows.append({"date": d, "Y": Y, "method": m, "has_quotes": True, "usable": False, "err": str(e)})
                continue
            F = r["F"]
            row = {"date": d, "Y": Y, "method": m, "has_quotes": True, "usable": r["usable"], "F": F,
                   "F_fut": r.get("F_futures", np.nan), "T": r["T"], "nq": r["n_quotes"], "fit_method": r["method"]}
            row.update({k: v for k, v in r["flags"].items() if np.isscalar(v)})
            if len(r["grid"]):
                for dl in DELTAS:
                    row[f"P{dl:+.2f}"] = float(r["prob_above"](F + dl))
                for qq, v in zip(QS, r["quantile"](QS)):
                    row[f"q{int(qq * 100):02d}"] = float(v)
                row["cdf_abs"] = r["cdf_f"](AGRID).astype(np.float32)
                if np.isfinite(ST):
                    row["ST"] = ST
                    row["pit"] = float(r["cdf_f"](ST))
                    row["logpdf"] = float(np.log(max(np.interp(ST, r["grid"], r["pdf"]), 1e-4)))
                    row["crps"] = crps(r["grid"], r["cdf"], ST)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    print(f"{tag} {Y}: {len(fd)} days, {time.time() - t0:.0f}s", flush=True)
    return df


# -------------------------------------------------------------------------------------------- LOSO
def _subset(q, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(q["K"]) else v) for k, v in q.items()}


def _loso_day(q, F, T, methods, cand_mask, max_k=15):
    out = []
    idx = np.where(cand_mask)[0]
    if len(idx) > max_k:
        idx = idx[np.argsort(np.abs(q["K"][idx] - F))[:max_k]]
    for i in idx:
        keep = np.ones(len(q["K"]), bool)
        keep[i] = False
        qi = _subset(q, keep)
        if len(qi["K"]) < 2:
            continue
        wing = q["K"][i] < qi["K"].min() or q["K"][i] > qi["K"].max()
        for m in methods:
            try:
                f = R.fit_quotes(qi, F, T, m)
                pred = float(f.otm(np.array([q["K"][i]]), np.array([q["isc"][i]]), F)[0])
            except Exception:
                pred = np.nan
            out.append({"method": m, "K": q["K"][i], "x": q["K"][i] - F, "px": q["px"][i], "pred": pred,
                        "wing": wing, "oi_ok": bool(q["oi_ok"][i]), "nq": len(q["K"])})
    return out


def run_loso(Y, force=False, step=3, zw=1.0, methods=None, tag="loso"):
    """Comparison: all quotes (zw = 1), all methods. tag 'losofinal': shipped quote rule (zw = None -> module
    default, marks dropped wing by wing), scored on the same kind of target: strikes with open interest."""
    methods = METHODS if methods is None else methods
    p = C / f"{tag}_{Y}.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    oy, fd = winter_days(Y)
    ds, g = _prev_map(oy)
    days = [d for d in fd if d in g][::step]
    rows, t0 = [], time.time()
    for d in days:
        i = np.searchsorted(ds, np.datetime64(d))
        prev = g[pd.Timestamp(ds[i - 1])] if i > 0 else None
        base = R.spread_density(d, Y, quotes=g[d], prev_quotes=prev, method="convex", zero_oi_weight=zw)
        F, T, q = base["F"], base["T"], base.get("quotes")
        if q is None or len(q["K"]) < 3:
            continue
        cand = q["oi_ok"] & (np.abs(q["K"] - F) <= 1.5) & (q["px"] >= 0.002)
        for r in _loso_day(q, F, T, methods, cand):
            r.update({"date": d, "Y": Y})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    print(f"{tag} {Y}: {len(days)} days, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    return df


# ------------------------------------------------------------------------------- thin-ification test
def run_thin(Y, force=False, step=2):
    """Dense days (>= 30 quotes): refit using only strikes on the thin-year lattice; score held-out strikes
    (|K-F| <= 1) and the change in P(S_T > F + d) against the same method's dense fit and the dense convex fit."""
    p = C / f"thin_{Y}.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    oy, fd = winter_days(Y)
    ds, g = _prev_map(oy)
    rows, t0 = [], time.time()
    for d in [d for d in fd if d in g][::step]:
        i = np.searchsorted(ds, np.datetime64(d))
        prev = g[pd.Timestamp(ds[i - 1])] if i > 0 else None
        ref = R.spread_density(d, Y, quotes=g[d], prev_quotes=prev, method="convex", zero_oi_weight=1.0)
        q, F, T = ref.get("quotes"), ref["F"], ref["T"]
        if q is None or len(q["K"]) < 30:
            continue
        on = np.isin(np.round(q["K"], 3), THIN_LATTICE)
        if on.sum() < 4:
            continue
        qt = _subset(q, on)
        ho = (~on) & (np.abs(q["K"] - F) <= 1.0) & (q["px"] >= 0.002)
        ks = F + np.array([0.10, 0.25, 0.50])
        Pref = ref["prob_above"](ks)
        for m in METHODS:
            try:
                fd_ = R.fit_quotes(q, F, T, m)
                ft = R.fit_quotes(qt, F, T, m)
                _, _, cd, _, _ = R.density_from_fit(fd_, F)
                _, _, ct, _, _ = R.density_from_fit(ft, F)
                gr = F + np.arange(R.GRID_LO, R.GRID_HI + 1e-9, R.GRID_DK)
                Pd = 1 - np.interp(ks, gr, cd)
                Pt = 1 - np.interp(ks, gr, ct)
                err = ft.otm(q["K"][ho], q["isc"][ho], F) - q["px"][ho]
            except Exception:
                continue
            rows.append({"date": d, "Y": Y, "method": m, "nq_dense": len(q["K"]), "nq_thin": int(on.sum()),
                         "ho_rmse": float(np.sqrt(np.mean(err ** 2))), "ho_mae": float(np.mean(np.abs(err))),
                         "ho_n": int(ho.sum()),
                         **{f"dP_self{dl}": float(Pt[j] - Pd[j]) for j, dl in enumerate([10, 25, 50])},
                         **{f"dP_ref{dl}": float(Pt[j] - Pref[j]) for j, dl in enumerate([10, 25, 50])}})
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    print(f"thin {Y}: {df.date.nunique() if len(df) else 0} days, {time.time() - t0:.0f}s", flush=True)
    return df


# ---------------------------------------------------------------------------- zero-OI weight variants
def run_oiw(Y, force=False, step=3, methods=("svi", "sln", "convex"), weights=(1.0, 0.3, 0.0)):
    p = C / f"oiw_{Y}.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    oy, fd = winter_days(Y)
    ds, g = _prev_map(oy)
    rows, t0 = [], time.time()
    for d in [d for d in fd if d in g][::step]:
        i = np.searchsorted(ds, np.datetime64(d))
        prev = g[pd.Timestamp(ds[i - 1])] if i > 0 else None
        if prev is None:
            continue
        for zw in weights:
            base = R.spread_density(d, Y, quotes=g[d], prev_quotes=prev, method="convex", zero_oi_weight=zw)
            q, F, T = base.get("quotes"), base["F"], base["T"]
            if q is None or len(q["K"]) < 3 or q["oi_ok"].all():
                break                                   # only days with some zero-OI quotes are informative
            if zw == 0.0:                               # weight 0 == drop the marks
                q = _subset(q, q["oi_ok"])
                if len(q["K"]) < 3:
                    continue
            cand = q["oi_ok"] & (np.abs(q["K"] - F) <= 1.5) & (q["px"] >= 0.002)
            for r in _loso_day(q, F, T, list(methods), cand, max_k=10):
                r.update({"date": d, "Y": Y, "zw": zw})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    print(f"oiw {Y}: {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    step = sys.argv[1]
    ys = [int(a) for a in sys.argv[2:]] or WINTERS
    fn = {"daily": run_daily, "loso": run_loso, "thin": run_thin, "oiw": run_oiw,
          "final": lambda Y, force: run_daily(Y, force, methods=["sln", "svi", "mixture"], zw=None, tag="final"),
          "losofinal": lambda Y, force: run_loso(Y, force, zw=None, methods=["sln", "svi", "mixture"], tag="losofinal")}[step]
    for Y in ys:
        fn(Y, force=True)
