"""Evaluation of the spread->storage map candidates (agent storage_map). Cached per spec.

python -m src.tool.storage_map_eval --spec hetero_exp     # LOWO + walk-forward folds, ~2-3 min per spec
python -m src.tool.storage_map_eval --baselines           # climatology + inverse regressions (seconds)
python -m src.tool.storage_map_eval --summary             # tables -> data/processed/research_tool_map/*.csv

Walk-forward: winter Y predicted by a map fitted on winters < Y (needs >= 10 winters, so Y >= 2004 for the
1994-start specs). LOWO (leave-one-winter-out) is descriptive only.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from src.tool import _map_data as MD, storage_map as SM

OUT = MD.CACHE
XG = SM.X_GRID
DX = XG[1] - XG[0]
TH = (-200, -400, -600)
B_EVAL = 30
WF_MIN = 10


def score_grid(p, x):
    """CRPS (Bcf), log score (per Bcf), PIT, P(x<x0) from a density on the x grid."""
    cdf = np.cumsum(p)
    pit = float(np.interp(x, XG + DX / 2, cdf))
    crps = float(np.sum((cdf - (XG + DX / 2 >= x)) ** 2) * DX)
    dens = float(np.interp(x, XG, p)) / DX
    med = float(np.interp(0.5, cdf, XG))
    sd = float(np.sqrt(max(p @ XG ** 2 - (p @ XG) ** 2, 0)))
    r = {"pit": pit, "crps": crps, "logs": np.log(max(dens, 1e-12)), "med": med, "sd": sd}
    for x0 in TH:
        r[f"P{-x0}"] = float(cdf[np.searchsorted(XG, x0) - 1])
    return r


def fold_rows(spec, B=B_EVAL):
    form, noise, yv, xv, y0 = SM.SPECS[spec]
    t = MD.winter_table()
    t = t[t.index >= y0]
    rows = []
    folds = [("lowo", Y) for Y in t.index] + [("wf", Y) for Y in t.index if (t.index < Y).sum() >= WF_MIN]
    for mode, Y in folds:
        tr = t[t.index != Y] if mode == "lowo" else t[t.index < Y]
        m = SM.fit_map(int(tr.index.max()), spec=spec, B=B, seed=Y, table=tr)
        if xv == "xpct":
            m["scale"] = t.at[Y, "avg5_m"] / 100.0            # avg5 of the held-out winter, known before it
        s_obs, x_obs = t.at[Y, yv], t.at[Y, "x"]
        p = SM.posterior(m, [s_obs])[0]
        r = {"spec": spec, "mode": mode, "Y": Y, "n": len(tr), "x": x_obs, "S": t.at[Y, "S"], **score_grid(p, x_obs)}
        if form == "inverse":
            rows.append(r)
            continue
        # forward direction: predictive of S given realised x (mixture over draws)
        f, s = SM.forward(m, [x_obs], draws=True)
        f, s = f[:, 0], s[:, 0]
        r["S_pred_med"] = float(np.median(f))
        r["S_pit"] = float(np.mean(ndtr((s_obs - f) / s)))
        r["S_logs"] = float(np.log(np.mean(norm.pdf(s_obs, f, s))))
        # v5-style deterministic inverse of the point curve (NaN below the asymptote)
        th = m["theta"]
        if form == "exp":
            z = (s_obs - th[0]) / np.exp(th[1])
            r["x_det"] = float(-np.exp(th[2]) * np.log(z) * m["scale"]) if z > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _gauss_rows(name, X, y, t, folds):
    rows = []
    for mode, Y in folds:
        tr = t.index != Y if mode == "lowo" else t.index < Y
        Xa = np.column_stack([np.ones(len(t)), X])
        beta, *_ = np.linalg.lstsq(Xa[tr], y[tr], rcond=None)
        res = y[tr] - Xa[tr] @ beta
        n, k = tr.sum(), Xa.shape[1]
        s2 = res @ res / (n - k)
        xo = Xa[t.index == Y][0]
        v = s2 * (1 + xo @ np.linalg.pinv(Xa[tr].T @ Xa[tr]) @ xo)
        mu, sd = xo @ beta, np.sqrt(v)
        p = norm.pdf(XG, mu, sd) * DX
        p /= p.sum()
        rows.append({"spec": name, "mode": mode, "Y": Y, "n": int(n), "x": y[t.index == Y][0],
                     "S": t.at[Y, "S"], **score_grid(p, y[t.index == Y][0])})
    return rows


def baselines():
    t = MD.winter_table()
    y = t.x.values
    folds = [("lowo", Y) for Y in t.index] + [("wf", Y) for Y in t.index if (t.index < Y).sum() >= WF_MIN]
    a = lambda s: np.arcsinh(s / 0.05)
    rows = []
    rows += _gauss_rows("climatology", np.zeros((len(t), 0)), y, t, folds)
    rows += _gauss_rows("inv_asinhS", a(t.S.values)[:, None], y, t, folds)
    rows += _gauss_rows("inv_asinhS_lnH", np.column_stack([a(t.S.values), np.log(t.H.values)]), y, t, folds)
    rows += _gauss_rows("inv_asinhS_post2010", np.column_stack([a(t.S.values), (t.index >= 2010).astype(float)]),
                        y, t, folds)
    # ceilings: curve information at expiry that the S_T RND does not carry
    rows += _gauss_rows("ceil_asinhJK", a(t.JK.values)[:, None], y, t, folds)
    rows += _gauss_rows("ceil_asinhS_asinhJK", np.column_stack([a(t.S.values), a(t.JK.values)]), y, t, folds)
    rows += _gauss_rows("ceil_JK_linear", t.JK.values[:, None], y, t, folds)
    return pd.DataFrame(rows)


def summarise():
    fs = sorted(OUT.glob("folds_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs])
    out = []
    for (spec, mode), g in d.groupby(["spec", "mode"]):
        for win, gg in (("all", g), ("2010+", g[g.Y >= 2010]), ("<2010", g[g.Y < 2010])):
            if gg.empty:
                continue
            r = {"spec": spec, "mode": mode, "win": win, "n": len(gg), "crps": gg.crps.mean(), "logs": gg.logs.mean(),
                 "mae_med": (gg.med - gg.x).abs().mean(), "sd": gg.sd.mean(),
                 "cov50": ((gg.pit > .25) & (gg.pit < .75)).mean(), "cov80": ((gg.pit > .1) & (gg.pit < .9)).mean(),
                 "cov90": ((gg.pit > .05) & (gg.pit < .95)).mean()}
            for x0 in TH:
                r[f"brier{-x0}"] = ((gg[f"P{-x0}"] - (gg.x < x0)) ** 2).mean()
            if "S_logs" in gg and gg.S_logs.notna().any():
                r["S_logs"] = gg.S_logs.clip(lower=-15).mean()       # density floor e^-15 (2003 held out)
                r["S_logs_med"] = gg.S_logs.median()
                r["S_cov80"] = ((gg.S_pit > .1) & (gg.S_pit < .9)).mean()
                r["S_mae"] = (gg.S_pred_med - gg.S).abs().mean()
            out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "eval_summary.csv", index=False)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.spec:
        fold_rows(a.spec).to_csv(OUT / f"folds_{a.spec}.csv", index=False)
    if a.baselines:
        baselines().to_csv(OUT / "folds_baselines.csv", index=False)
    if a.summary:
        pd.set_option("display.width", 250)
        print(summarise().round(3).to_string())
