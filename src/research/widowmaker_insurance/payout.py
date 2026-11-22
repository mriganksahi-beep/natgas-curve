"""Expiry payout curve: expiry spread vs end-March surplus x (Bcf vs prior-years average).

Form (convex, monotone decreasing in x):   y = a + b * exp(-x / tau),  b >= 0, tau in [150, 3000] Bcf.
Fitted by profile least squares: for each tau on a log grid, OLS for (a, b) with b clipped at 0.
Uncertainty: residual bootstrap (residuals inflated by sqrt(n / (n - 3))), refitting each draw.
"""
import numpy as np
import pandas as pd

TAU_GRID = np.exp(np.linspace(np.log(150), np.log(3000), 80))


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for tau in TAU_GRID:
        z = np.exp(-x / tau)
        zc, yc = z - z.mean(), y - y.mean()
        b = max(0.0, (zc @ yc) / (zc @ zc)) if zc @ zc > 0 else 0.0
        a = y.mean() - b * z.mean()
        sse = ((y - a - b * z) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, a, b, tau)
    return np.array(best[1:])  # a, b, tau


def predict(p, x):
    p = np.atleast_2d(p)
    x = np.asarray(x, float)
    return p[:, [0]] + p[:, [1]] * np.exp(-x[None, :] / p[:, [2]])


def fit_linear(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def bootstrap(x, y, B=500, seed=0):
    """Returns (params[B,3], residual pool) for the predictive distribution."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0 = fit(x, y)
    yhat = predict(p0, x)[0]
    n = len(x)
    e = (y - yhat) * np.sqrt(n / max(n - 3, 1))
    ps = np.empty((B, 3))
    for i in range(B):
        ps[i] = fit(x, yhat + rng.choice(e, n, replace=True))
    return p0, ps, e


def curves(tab, target="S_exp", scaled=False, min_prior=1, B=500):
    """Walk-forward (earlier winters only) and leave-one-winter-out fits for each Y in 2012..2022.
    tab: index Y with columns x, n_prior, S_exp, S_exp5, J_exp."""
    t = tab[(tab.n_prior >= min_prior) & tab[target].notna()].copy()
    t["y"] = t[target] / t.J_exp if scaled else t[target]
    out = {}
    for Y in range(2011, 2023):
        for mode in ("wf", "lowo"):
            tr = t[t.index < Y] if mode == "wf" else t[t.index != Y]
            if len(tr) < 5:
                continue
            p0, ps, e = bootstrap(tr.x.values, tr.y.values, B=B, seed=Y)
            out[(Y, mode)] = {"p0": p0, "ps": ps, "e": e, "n": len(tr), "years": list(tr.index)}
    return out, t


def oos_errors(t, curve_dict):
    """Out-of-sample error of the point curve at the realised x, for each Y with an outcome."""
    rows = []
    for (Y, mode), c in curve_dict.items():
        if Y not in t.index:
            continue
        pred = predict(c["p0"], [t.x[Y]])[0, 0]
        tr = t.loc[c["years"]]
        a, b = fit_linear(tr.x.values, tr.y.values)
        rows.append({"Y": Y, "mode": mode, "n_fit": c["n"], "x": t.x[Y], "y": t.y[Y], "pred_convex": pred,
                     "pred_linear": a + b * t.x[Y], "pred_mean": tr.y.mean(), "tau": c["p0"][2]})
    return pd.DataFrame(rows)
