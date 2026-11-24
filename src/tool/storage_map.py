"""Spread -> storage map for the Winter Squeeze-Risk Monitor (link C of reports/tool/SPEC.md).

Agent `storage_map`, 24 Nov 2026. Report: reports/tool/storage_map.md.

What it does
------------
Maps the March-April futures spread at the March option expiry, S_T ($/MMBtu), to the end-March storage
deviation x (Bcf, end-March working gas minus the mean of the 5 prior end-March levels), WITH map
uncertainty, and turns a risk-neutral density of S_T into implied odds P(x < x0) and quantiles of x.

Default spec "inv_asinh" (chosen in the report on 1994-2021 walk-forward and leave-one-winter-out CRPS):
  x | S_T  ~  N( alpha + beta * asinh(S_T / 0.05),  sigma^2 * n/(n-2) )
  i.e. a DIRECT regression in the direction the tool uses. asinh is linear for |S| < ~5c and logarithmic
  above, so a 2003-type squeeze ($3 spread) cannot drag x to -1,600 Bcf. It is defined for every S_T
  (no asymptote problem). In the flat region it reverts toward climatology by construction.
  Map uncertainty: pairs (winter) bootstrap, B draws; p(x | S_T) is the average of the draws' normals.
Alternative specs kept for comparison and for the FORWARD direction S_T | x (e.g. to put physical
scenarios into spread space): "hetero_exp"
  S_T | x ~ N(f(x), s(x)^2), f = a + b exp(-x/tau), s^2 = s0^2 + (k b exp(-x/tau))^2,
  inverted by Bayes with a climatological N(mu, sd) prior on x;  "homo_exp" (k = 0, the v3/v5 curve),
  "hetero_hinge", "hetero_exp_rel" (y = ln H/J), "hetero_exp_pct" (x in % of 5-yr avg), start-year variants.
Tool:  P_Q(x < x0) = integral p(x | s) q(s) ds,  q = RND of S_T. The band reported is the 5-95% range of that
       probability across bootstrap draws (parameter uncertainty); the point value already integrates the
       map's residual noise and parameter uncertainty.

Data (src/tool/_map_data.py): winters Y = 1994..2021 (Y = year of the March contract). S_T from EIA
c1-c2 (Y <= 2010; pre-2010 data no longer locked, v4 spent it) and the CME curve file (Y >= 2011; the two
agree exactly on 2011-21). x from EIA monthly US working gas (NG.N5020US2.M), end of March. Every row is
dated <= 2021-12-31. fit_map(last_winter) uses only winters <= last_winter, so it is walk-forward by
construction: for winter Y call fit_map(Y - 1) (that winter's end-March storage is public in April Y-1).

Usage
-----
    from src.tool.storage_map import fit_map, implied_storage
    m = fit_map(2021)                                    # frozen map for 2022+ (dev data only)
    out = implied_storage(samples_of_S_T, m)             # or (grid, cdf) tuple, or a callable cdf
    out["P"][-400], out["P_band"][-400], out["q"][0.5]
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from src.tool import _map_data as MD

X_GRID = np.arange(-2000.0, 2000.0 + 1e-9, 10.0)          # Bcf
THRESHOLDS = (-200, -400, -600, -800)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
S0_MIN = 0.01                                              # settle rounding / quote noise floor ($)

SPECS = {
    # name: (form, noise, y-variable, x-variable, first winter used)
    "inv_asinh":      ("inverse", "homo", "S", "x", 1994),       # DEFAULT: x = al + be*asinh(S/c) + sig*eps
    "homo_exp":       ("exp", "homo", "S", "x", 1994),
    "hetero_exp":     ("exp", "hetero", "S", "x", 1994),
    "hetero_hinge":   ("hinge", "hetero", "S", "x", 1994),
    "hetero_exp_rel": ("exp", "hetero", "lnHJ", "x", 1994),
    "hetero_exp_pct": ("exp", "hetero", "S", "xpct", 1994),
    "hetero_exp_1997": ("exp", "hetero", "S", "x", 1997),
    "hetero_exp_2010": ("exp", "hetero", "S", "x", 2010),
}
DEFAULT_SPEC = "inv_asinh"
C_ASINH = 0.05                                             # $/MMBtu: linear below ~5c, log above


# --------------------------------------------------------------------------------------------- forward fit
def _f_sig(th, x, form, noise):
    """Mean and s.d. of S given x. th = (a, log b, log tau_or_knot-shift, log s0, log k)."""
    a, b, c, s0, k = th[0], np.exp(th[1]), th[2], np.exp(th[3]), (np.exp(th[4]) if noise == "hetero" else 0.0)
    if form == "exp":
        g = b * np.exp(-x / np.exp(c))
        f = a + g
    else:  # hinge: flat-ish slope above knot c, steeper below; monotone decreasing, convex
        g = b * np.maximum(0.0, c - x) / 100.0
        f = a + g
    return f, np.sqrt(s0 ** 2 + (k * g) ** 2)


def _nll(th, x, y, form, noise):
    f, s = _f_sig(th, x, form, noise)
    return float(np.sum(np.log(s) + 0.5 * ((y - f) / s) ** 2))


def fit_forward(x, y, form="exp", noise="hetero", xscale=450.0):
    """ML fit, profiled over tau (exp) or the knot (hinge) on a grid, L-BFGS for the rest."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if form == "exp":
        grid = np.log(xscale * np.exp(np.linspace(np.log(0.33), np.log(6.7), 16)))
    else:
        grid = np.quantile(x, [0.15, 0.25, 0.35, 0.45, 0.55]) if len(x) > 5 else [np.median(x)]
    best = None
    for c in grid:
        if form == "exp":
            z = np.exp(-x / np.exp(c))
        else:
            z = np.maximum(0.0, c - x) / 100.0
        zc = z - z.mean()
        b0 = max((zc @ (y - y.mean())) / max(zc @ zc, 1e-12), 1e-4)
        a0 = y.mean() - b0 * z.mean()
        r = y - a0 - b0 * z
        s0 = max(np.std(r), S0_MIN)
        bnd = [(None, None), (np.log(1e-6), np.log(1e3)), (c, c), (np.log(S0_MIN), np.log(5.0)),
               (np.log(1e-3), np.log(20.0))]
        for lk in ((np.log(0.3),) if noise == "hetero" else (np.log(1e-3),)):
            th0 = np.array([a0, np.log(b0), c, np.log(s0), lk])
            res = minimize(_nll, th0, args=(x, y, form, noise), method="L-BFGS-B", bounds=bnd)
            if best is None or res.fun < best.fun:
                best = res
    return best.x, best.fun


def fit_inverse(s, x, c=C_ASINH):
    """OLS of x on asinh(S/c). Returns (alpha, beta, log sigma, 0) with sigma the unbiased residual s.d."""
    z = np.arcsinh(np.asarray(s, float) / c)
    X = np.column_stack([np.ones_like(z), z])
    b, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)
    r = x - X @ b
    return np.array([b[0], b[1], np.log(np.sqrt(r @ r / max(len(z) - 2, 1))), 0.0])


# --------------------------------------------------------------------------------------------- map object
def _training(last_winter, spec):
    form, noise, yv, xv, y0 = SPECS[spec]
    t = MD.winter_table()
    t = t[(t.index >= y0) & (t.index <= last_winter)].dropna(subset=[yv, xv])
    return t


def fit_map(last_winter, spec=DEFAULT_SPEC, B=200, seed=0, exclude=None, table=None):
    """Fit the map on winters <= last_winter (optionally excluding some, for leave-one-out description).

    Returns a dict: spec, years, theta (point fit), draws (B bootstrap thetas), prior (mu, sd) in x-units,
    avg5_scale (Bcf per x-unit for 'xpct' specs: the latest training avg5 / 100; 1 otherwise), x_grid.
    """
    form, noise, yv, xv, _ = SPECS[spec]
    t = _training(last_winter, spec) if table is None else table
    if exclude is not None:
        t = t[~t.index.isin(np.atleast_1d(exclude))]
    x, y = t[xv].values.astype(float), t[yv].values.astype(float)
    xscale = float(np.std(x))
    fitter = (lambda xx, yy: fit_inverse(yy, xx)) if form == "inverse" else \
        (lambda xx, yy: fit_forward(xx, yy, form, noise, xscale)[0])
    th = fitter(x, y)
    rng = np.random.default_rng(seed)
    draws = []
    n = len(x)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.unique(i).size < 5:
            continue
        draws.append(fitter(x[i], y[i]))
    if xv == "xpct":
        # x in % of the 5-yr average; apply to the CURRENT avg5 level (the next winter's, known at fit time)
        ms = MD.monthly_storage()["wg"]
        Ynext = int(last_winter) + 1
        avg5_next = np.mean([ms.get((Ynext - k) * 100 + 3) for k in range(1, 6)])
        scale = avg5_next / 100.0
    else:
        scale = 1.0
    return {"spec": spec, "form": form, "noise": noise, "yvar": yv, "xvar": xv, "years": list(map(int, t.index)),
            "theta": th, "draws": np.array(draws) if draws else th[None, :],
            "prior": (float(np.mean(x)), float(np.std(x, ddof=1))), "scale": scale,
            "last_winter": int(last_winter)}


def forward(m, x_bcf, draws=False):
    """Mean and s.d. of S_T (in the spec's y-units) given x in Bcf. draws=True -> per bootstrap draw."""
    if m["form"] == "inverse":
        raise ValueError("inverse spec has no forward curve; fit a forward spec (e.g. 'hetero_exp') for S|x")
    xu = np.asarray(x_bcf, float) / m["scale"]
    ths = m["draws"] if draws else m["theta"][None, :]
    out = [_f_sig(th, xu, m["form"], m["noise"]) for th in ths]
    f = np.array([o[0] for o in out])
    s = np.array([o[1] for o in out])
    return (f, s) if draws else (f[0], s[0])


def posterior(m, s_vals, x_grid=X_GRID, per_draw=False):
    """p(x | S_T = s) on x_grid (Bcf), rows = s_vals. Averaged over bootstrap draws (or per draw)."""
    s_vals = np.atleast_1d(np.asarray(s_vals, float))
    if m["form"] == "inverse":
        mu, sd = _inv_moments(m, s_vals)                       # [draw, s]
        p = np.exp(-0.5 * ((x_grid[None, None, :] - mu[..., None]) / sd[..., None]) ** 2) / sd[..., None]
        posts = p / p.sum(axis=2, keepdims=True)
        return posts if per_draw else posts.mean(axis=0)
    mu, sd = m["prior"]
    xu = x_grid / m["scale"]
    lp_prior = -0.5 * ((xu - mu) / sd) ** 2
    posts = []
    for th in m["draws"]:
        f, s = _f_sig(th, xu, m["form"], m["noise"])
        ll = -np.log(s)[None, :] - 0.5 * ((s_vals[:, None] - f[None, :]) / s[None, :]) ** 2 + lp_prior[None, :]
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        posts.append(p / p.sum(axis=1, keepdims=True))
    posts = np.array(posts)                                    # [draw, s, x]
    return posts if per_draw else posts.mean(axis=0)


def _inv_moments(m, s_vals):
    """Per-draw predictive mean and s.d. (Bcf) of x given S for the inverse spec. The s.d. is inflated by
    sqrt(n/(n-2)) for the out-of-sample (leverage-free) predictive."""
    n = len(m["years"])
    d = m["draws"]
    z = np.arcsinh(np.asarray(s_vals, float) / C_ASINH)
    mu = (d[:, [0]] + d[:, [1]] * z[None, :]) * m["scale"]
    sd = np.exp(d[:, [2]]) * np.sqrt(n / max(n - 2, 1)) * m["scale"] * np.ones_like(mu)
    return mu, sd


# --------------------------------------------------------------------------------------------- the tool
def _spread_weights(spread, y_transform=None):
    """Turn samples / (grid, cdf) / callable cdf into (support points, weights) on a compact S grid."""
    edges = np.concatenate([[-np.inf], np.arange(-1.0, 1.0, 0.005), np.arange(1.0, 6.0, 0.05), [np.inf]])
    if callable(spread):
        cdf = np.array([spread(e) if np.isfinite(e) else (0.0 if e < 0 else 1.0) for e in edges])
        w = np.diff(cdf)
    elif isinstance(spread, tuple):
        g, c = map(np.asarray, spread)
        cdf = np.interp(edges, g, c, left=0.0, right=1.0)
        w = np.diff(cdf)
    else:
        s = np.asarray(spread, float)
        s = s[np.isfinite(s)]
        w = np.histogram(s, bins=edges)[0].astype(float)
    w = np.clip(w, 0, None)
    w /= w.sum()
    mid = np.concatenate([[-1.0025], (edges[1:-2] + edges[2:-1]) / 2, [6.025]])
    keep = w > 1e-7
    pts = mid[keep]
    if y_transform is not None:
        pts = y_transform(pts)
    return pts, w[keep]


def implied_storage(spread, m, thresholds=THRESHOLDS, quantiles=QUANTILES, J_level=None, band=(0.05, 0.95)):
    """Implied end-March storage distribution from a density of S_T, including map uncertainty.

    spread : 1-D array of S_T samples, or (grid, cdf) tuple, or callable cdf(s); $/MMBtu.
    m      : output of fit_map.
    J_level: April futures level (needed only for the relative spec 'hetero_exp_rel', y = ln(H/J) =
             ln(1 + S/J)); use today's April futures price.
    Returns dict: P {x0: P(x<x0)}, P_band {x0: (lo, hi)} across bootstrap draws, q {quantile: Bcf},
    density (x_grid, p), mean, post_sd_mean (q-weighted s.d. of x | S_T; ~345 Bcf for inv_asinh) and
    prior_sd (climatology, ~440): their ratio is the honest resolution of the reading.
    """
    tr = None
    if m["yvar"] == "lnHJ":
        if J_level is None:
            raise ValueError("relative spec needs J_level")
        tr = lambda s: np.log1p(np.maximum(s, -0.9 * J_level) / J_level)
    pts, w = _spread_weights(spread, tr)
    post_d = posterior(m, pts, per_draw=True)                 # [draw, s, x]
    mix_d = np.einsum("s,dsx->dx", w, post_d)                 # per-draw implied x density
    mix = mix_d.mean(axis=0)
    cdf_d = np.cumsum(mix_d, axis=1)
    cdf = np.cumsum(mix)
    xg = X_GRID
    P, Pb = {}, {}
    for x0 in thresholds:
        j = np.searchsorted(xg, x0)                           # mass strictly below x0 (cell centres < x0)
        pd_ = cdf_d[:, j - 1] if j > 0 else np.zeros(len(cdf_d))
        P[x0] = float(cdf[j - 1]) if j > 0 else 0.0
        Pb[x0] = (float(np.quantile(pd_, band[0])), float(np.quantile(pd_, band[1])))
    q = {qq: float(np.interp(qq, cdf, xg)) for qq in quantiles}
    pm = post_d.mean(axis=0)
    m1 = pm @ xg
    sd = np.sqrt(np.clip(pm @ xg ** 2 - m1 ** 2, 0, None))
    prior_sd = m["prior"][1] * m["scale"]
    return {"P": P, "P_band": Pb, "q": q, "density": (xg, mix), "mean": float(mix @ xg),
            "post_sd_mean": float(w @ sd), "prior_sd": float(prior_sd)}


def describe(m, s_vals=(-0.15, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.5, 3.0)):
    """Table of p(x | S_T = s): median, 10/90%, P(x<x0) - the honest resolution of the map."""
    pm = posterior(m, np.asarray(s_vals))
    cdf = np.cumsum(pm, axis=1)
    rows = []
    for i, s in enumerate(s_vals):
        r = {"S_T": s, "x_p10": np.interp(0.1, cdf[i], X_GRID), "x_p50": np.interp(0.5, cdf[i], X_GRID),
             "x_p90": np.interp(0.9, cdf[i], X_GRID)}
        for x0 in THRESHOLDS:
            r[f"P<{x0}"] = cdf[i, np.searchsorted(X_GRID, x0) - 1]
        rows.append(r)
    return pd.DataFrame(rows)
