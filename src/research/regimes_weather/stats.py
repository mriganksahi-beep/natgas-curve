"""Small statistics helpers: OLS with HAC errors and a bootstrap that resamples whole winters."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def ols(df, y, xs, hac_lags=1):
    d = df.dropna(subset=[y] + xs)
    X = sm.add_constant(d[xs].astype(float))
    m = sm.OLS(d[y].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return m, d


def winter_boot(df, fn, n_boot=2000, seed=0, group="winter"):
    """fn(sub_df) -> scalar or 1-D array. Resample winters with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(group)}
    keys = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        sub = pd.concat([groups[k] for k in pick], ignore_index=True)
        try:
            out.append(fn(sub))
        except Exception:
            continue
    return np.array(out)


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return np.percentile(a, lo), np.percentile(a, hi)


def r2(df, y, xs):
    d = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(d))] + [d[x].values for x in xs])
    b, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    e = d[y].values - X @ b
    return 1 - e.var() / d[y].values.var()


def slope(df, y, x, controls=()):
    xs = [x] + list(controls)
    d = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in xs])
    b, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    return b[1]
