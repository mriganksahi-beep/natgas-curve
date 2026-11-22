"""Bootstrap helpers (year / winter blocks) and HAC regression."""
import numpy as np
import pandas as pd

RNG_SEED = 7
NBOOT = 2000


def boot_mean(x, groups=None, nboot=NBOOT, seed=RNG_SEED):
    """Mean and 95% CI; resample whole groups (e.g. gas years) if given, else observations."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(x))
    groups = np.asarray(groups)
    ug = np.unique(groups)
    sums = np.array([x[groups == g].sum() for g in ug])
    cnts = np.array([(groups == g).sum() for g in ug])
    idx = rng.integers(0, len(ug), size=(nboot, len(ug)))
    bm = sums[idx].sum(1) / cnts[idx].sum(1)
    m = x.mean()
    se = bm.std(ddof=1)
    return m, np.percentile(bm, 2.5), np.percentile(bm, 97.5), (m / se if se > 0 else np.nan)


def cell_table(P, value="drift", by=("pair", "L"), groups="gy"):
    rows = []
    for key, g in P.groupby(list(by)):
        if len(g) < 3:
            continue
        m, lo, hi, t = boot_mean(g[value].values, g[groups].values)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), mean=m, lo=lo, hi=hi, t=t,
                         n=len(g), frac_pos=(g[value] > 0).mean(), median=g[value].median()))
    return pd.DataFrame(rows)


def ols_cluster(y, X, groups, nboot=NBOOT, seed=RNG_SEED):
    """OLS coefficients with a cluster (group-block) bootstrap 95% CI. X includes no constant."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    groups = np.asarray(groups)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        s = rng.choice(ug, len(ug))
        ii = np.concatenate([pos[g] for g in s])
        try:
            bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    return beta, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), beta / bs.std(0, ddof=1)
