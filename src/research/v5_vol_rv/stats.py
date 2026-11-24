"""Winter (gas-year Apr..Mar) block bootstrap helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261124)
B = 2000


def block_mean_ci(x, blocks, B=B, rng=RNG):
    """Pooled mean of x and 95% CI from resampling whole blocks (gas years)."""
    df = pd.DataFrame({"x": np.asarray(x, float), "g": np.asarray(blocks)}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan, 0
    agg = df.groupby("g").x.agg(["sum", "count"])
    s, c = agg["sum"].values, agg["count"].values
    k = len(s)
    idx = rng.integers(0, k, size=(B, k))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), k


def table(df, col, by, blocks="gy", scale=1.0):
    rows = []
    for key, g in df.groupby(by, observed=True):
        m, lo, hi, k = block_mean_ci(g[col] * scale, g[blocks])
        rows.append({**(dict(zip(by, key)) if isinstance(by, list) else {by: key}),
                     "mean": m, "lo": lo, "hi": hi, "n": len(g.dropna(subset=[col])), "blocks": k})
    return pd.DataFrame(rows)


def sharpe_ci(pnl, blocks, periods_per_year, B=B, rng=RNG):
    """Annualised Sharpe of a periodic P&L series, CI by gas-year block bootstrap."""
    df = pd.DataFrame({"x": np.asarray(pnl, float), "g": np.asarray(blocks)}).dropna()
    gs = [g.x.values for _, g in df.groupby("g")]
    k = len(gs)
    sh = lambda v: v.mean() / v.std(ddof=1) * np.sqrt(periods_per_year) if v.std() > 0 else np.nan
    est = sh(df.x.values)
    bs = []
    for _ in range(B):
        v = np.concatenate([gs[i] for i in rng.integers(0, k, k)])
        bs.append(sh(v))
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
