"""Block-bootstrap confidence intervals for Model 1's holdout advantage.

For each holdout week we compute the loss differential
    d_t = |miss of benchmark| - |miss of model|      (Bcf; positive = model closer)
The model's average advantage is mean(d_t). Misses come in runs, so weeks are not
independent; we resample blocks of consecutive weeks (moving block bootstrap) to keep
that dependence, and report percentile intervals. A HAC (Newey-West) t-test on d_t is
a cross-check (this is the Diebold-Mariano test).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def block_bootstrap(x, stat, block=8, n_boot=10_000, seed=0):
    x = np.asarray(x)
    n, rng = len(x), np.random.default_rng(seed)
    starts_max = n - block + 1
    k = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, k)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        out[b] = stat(x[idx])
    return out


def newey_west_t(d, lags=8):
    d = np.asarray(d) - np.mean(d)
    n = len(d)
    var = d @ d / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        var += 2 * w * (d[L:] @ d[:-L]) / n
    return np.sqrt(var / n)  # standard error of the mean, robust to autocorrelation


def main(tag="v2_own"):
    p = pd.read_csv(ROOT / f"data/processed/model1_predictions_{tag}.csv", index_col=0, parse_dates=True)
    h = p[p.index >= "2022-01-01"].dropna()
    ridge = [c for c in h if c.startswith("Ridge")][0]
    gbm = [c for c in h if c.startswith("Gradient")][0]
    a = {c: (h["actual"] - h[c]).abs().values for c in ["5-year average", ridge, gbm]}

    # autocorrelation of the model's misses: why blocks are needed
    e = (h["actual"] - h[ridge])
    print(f"Holdout weeks: {len(h)}. Lag-1 autocorrelation of ridge misses: {e.autocorr(1):.2f}, "
          f"lag-4: {e.autocorr(4):.2f}")

    d5 = a["5-year average"] - a[ridge]
    print(f"Lag-1 autocorrelation of the advantage d_t (ridge vs 5-yr): {pd.Series(d5).autocorr(1):.2f}")
    rows = []
    comps = [("Ridge vs 5-year average", a["5-year average"], a[ridge]),
             ("Gradient boosting vs 5-year average", a["5-year average"], a[gbm]),
             ("Ridge vs gradient boosting", a[gbm], a[ridge])]
    for name, bench, model in comps:
        d = bench - model
        nw_t = d.mean() / newey_west_t(d)
        for block in (1, 4, 8, 13, 26):
            bs = block_bootstrap(d, np.mean, block=block)
            # ratio of MAEs, resampling the same weeks for both (paired)
            pair = np.column_stack([bench, model])
            ratio = block_bootstrap(np.arange(len(d)), lambda i: pair[i, 1].mean() / pair[i, 0].mean(), block=block)
            rows.append({"comparison": name, "block (weeks)": block,
                         "advantage (Bcf)": d.mean(),
                         "95% CI low": np.percentile(bs, 2.5), "95% CI high": np.percentile(bs, 97.5),
                         "share of resamples <= 0 (%)": 100 * (bs <= 0).mean(),
                         "MAE ratio": model.mean() / bench.mean(),
                         "ratio CI low": np.percentile(ratio, 2.5), "ratio CI high": np.percentile(ratio, 97.5),
                         "Newey-West t (8 lags)": nw_t})
    t = pd.DataFrame(rows).round(3)
    print(t.to_string(index=False))
    t.to_csv(ROOT / f"reports/model1_bootstrap_{tag}.csv", index=False)
    return t


if __name__ == "__main__":
    main()
