"""Part 3: do kinks correct? Pooled regressions and a butterfly trade."""
import numpy as np
import pandas as pd

from .data import MULT, TICK

RNG = np.random.default_rng(20261121)
IS = ("2011-01-01", "2017-12-31")
OOS = ("2018-01-01", "2021-12-31")
TICKS = (2, 4, 6, 10)


def ahead(df, cols, h, key=("k",)):
    """Attach value of cols at di + h for the same contract (rows absent at t+h are dropped)."""
    fut = df[["di", *key, *cols]].copy()
    fut["di"] = fut.di - h
    return df.merge(fut, on=["di", *key], suffixes=("", f"_h"))


def _month_id(dates):
    return (dates.dt.year * 12 + dates.dt.month).values


def cluster_boot_beta(x, y, months, reps=2000):
    """Pooled OLS slope of y on x (with intercept); bootstrap resampling calendar months."""
    u, inv = np.unique(months, return_inverse=True)
    n = np.bincount(inv); sx = np.bincount(inv, x); sy = np.bincount(inv, y)
    sxx = np.bincount(inv, x * x); sxy = np.bincount(inv, x * y)

    def beta(w):
        N = w @ n; Sx = w @ sx; Sy = w @ sy; Sxx = w @ sxx; Sxy = w @ sxy
        return (Sxy - Sx * Sy / N) / (Sxx - Sx * Sx / N)
    b0 = beta(np.ones(len(u)))
    W = np.stack([np.bincount(RNG.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(reps)])
    bs = np.array([beta(w) for w in W])
    return b0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def regressions(fit, fl):
    rows = []
    specs = [("contract kink, var A", fit, "kA"), ("contract kink, var B", fit, "kB"),
             ("fly kink, var A", fl, "fA"), ("fly kink, var B", fl, "fB"),
             ("fly benchmark", fl, "fbench")]
    for name, df, col in specs:
        for h in (1, 5, 20):
            j = ahead(df[["di", "k", "date", col]].dropna(), [col], h)
            for per, (a, b) in (("2011-17", IS), ("2018-21", OOS)):
                g = j[j.date.between(a, b)]
                x = g[col].values; dy = (g[f"{col}_h"] - g[col]).values
                b0, lo, hi = cluster_boot_beta(x, dy, _month_id(g.date))
                # OOS R^2 of IS beta vs no-change, only meaningful in OOS row
                rows.append(dict(signal=name, h=h, period=per, n=len(g), beta=b0, lo=lo, hi=hi,
                                 sd_bp=1e4 * x.std()))
    out = pd.DataFrame(rows)
    return out


def oos_r2(fit, fl):
    """Use IS slope to predict OOS kink change; R^2 vs 'no change'."""
    rows = []
    for name, df, col in [("fly kink, var A", fl, "fA"), ("fly kink, var B", fl, "fB"),
                          ("fly benchmark", fl, "fbench")]:
        for h in (1, 5, 20):
            j = ahead(df[["di", "k", "date", col]].dropna(), [col], h)
            j["dy"] = j[f"{col}_h"] - j[col]
            a = j[j.date.between(*IS)]; o = j[j.date.between(*OOS)]
            beta = np.polyfit(a[col], a.dy, 1)
            pred = np.polyval(beta, o[col])
            rows.append(dict(signal=name, h=h, oos_r2=1 - ((o.dy - pred) ** 2).sum() / (o.dy ** 2).sum()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- trade
def trades(fl, sig, h, thr, liquid=False):
    """Daily entries; one fly (1 -2 +1, 4 contracts) per eligible centre with |sig| > thr.
    Short the centre when it is rich. Exit after h trading days at settle."""
    df = fl[["di", "k", "date", "cm", "tau", "B", sig, "vol_min", "oi_min"]].dropna(subset=[sig])
    df = df[df[sig].abs() > thr]
    if liquid:
        df = df[(df.vol_min >= 1000) & (df.oi_min >= 10_000)]
    # exit value from the full table (the t+h row exists only if all legs still have > 5 days to expiry)
    full = fl[["di", "k", "B"]].copy(); full["di"] -= h
    j = df.merge(full.rename(columns={"B": "B_h"}), on=["di", "k"])
    j["gross"] = -np.sign(j[sig]) * 2 * (j.B_h - j.B) * MULT     # $ per fly
    return j


def summarize(j, ticks=4, block=3, reps=2000):
    if len(j) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan)
    net = j.gross - 4 * ticks * TICK * MULT
    m = _month_id(j.date)
    u, inv = np.unique(m, return_inverse=True)
    allm = np.arange(u.min(), u.max() + 1)
    s = pd.Series(np.bincount(inv, net), index=u).reindex(allm, fill_value=0).values
    n = pd.Series(np.bincount(inv), index=u).reindex(allm, fill_value=0).values
    T = len(allm); nb = int(np.ceil(T / block))
    st = RNG.integers(0, T - block + 1, (reps, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :T]
    bm = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    pos_m = (s[n > 0] > 0).mean()
    return dict(n=len(j), months=int((n > 0).sum()), mean=net.mean(),
                lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5),
                t=net.mean() / bm.std() if bm.std() > 0 else np.nan, pct_months_pos=100 * pos_m)


def grid_select(fl, signals=("fA", "fB", "fbench"), hs=(1, 5, 20), qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    """Choose (h, threshold) per signal on 2011-17 by the block-bootstrap t of net P&L at 4 ticks."""
    rows = []
    for sig in signals:
        isrows = fl[fl.date.between(*IS)][sig].abs().dropna()
        for q in qs:
            thr = isrows.quantile(q)
            for h in hs:
                j = trades(fl, sig, h, thr)
                for per, (a, b) in (("2011-17", IS), ("2018-21", OOS)):
                    r = summarize(j[j.date.between(a, b)], ticks=4, reps=500)
                    rows.append(dict(signal=sig, q=q, thr_bp=1e4 * thr, h=h, period=per, **r))
    return pd.DataFrame(rows)
