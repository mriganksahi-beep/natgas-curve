"""Parts 1-2: weather sensitivity of curve moves, its walk-forward inversion, and skill.

Two readings of "market-implied HDD anomaly" for the next 2 weeks (wx2 = a1 + a2):
  structural ("literal inversion"): fit x = alpha + beta * wx2 (+ beta2 * wx2 * S_prev for the
      scaled spread) on past weeks, then implied = (x - alpha) / beta. HDD units, unshrunk.
  calibrated: regress wx2 on the curve features on past weeks and predict. This is the
      MSE-optimal reading; it equals the structural one shrunk by about R^2.
Walk-forward: a row at F_t is predicted from rows with F' + 14 days <= F_t (their wx2 fully
realised), at least MIN_TRAIN rows. Evaluation: winters 2012-13 to 2021-22 (Nov-Dec only).
"""
import numpy as np
import pandas as pd

MIN_TRAIN = 30
EVAL_FIRST = "2012-13"
DEV = ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17"]
CHECK = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]


def _lstsq(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def _design(d, xs):
    return np.column_stack([np.ones(len(d))] + [d[x].values.astype(float) for x in xs])


def walk_forward(df, fn, need):
    """fn(train, row) -> float. need: columns that must be non-NaN in train (incl. target)."""
    out = pd.Series(np.nan, index=df.index)
    for i, r in df.iterrows():
        tr = df[(df.F + pd.Timedelta(days=14) <= r.F)].dropna(subset=need)
        if len(tr) < MIN_TRAIN:
            continue
        try:
            out[i] = fn(tr, r)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            pass
    return out


def calibrated(xs, target="wx2"):
    def fn(tr, r):
        if any(pd.isna(r[x]) for x in xs):
            return np.nan
        b = _lstsq(_design(tr, xs), tr[target].values)
        return float(np.r_[1.0, [r[x] for x in xs]] @ b)
    return fn, xs + [target]


def structural(x, target="wx2", scaled=False):
    """Literal inversion of x = a + b*wx2 (+ c*wx2*S_prev)."""
    def fn(tr, r):
        if pd.isna(r[x]) or (scaled and pd.isna(r["S_prev"])):
            return np.nan
        if scaled:
            X = np.column_stack([np.ones(len(tr)), tr[target], tr[target] * tr["S_prev"]])
            a, b, c = _lstsq(X, tr[x].values)
            dens = b + c * tr["S_prev"].values
            floor = 0.25 * np.median(dens[dens > 0])
            den = max(b + c * r["S_prev"], floor)
        else:
            a, den = _lstsq(_design(tr, [target]), tr[x].values)
        return float((r[x] - a) / den)
    need = [x, target] + (["S_prev"] if scaled else [])
    return fn, need


def gls(xs, target="wx2", ridge=0.05):
    """Structural GLS combination: x_i = a_i + b_i*wx2 + e_i, Sigma from residuals."""
    def fn(tr, r):
        if any(pd.isna(r[x]) for x in xs):
            return np.nan
        A, B, E = [], [], []
        for x in xs:
            a, b = _lstsq(_design(tr, [target]), tr[x].values)
            A.append(a); B.append(b); E.append(tr[x].values - a - b * tr[target].values)
        A, B = np.array(A), np.array(B)
        S = np.cov(np.array(E))
        S = S + ridge * np.diag(np.diag(S))  # small ridge: front/second are ~0.99 correlated
        Si = np.linalg.inv(S)
        xv = np.array([r[x] for x in xs]) - A
        return float(B @ Si @ xv / (B @ Si @ B))
    return fn, xs + [target]


def add_scaled_feature(df):
    """Feature for calibrated use of the spread: dS / (0.04 + max(S_prev, 0)).
    0.04 is the intercept implied by the regimes_weather convexity fit (6c at S=0.10,
    19c at S=0.40 per +100 HDD); fixed a priori, not tuned here."""
    df = df.copy()
    df["dS_sc"] = df["dS"] / (0.04 + df["S_prev"].clip(lower=0))
    return df


def build_predictions(df):
    df = add_scaled_feature(df)
    preds = {}
    preds["clim"] = pd.Series(0.0, index=df.index)
    preds["pers_naive"] = df["pers"]
    specs = {
        "pers": calibrated(["a0"]),
        "front": calibrated(["r_front"]),
        "second": calibrated(["r_second"]),
        "strip": calibrated(["r_strip"]),
        "marapr": calibrated(["dS"]),
        "marapr_sc": calibrated(["dS_sc"]),
        "combo3": calibrated(["r_front", "r_second", "dS"]),
        "combo3_sc": calibrated(["r_front", "r_second", "dS_sc"]),
        "combo3+pers": calibrated(["r_front", "r_second", "dS", "a0"]),
        "front+pers": calibrated(["r_front", "a0"]),
        # structural (literal inversion), HDD units
        "S_front": structural("r_front"),
        "S_second": structural("r_second"),
        "S_marapr": structural("dS"),
        "S_marapr_sc": structural("dS", scaled=True),
        "S_gls3": gls(["r_front", "r_second", "dS"]),
    }
    for k, (fn, need) in specs.items():
        preds[k] = walk_forward(df, fn, need)
    # calibrated GLS: regress wx2 on the walk-forward structural GLS estimate
    df["_gls"] = preds["S_gls3"]
    fn, need = calibrated(["_gls"])
    preds["gls3"] = walk_forward(df, fn, need)
    # same machinery for next-week anomaly a1 alone (used in the storage nowcast)
    for k, xs in (("a1_front", ["r_front"]), ("a1_combo3", ["r_front", "r_second", "dS"]),
                  ("a1_pers", ["a0"]), ("a1_combo3+pers", ["r_front", "r_second", "dS", "a0"])):
        fn, need = calibrated(xs, target="a1")
        preds[k] = walk_forward(df, fn, need)
    P = pd.DataFrame(preds)
    return df.drop(columns="_gls"), P


def metrics(y, p):
    m = (~np.isnan(y)) & (~np.isnan(p))
    y, p = y[m], p[m]
    if len(y) < 3:
        return dict(n=len(y), corr=np.nan, ss=np.nan, hit=np.nan)
    corr = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan
    ss = 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)
    nz = (y != 0) & (p != 0)
    hit = np.mean(np.sign(y[nz]) == np.sign(p[nz])) if nz.any() else np.nan
    return dict(n=len(y), corr=corr, ss=ss, hit=hit)


def boot_metrics(E, models, n_boot=2000, seed=0, ref="pers"):
    """Winter-block bootstrap of corr, skill score and hit rate for each model, and the
    difference to `ref` (paired)."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.values for k, g in E.groupby("winter")}
    keys = list(groups)
    res = {m: {"corr": [], "ss": [], "hit": [], "dss": [], "dcorr": [], "dhit": []} for m in models}
    y_all = E["wx2"].values
    pos = {i: j for j, i in enumerate(E.index)}
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        idx = np.concatenate([[pos[i] for i in groups[k]] for k in pick])
        y = y_all[idx]
        mr = metrics(y, E[ref].values[idx])
        for m in models:
            mm = metrics(y, E[m].values[idx])
            for s in ("corr", "ss", "hit"):
                res[m][s].append(mm[s])
                res[m]["d" + s].append(mm[s] - mr[s])
    return res
