"""Part 3: when the curve's implied weather view is extreme, does the move reverse or persist?

For instrument x (front-month % move, or March-April $ move):
  implied_t   = walk-forward literal inversion (x - a) / b   (HDD units)
  ext_t       = |implied_t| above the walk-forward 90th percentile of past |implied| (>= 30 past
                weeks). The literal inversion is ~3.5x as volatile as realised wx2, so "beyond what
                realised weather delivers" (> q95 of |wx2|) flags about half of all weeks; the top
                decile corresponds to implied anomalies of roughly 2-3x the largest realised ones.
  u_t         = x_t - (a + b*wx2_t [+ c*wx2_t*S_prev]): the part of the move NOT justified by the
                weather that followed (ex post), i.e. a "premium" move. ext_u = top decile of |u|.
Reversal regression (HAC, lags h):
  fwd_h = c0 + c1*z_t + c2*z_t*ext_t + c3*ext_t + sum_k g_k * a_k (k = 1..h+2)
  slope in extreme weeks = c1 + c2 (-1 = full reversal, 0 = persistence of the level).
The future-weather controls stop the reversal test from being mechanical (u is built with future
weather, which also drives later moves).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .skill import _lstsq, _design, MIN_TRAIN


def wf_structural_resid(df, x, scaled):
    """Walk-forward (a, b, c) from rows with F'+14 <= F_t; return implied and residual u."""
    imp = pd.Series(np.nan, index=df.index)
    u = pd.Series(np.nan, index=df.index)
    for i, r in df.iterrows():
        tr = df[df.F + pd.Timedelta(days=14) <= r.F].dropna(subset=[x, "wx2", "S_prev"])
        if len(tr) < MIN_TRAIN or pd.isna(r[x]):
            continue
        if scaled:
            X = np.column_stack([np.ones(len(tr)), tr.wx2, tr.wx2 * tr.S_prev])
            a, b, c = _lstsq(X, tr[x].values)
            dens = b + c * tr.S_prev.values
            den = max(b + c * r.S_prev, 0.25 * np.median(dens[dens > 0]))
        else:
            a, b = _lstsq(_design(tr, ["wx2"]), tr[x].values)
            den = b
        imp[i] = (r[x] - a) / den
        if not pd.isna(r.wx2):
            u[i] = r[x] - a - den * r.wx2
    return imp, u


def wf_top_decile(s, F):
    flag = pd.Series(np.nan, index=s.index)
    for i in s.index:
        past = s[(F + pd.Timedelta(days=14) <= F[i])].dropna().abs()
        if len(past) >= MIN_TRAIN and not pd.isna(s[i]):
            flag[i] = float(abs(s[i]) > past.quantile(0.9))
    return flag


def reversal(d, y, z, ext, h, n_boot=2000, seed=0):
    ctrls = [f"a{k}" for k in range(1, h + 3)]
    dd = d.dropna(subset=[y, z, ext] + ctrls).copy()
    dd["zx"] = dd[z] * dd[ext]

    def fit(sub):
        X = sm.add_constant(sub[[z, "zx", ext] + ctrls].astype(float))
        return sm.OLS(sub[y].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": h})

    m = fit(dd)
    b_ext = m.params[z] + m.params["zx"]
    cov = m.cov_params()
    se_ext = np.sqrt(cov.loc[z, z] + cov.loc["zx", "zx"] + 2 * cov.loc[z, "zx"])
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in dd.groupby("winter")}
    keys = list(groups)
    bs_e, bs_n = [], []
    for _ in range(n_boot):
        sub = pd.concat([groups[k] for k in rng.choice(keys, len(keys))], ignore_index=True)
        if sub[ext].sum() < 3:
            continue
        try:
            mm = fit(sub)
            bs_e.append(mm.params[z] + mm.params["zx"]); bs_n.append(mm.params[z])
        except Exception:
            continue
    return dict(h=h, n=len(dd), n_ext=int(dd[ext].sum()),
                slope_ext=b_ext, t_ext=b_ext / se_ext,
                ci_ext=tuple(np.percentile(bs_e, [2.5, 97.5])),
                slope_norm=m.params[z], t_norm=m.tvalues[z],
                ci_norm=tuple(np.percentile(bs_n, [2.5, 97.5])))
