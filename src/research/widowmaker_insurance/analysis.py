"""Actuarial value, premium, theta, premium test, conditional rule and price-only check."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import data as D
from . import payout as P

K_LARGE = 0.20      # "large" expiry spread, $/MMBtu (above every non-2014 expiry 2011-21; fixed a priori)
FLOOR = 700.0       # physical floor on end-March storage scenarios, Bcf (near the record low)
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


# ---------------------------------------------------------------- payout curve inputs
def payout_table():
    return D.avg5_table(min_prior=1).join(D.expiry_table(), how="left")


# ---------------------------------------------------------------- actuarial distribution
def actuarial(panel, m2, curves, centre="m2", mode="wf", floor=FLOOR, scaled=False, seed=1):
    """For each weekly row with a Model 2 origin: distribution of the expiry spread.
    centre='m2': Model 2 percentiles as they are; '5y': re-centred so the median = 5-year path."""
    rng = np.random.default_rng(seed)
    tab = D.avg5_table(min_prior=1)
    m2 = m2.set_index("origin")
    rows = []
    for _, r in panel.iterrows():
        if r.origin not in m2.index:
            continue
        m = m2.loc[r.origin]
        c = curves.get((r.Y, mode))
        if c is None:
            continue
        s = m[D.PCOLS].values.astype(float)
        if centre == "5y":
            s = s - m["p50"] + m["benchmark"]
        s = np.maximum(s, floor)
        x = s - tab.avg5[r.Y]
        f = P.predict(c["ps"], x)                         # [B, 99]: parameter uncertainty
        y = f + rng.choice(c["e"], size=f.shape)          # + residual noise
        if scaled:
            y = y * r.J                                    # % of April -> $ with April at t
        y = y.ravel()
        out = {"origin": r.origin, "av": y.mean(), "p_above_mkt": (y > r.S).mean(),
               "p_large": (y > K_LARGE).mean(), "av_point": (P.predict(c["p0"], x)[0] * (r.J if scaled else 1)).mean(),
               "x_med": np.median(x)}
        for q, v in zip(QS, np.quantile(y, QS)):
            out[f"q{int(q * 100)}"] = v
        rows.append(out)
    a = pd.DataFrame(rows)
    res = panel.merge(a, on="origin", how="inner")
    res["prem"] = res.S - res.av
    res["dS_exp"] = res.S_exp - res.S
    res["dS_4w"] = res.S_4w - res.S
    return res


# ---------------------------------------------------------------- bootstrap helpers
def winter_boot(df, stat, B=2000, seed=0, col="winter"):
    rng = np.random.default_rng(seed)
    ws = df[col].unique()
    g = {w: df[df[col] == w] for w in ws}
    vals = []
    for _ in range(B):
        pick = rng.choice(ws, len(ws), replace=True)
        try:
            vals.append(stat(pd.concat([g[w] for w in pick])))
        except Exception:
            continue
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)


def theta_curve(res, col="prem"):
    r = res.copy()
    r["wk"] = np.clip(np.floor(r.wte / 2) * 2, 0, 16).astype(int)   # 2-week buckets
    rows = []
    for wk, g in r.groupby("wk"):
        lo, hi = winter_boot(g, lambda d: d[col].mean(), B=1000)
        rows.append({"wte_bucket": f"{wk}-{wk + 2}" if wk < 16 else "16+", "n": len(g), "mean": g[col].mean(),
                     "lo": lo, "hi": hi, "S": g.S.mean(), "av": g.av.mean()})
    return pd.DataFrame(rows)


def premium_regression(res, xcol="prem", ycol="dS_exp"):
    d = res.dropna(subset=[ycol, xcol])
    X = sm.add_constant(-d[xcol].values)
    fit = sm.OLS(d[ycol].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})

    def slope(z):
        return np.polyfit(-z[xcol].values, z[ycol].values, 1)[0]

    def icpt(z):
        return np.polyfit(-z[xcol].values, z[ycol].values, 1)[1]
    lo, hi = winter_boot(d, slope)
    ilo, ihi = winter_boot(d, icpt)
    return {"x": xcol, "y": ycol, "n": len(d), "winters": d.winter.nunique(), "slope": fit.params[1],
            "slope_hac_se": fit.bse[1], "slope_boot_lo": lo, "slope_boot_hi": hi, "intercept": fit.params[0],
            "icpt_boot_lo": ilo, "icpt_boot_hi": ihi, "r2": fit.rsquared,
            "mean_dS": d[ycol].mean(), "mean_minus_prem": -d[xcol].mean()}


# ---------------------------------------------------------------- trading rules
def trades(res, mask, hold="exp"):
    """Short one spread at each flagged week. P&L $ per spread net of the $160 round trip."""
    d = res[mask].copy()
    ycol = "S_exp" if hold == "exp" else "S_4w"
    d = d.dropna(subset=[ycol])
    d["gross"] = -(d[ycol] - d.S) * D.MULT
    d["net"] = d.gross - D.ROUND_TRIP
    return d


def summarize(t):
    if len(t) == 0:
        return {"trades": 0, "winters": 0, "mean_net": np.nan, "lo": np.nan, "hi": np.nan, "hit": np.nan,
                "worst": np.nan, "total": 0.0}
    lo, hi = winter_boot(t, lambda d: d.net.mean(), B=1000) if t.winter.nunique() > 1 else (np.nan, np.nan)
    return {"trades": len(t), "winters": t.winter.nunique(), "mean_net": t.net.mean(), "lo": lo, "hi": hi,
            "hit": (t.net > 0).mean(), "worst": t.net.min(), "total": t.net.sum()}


def mae_path(sp, t):
    """Worst mark-to-market loss ($/spread, before costs) of each trade while it was open."""
    out = []
    for _, r in t.iterrows():
        d = sp[r.Y].S
        end = d.index[-1] if r.get("hold", "exp") == "exp" else r.date_4w
        path = d[(d.index >= r.obs) & (d.index <= end)]
        out.append(-(path.max() - r.S) * D.MULT)
    return np.array(out)


# ---------------------------------------------------------------- price-only check
def price_only(sp):
    rows = []
    for Y, d in sp.items():
        if Y >= 2022:
            continue
        S = d.S
        nov = S[(S.index >= pd.Timestamp(Y - 1, 11, 1)) & (S.index < pd.Timestamp(Y - 1, 12, 1))]
        win = S[S.index >= nov.index[0]]
        entry, exitv = nov.iloc[0], S.iloc[-1]
        rows.append({"Y": Y, "winter": D.winter_label(Y), "S_nov1": entry, "S_novavg": nov.mean(),
                     "S_exp": exitv, "nov_minus_exp": nov.mean() - exitv,
                     "net_short_nov1": -(exitv - entry) * D.MULT - D.ROUND_TRIP,
                     "worst_mtm": -(win.max() - entry) * D.MULT, "S_max": win.max(),
                     "date_max": win.idxmax()})
    return pd.DataFrame(rows).set_index("Y")
