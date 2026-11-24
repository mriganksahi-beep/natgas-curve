"""Frozen tests for the locked 2022-2026 options sample (section 6 of the report).

Step 1 (dev, run once - already done): `python -m src.research.v5_weather_uncertainty.frozen --freeze`
  writes frozen_constants.json (thresholds and coefficients fitted on ALL development Oct-Apr rows).
Step 2 (locked, run once): V5_DIR=data/processed/v5_locked V5_CUT=<panel end> \\
      V5WU_CACHE=data/processed/research_v5_weather_uncertainty_locked \\
      python -m src.research.v5_weather_uncertainty.frozen --test --start 2022-01-01
  Same columns (weather_unc, surface, futures, storage); nothing is re-fitted.  On dev data
  (`--test --start 2011-10-01`) it reproduces the dev reference values in the report.

Tests (rows Oct-Apr, dates >= start; forward-RV rows only where the 20-day window has resolved):
  T-WU1 (primary, falsification): HAR+IV vs HAR+IV+U (UL+UD blocks) with FROZEN coefficients, h=20 leg 1.
         dR2 = R2(alt) - R2(base) (R2 against the frozen dev mean); Clark-West t, HAC 20 lags.
         Study conclusion HOLDS if NOT (dR2 >= 0.01 and CW t >= 1.645).  Dev reference is IN-SAMPLE
         (coefficients fitted on the same rows): dR2 +0.011, t 2.9 - an upper bound; walk-forward OOS was -0.001.
  T-WU2 (IV reaction): event.regressions on the same panel, unc_n_w23 one-run change, standardised with
         the frozen dev mean/sd: same-day slope and d->d+5 slope.  HOLDS if both |t| < 2.
  T-WU3 (tradable): weekly long front straddle (trade.simulate, 1.5% + 2 ticks) when unc_n_w23_m3 >
         frozen dev 67th pct.  HOLDS if mean net per winter <= 0 or its 95% winter-bootstrap CI spans 0.
         Also report the cost-free selection difference (on - off weekly gross) with CI.
  T-WU4 (secondary, the one directional hint): front IV - RV20 (vol pts) in the high (> frozen q67)
         minus the low (< frozen q33) disagreement tercile.  PASS (hint confirmed) if < 0 with one-sided
         winter-bootstrap p < 0.10; report CI.  Dev reference -3.3 vol pts [-8.9, +1.6], p 0.11 (does not pass on dev).
  T-WU5 (Q4 log): weekly gross long-straddle P&L in tight (sdev < -150) x high/low disagreement (frozen q50).
         Report weeks, gas years, means.  'Not testable' if either cell < 10 weeks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data as D
from .forecast import MODELS, design, hac_mean_t

FROZEN = Path(__file__).with_name("frozen_constants.json")
RNG = np.random.default_rng(20261124)


def freeze():
    w = D.weather_unc()
    x = w.unc_n_w23_m3.dropna()
    dl = w.dl_unc_n_w23.dropna()
    b = D.base()
    g = b[(b.h == 20) & (b.leg == 1)].sort_values("date").reset_index(drop=True)
    X = design(g)
    y = np.log(g.rvf)
    coefs = {}
    for name in ("HAR+IV", "HAR+IV+U"):
        cols = MODELS[name]
        ok = X[cols].notna().all(1) & y.notna()
        beta = np.linalg.lstsq(X.loc[ok, cols].values, y[ok].values, rcond=None)[0]
        coefs[name] = dict(zip(cols, beta.tolist()))
    const = {"q33": float(x.quantile(1 / 3)), "q50": float(x.median()), "q67": float(x.quantile(0.67)),
             "dl_mean": float(dl.mean()), "dl_sd": float(dl.std()), "ybar": float(y.mean()), "coefs": coefs,
             "fitted_on": "all development Oct-Apr rows 2010-10..2021-12 (h=20, leg 1)"}
    FROZEN.write_text(json.dumps(const, indent=1))
    return const


def _boot_diff(df, col, grp_col, a, b_, B=2000):
    grp = {k: v for k, v in df.groupby("gy")}
    ug = list(grp)
    est = df[df[grp_col] == a][col].mean() - df[df[grp_col] == b_][col].mean()
    bs = []
    for _ in range(B):
        q = pd.concat([grp[k] for k in RNG.choice(ug, len(ug))])
        bs.append(q[q[grp_col] == a][col].mean() - q[q[grp_col] == b_][col].mean())
    bs = np.array(bs)
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), float(np.nanmean(bs >= 0))


def test(start):
    K = json.loads(FROZEN.read_text())
    start = pd.Timestamp(start)
    out = {}
    b = D.base(force=True)
    # ---- T-WU1
    g = b[(b.h == 20) & (b.leg == 1) & (b.date >= start)].sort_values("date").reset_index(drop=True)
    X = design(g)
    y = np.log(g.rvf)
    preds = {}
    for name, cf in K["coefs"].items():
        cols = list(cf)
        preds[name] = X[cols].values @ np.array([cf[c] for c in cols])
    ok = y.notna() & np.isfinite(preds["HAR+IV"]) & np.isfinite(preds["HAR+IV+U"])
    e0 = (y - preds["HAR+IV"])[ok]
    e1 = (y - preds["HAR+IV+U"])[ok]
    eb = (y - K["ybar"])[ok]
    cw = e0 ** 2 - (e1 ** 2 - (preds["HAR+IV"][ok] - preds["HAR+IV+U"][ok]) ** 2)
    _, t = hac_mean_t(cw.values, 20)
    r0, r1 = 1 - (e0 ** 2).sum() / (eb ** 2).sum(), 1 - (e1 ** 2).sum() / (eb ** 2).sum()
    out["T-WU1"] = {"n": int(ok.sum()), "r2_base": r0, "r2_alt": r1, "dR2": r1 - r0, "cw_t": t,
                    "holds": not (r1 - r0 >= 0.01 and t >= 1.645)}
    # ---- T-WU2
    from .event import daily as ev_daily
    import statsmodels.api as sm
    Q = ev_daily(force=True)
    Q = Q[Q.date >= start].reset_index(drop=True)
    w = D.weather_unc()
    z = (w.loc[Q.date, "dl_unc_n_w23"].values - K["dl_mean"]) / K["dl_sd"]
    iv = np.vstack(Q.iv.values)
    lp = np.vstack(Q.lp.values)
    o = 6
    r = lp[:, o]
    res = {}
    for name, yv, lag in (("same_day", iv[:, o], 5), ("d_to_d5", iv[:, o + 5] - iv[:, o], 5)):
        df = pd.DataFrame({"y": yv, "x": z, "r": r, "ar": np.abs(r)}).dropna()
        m = sm.OLS(df.y, sm.add_constant(df[["x", "r", "ar"]])).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        res[name] = {"b": float(m.params["x"]), "t": float(m.tvalues["x"]), "n": len(df)}
    res["holds"] = all(abs(v["t"]) < 2 for v in res.values() if isinstance(v, dict))
    out["T-WU2"] = res
    # ---- T-WU3 + T-WU5
    from .trade import Market, decision_days, simulate, weekly_outcomes
    mk = Market()
    S = pd.DataFrame(index=w.index)
    S["unc_n_w23"] = w.unc_n_w23_m3
    S["sdev"] = D.storage_dev(w.index)
    S = S[S.index >= start]
    on = (S.unc_n_w23 > K["q67"]).astype(int)
    import src.research.v5_weather_uncertainty.trade as TR
    TR.START = start
    df, tr = simulate(mk, S, on, c=0.015)
    df["gy"] = D.gas_year(df.date)
    per = df.groupby("gy").net.sum()
    bs = per.values[RNG.integers(0, len(per), (2000, len(per)))].mean(1)
    Wk = weekly_outcomes(mk, S)
    Wk["gy"] = D.gas_year(Wk.date)
    Wk["on"] = Wk.date.map(on).fillna(0).astype(int)
    sel = _boot_diff(Wk, "gross", "on", 1, 0)
    out["T-WU3"] = {"net_per_winter": float(per.mean()), "lo": float(np.percentile(bs, 2.5)),
                    "hi": float(np.percentile(bs, 97.5)), "winters": len(per), "weeks_on": int(Wk.on.sum()),
                    "selection_diff": sel[0], "sel_lo": sel[1], "sel_hi": sel[2],
                    "holds": bool(per.mean() <= 0 or np.percentile(bs, 2.5) <= 0)}
    sd = S.sdev.reindex(Wk.date).values
    Wk["cell"] = np.where(sd < -150, np.where(Wk.date.map(S.unc_n_w23).values > K["q50"], "tight_high", "tight_low"), "other")
    cells = Wk.groupby("cell").agg(weeks=("gross", "size"), gy=("gy", "nunique"), mean=("gross", "mean"))
    out["T-WU5"] = cells.reset_index().to_dict("records")
    # ---- T-WU4
    g = g.join(w[["unc_n_w23_m3"]], on="date", rsuffix="_w")
    xx = g["unc_n_w23_m3"]
    g["terc"] = np.where(xx > K["q67"], "high", np.where(xx < K["q33"], "low", "mid"))
    g["gap"] = (g.atm_iv - g.rvf) * 100
    est, lo, hi, p = _boot_diff(g.dropna(subset=["gap"]), "gap", "terc", "high", "low")
    out["T-WU4"] = {"diff": est, "lo": lo, "hi": hi, "p_one_sided": p, "n_high": int((g.terc == "high").sum()),
                    "n_low": int((g.terc == "low").sum()), "pass": bool(est < 0 and p < 0.10)}
    return out


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        print(json.dumps(freeze(), indent=1)[:600])
    if "--test" in sys.argv:
        st = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else "2022-01-01"
        r = test(st)
        print(json.dumps(r, indent=1, default=float))
        (D.CACHE / f"frozen_result_{st}.json").write_text(json.dumps(r, indent=1, default=float))
