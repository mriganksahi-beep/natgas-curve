"""Storage-space option: payout curve, actuarial value, premium, drift tests, trade grid,
price-only drift. Data < 2022 only.

Conventions
  * Contango C = far - near ($/MMBtu). "V-X" in the brief = -C_XV. Long V-X = short contango.
  * Targets: C_XV (Nov-Oct), C_VU (Oct-Sep), C_FV (Jan(next)-Oct), all of the same autumn.
  * Payout: C at the near contract's last trading day (XV and FV: October's; VU: September's).
  * Fullness: dev = S_T - avg5(T) (prior-5-year calendar average, >=3 yrs), or
    head = max level before Jan 1 of the year - S_T (walk-forward capacity proxy).
  * Payout family (pre-declared main): C = a + b * exp(dev / 150 Bcf) [convex in fullness].
    Alternatives: linear in dev; exp(-head/150); lambda = 100, 250.
  * Actuarial value at origin t: mean over Model 2's 99 percentile levels of payout(level),
    with the payout fitted only on autumns < year(t).
    "raw": Model 2 percentiles as simulated; "robust": percentiles shifted so that the median
    equals the 5-year path (Model 2 supplies the shape, the 5-year path the centre);
    "point5": payout(5-year path) (no dispersion).
  * Market at origin t: settle on the first trading day >= t + 6 days (storage of week t public).
  * Premium pi = market - actuarial. Drift D = C(expiry) - C(market date).
  * Cost: 4 ticks/contract/side -> 2 contracts x 2 sides x 4 x $10 = $160 per spread round trip.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.research.storage_space_option.data import (CACHE, MULT, SPREADS, autumn_table,
                                                    five_year_path, load_storage, spread_series,
                                                    wide_settles)

RNG = np.random.default_rng(7)
PCT_COLS = [f"p{i}" for i in range(1, 100)]
TICKS = 4
COST = 2 * 2 * TICKS * 10.0  # $ per spread round trip
N_VARIANTS = {}


# ---------------------------------------------------------------- payout curve
def design(x, form, lam):
    if form == "lin":
        return np.column_stack([np.ones_like(x), x])
    return np.column_stack([np.ones_like(x), np.exp(x / lam)])


def fit_payout(df, ycol, xcol, form="exp", lam=150.0):
    d = df.dropna(subset=[ycol, xcol])
    x = d[xcol].values.astype(float)
    if xcol == "head_T":
        x = -x
    X = design(x, form, lam)
    beta, *_ = np.linalg.lstsq(X, d[ycol].values, rcond=None)
    return beta


def payout(beta, x, xcol, form="exp", lam=150.0):
    x = np.asarray(x, dtype=float)
    if xcol == "head_T":
        x = -x
    return design(x.ravel(), form, lam) @ beta


def payout_walkforward(a, ycol, xcol, form, lam, min_train=3):
    """Out-of-sample error of payout(realised storage) vs the prior-years mean."""
    rows = []
    for y in a.year:
        tr = a[(a.year < y)].dropna(subset=[ycol, xcol])
        te = a[a.year == y].dropna(subset=[ycol, xcol])
        if len(tr) < min_train or te.empty:
            continue
        b = fit_payout(tr, ycol, xcol, form, lam)
        rows.append({"year": y, "pred": float(payout(b, te[xcol].values, xcol, form, lam)[0]),
                     "naive": tr[ycol].mean(), "actual": float(te[ycol].iloc[0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- panel of origins
def origins_panel(a, w, m2):
    """One row per (summer origin week, spread): market, drift, actuarial variants."""
    s = load_storage()
    tdates = w.index
    rows = []
    for _, o in m2.iterrows():
        t, y = o.origin, o.origin.year
        rel = t + pd.Timedelta(days=6)
        i = tdates.searchsorted(rel)
        p = tdates[i]
        ar = a.set_index("year").loc[y]
        lev = o[PCT_COLS].values.astype(float)
        bench = five_year_path(s, t, ar["T"])
        for name, (nm, fo, fm) in SPREADS.items():
            exp_day = ar["U_last"] if name == "VU" else ar["V_last"]
            if p >= exp_day:
                continue
            sp = spread_series(w, y, nm, y + fo, fm)
            rows.append({"origin": t, "year": y, "mdate": p, "spread": name,
                         "weeks_to_go": o.weeks_to_go, "market": float(sp[p]),
                         "expiry": float(sp[exp_day]), "p50": lev[49], "bench": bench,
                         "bench_m2": o.benchmark, "sd": o.sd, "lev": lev,
                         "avg5_T": ar["avg5_T"], "max_prior": ar["max_prior"], "actual": ar["S_T"],
                         "days_to_exp": int((tdates[(tdates > p) & (tdates <= exp_day)]).size)})
    P = pd.DataFrame(rows)
    P["drift"] = P.expiry - P.market
    return P


def to_x(lev, row, xcol):
    return lev - row.avg5_T if xcol == "dev_T" else row.max_prior - lev


def add_actuarial(P, a, xcol="dev_T", form="exp", lam=150.0, tag="main", clip=False, cap=None):
    """clip: fullness limited to the training range (no extrapolation of the payout);
    cap: simulated levels limited to a physical capacity (Bcf). Both are post-hoc diagnostics."""
    betas, rng_ = {}, {}
    out = {f"act_raw_{tag}": [], f"act_rob_{tag}": [], f"act_pt5_{tag}": [], f"act_ptm_{tag}": []}
    for _, r in P.iterrows():
        key = (r.spread, r.year)
        if key not in betas:
            tr = a[a.year < r.year].dropna(subset=[f"{r.spread}_exp", xcol])
            betas[key] = fit_payout(tr, f"{r.spread}_exp", xcol, form, lam)
            rng_[key] = (tr[xcol].min(), tr[xcol].max())
        b = betas[key]
        lo_, hi_ = rng_[key] if clip else (-np.inf, np.inf)
        lev = r.lev if cap is None else np.minimum(r.lev, cap)
        rob = lev - r.p50 + r.bench
        if cap is not None:
            rob = np.minimum(rob, cap)
        _x = to_x
        to_x_ = lambda L, rr, xc: np.clip(_x(L, rr, xc), lo_, hi_)
        out[f"act_raw_{tag}"].append(payout(b, to_x_(lev, r, xcol), xcol, form, lam).mean())
        out[f"act_rob_{tag}"].append(payout(b, to_x_(rob, r, xcol), xcol, form, lam).mean())
        out[f"act_pt5_{tag}"].append(float(payout(b, to_x_(np.array([r.bench]), r, xcol), xcol, form, lam)[0]))
        out[f"act_ptm_{tag}"].append(float(payout(b, to_x_(np.array([r.p50]), r, xcol), xcol, form, lam)[0]))
    for k, v in out.items():
        P[k] = v
        P["prem" + k[3:]] = P.market - P[k]
    return P


# ---------------------------------------------------------------- statistics
def block_boot_slope(d, xcol, ycol, B=4000):
    yrs = d.year.unique()
    g = {y: d[d.year == y] for y in yrs}
    bs = []
    for _ in range(B):
        s = pd.concat([g[y] for y in RNG.choice(yrs, len(yrs))])
        if s[xcol].std() == 0:
            continue
        bs.append(np.polyfit(s[xcol], s[ycol], 1)[0])
    return np.percentile(bs, [2.5, 97.5])


def drift_regression(P, prem, spread="XV"):
    d = P[(P.spread == spread)].dropna(subset=[prem, "drift"]).copy()
    d["neg"] = -d[prem]
    b = np.polyfit(d.neg, d.drift, 1)
    lo, hi = block_boot_slope(d, "neg", "drift")
    loo = [np.polyfit(d[d.year != y].neg, d[d.year != y].drift, 1)[0] for y in d.year.unique()]
    r = np.corrcoef(d.neg, d.drift)[0, 1]
    return {"spread": spread, "premium": prem, "n": len(d), "years": d.year.nunique(),
            "slope": b[0], "ci_lo": lo, "ci_hi": hi, "loo_min": min(loo), "loo_max": max(loo),
            "r2": r ** 2, "mean_prem": d[prem].mean()}


def year_boot_mean(vals_by_year, B=4000):
    v = np.array(vals_by_year, dtype=float)
    bs = [RNG.choice(v, len(v)).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def trade_grid(P, prem, spread="XV", thetas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10), first_only=False):
    """Long V-X (short contango) when prem > theta; short V-X when prem < -theta. Hold to expiry."""
    rows = []
    d = P[P.spread == spread].sort_values("origin")
    for th in thetas:
        sig = np.where(d[prem] > th, -1, np.where(d[prem] < -th, 1, 0))  # position in contango
        tr = d.assign(pos=sig)
        tr = tr[tr.pos != 0]
        if first_only:
            tr = tr.groupby("year").head(1)
        pnl = tr.pos * tr.drift * MULT - COST
        by_year = pnl.groupby(tr.year).mean()
        ci = year_boot_mean(by_year.values) if len(by_year) > 1 else [np.nan, np.nan]
        rows.append({"premium": prem, "theta": th, "first_only": first_only, "trades": len(tr),
                     "years": len(by_year), "mean_$": pnl.mean() if len(tr) else np.nan,
                     "mean_by_year_$": by_year.mean() if len(by_year) else np.nan,
                     "ci_lo": ci[0], "ci_hi": ci[1], "pct_years_pos": (by_year > 0).mean() if len(by_year) else np.nan,
                     "ex2020_$": pnl[tr.year != 2020].mean() if (tr.year != 2020).any() else np.nan,
                     "by_year": {int(k): round(v) for k, v in by_year.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- price-only 2010-2021
def price_only(a, w):
    rows = []
    for _, r in a.iterrows():
        y = r.year
        for name, (nm, fo, fm) in SPREADS.items():
            exp_day = r.U_last if name == "VU" else r.V_last
            sp = spread_series(w, y, nm, y + fo, fm)
            sp = sp[sp.index <= exp_day]
            end_summer = pd.Timestamp(y, 7, 31) if name == "VU" else pd.Timestamp(y, 8, 31)
            sm = sp[(sp.index >= pd.Timestamp(y, 4, 1)) & (sp.index <= end_summer)]
            if sm.empty:
                continue
            # short-contango P&L for an entry every trading day of the window, held to expiry
            pnl = (-(sp[exp_day] - sm) * MULT - COST)
            rows.append({"year": y, "spread": name, "summer_avg": sm.mean(), "expiry": sp[exp_day],
                         "drift": sp[exp_day] - sm.mean(), "short_pnl_$": pnl.mean(),
                         "start": sm.index.min()})
    return pd.DataFrame(rows)


def summarize_price_only(po):
    rows = []
    for (name, per), g in po.assign(per=np.where(po.year <= 2017, "2010-17", "2018-21")).groupby(["spread", "per"]):
        rows.append(_po_row(name, per, g))
    for name, g in po.groupby("spread"):
        rows.append(_po_row(name, "2010-21", g))
        rows.append(_po_row(name, "2010-21 ex2020", g[g.year != 2020]))
    return pd.DataFrame(rows)


def _po_row(name, per, g):
    ci = year_boot_mean(g.drift.values)
    t = stats.wilcoxon(g.drift.values) if len(g) >= 5 else None
    return {"spread": name, "period": per, "n_years": len(g), "mean_drift_c": 100 * g.drift.mean(),
            "median_drift_c": 100 * g.drift.median(), "ci_lo_c": 100 * ci[0], "ci_hi_c": 100 * ci[1],
            "n_neg": int((g.drift < 0).sum()), "wilcoxon_p": t.pvalue if t else np.nan,
            "short_pnl_$": g["short_pnl_$"].mean()}


def price_only_by_month(a, w):
    """XV drift to expiry by entry month (monthly average contango vs expiry), 2010-21."""
    rows = []
    for _, r in a.iterrows():
        sp = spread_series(w, r.year, 10, r.year, 11)
        sp = sp[sp.index <= r.V_last]
        for mth in range(4, 10):
            m = sp[sp.index.month == mth]
            if len(m):
                rows.append({"year": r.year, "month": mth, "drift": sp[r.V_last] - m.mean()})
    d = pd.DataFrame(rows)
    g = d.groupby("month").drift
    out = pd.DataFrame({"n": g.size(), "mean_c": 100 * g.mean(), "median_c": 100 * g.median(),
                        "frac_neg": g.apply(lambda v: (v < 0).mean()),
                        "mean_ex2020_c": 100 * d[d.year != 2020].groupby("month").drift.mean()})
    return out


# ---------------------------------------------------------------- main
def run():
    w = wide_settles()
    a = autumn_table()
    assert a.V_last.max() < pd.Timestamp("2022-01-01")
    m2 = pd.read_csv(CACHE / "model2_summer_percentiles_dev.csv", parse_dates=["origin", "target"])
    m2 = m2[m2.origin <= "2021-12-31"]
    res = {}

    # 1. relations at expiry
    rel = []
    for name in SPREADS:
        for xcol, yrs in (("dev_T", a.year >= 2013), ("head_T", a.year >= 2011)):
            d = a[yrs].dropna(subset=[f"{name}_exp", xcol])
            rho, p = stats.spearmanr(d[xcol], d[f"{name}_exp"])
            bs = []
            for _ in range(4000):
                s = d.sample(len(d), replace=True, random_state=int(RNG.integers(1e9)))
                if s[xcol].nunique() > 2:
                    bs.append(stats.spearmanr(s[xcol], s[f"{name}_exp"])[0])
            rel.append({"spread": name, "x": xcol, "n": len(d), "spearman": rho, "p": p,
                        "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5),
                        "spearman_ex2020": stats.spearmanr(d[d.year != 2020][xcol], d[d.year != 2020][f"{name}_exp"])[0]})
    res["relations"] = pd.DataFrame(rel)

    # 2. payout curve: in-sample fits + walk-forward error vs naive
    pw = []
    for name in SPREADS:
        for xcol, form, lam in (("dev_T", "exp", 150), ("dev_T", "lin", 150), ("dev_T", "exp", 100),
                                ("dev_T", "exp", 250), ("head_T", "exp", 150)):
            wf = payout_walkforward(a, f"{name}_exp", xcol, form, lam)
            b = fit_payout(a[a.year >= 2011], f"{name}_exp", xcol, form, lam)
            pw.append({"spread": name, "x": xcol, "form": form, "lam": lam, "a": b[0], "b": b[1],
                       "wf_years": f"{wf.year.min()}-{wf.year.max()}", "n_wf": len(wf),
                       "mae_model_c": 100 * (wf.pred - wf.actual).abs().mean(),
                       "mae_naive_c": 100 * (wf.naive - wf.actual).abs().mean(),
                       "mae_model_ex2020_c": 100 * (wf.pred - wf.actual)[wf.year != 2020].abs().mean(),
                       "mae_naive_ex2020_c": 100 * (wf.naive - wf.actual)[wf.year != 2020].abs().mean()})
    res["payout"] = pd.DataFrame(pw)
    N_VARIANTS["payout_forms"] = len(pw)

    # 3. origins panel + actuarial values
    P = origins_panel(a, w, m2)
    P = add_actuarial(P, a, "dev_T", "exp", 150, "main")
    P = add_actuarial(P, a, "dev_T", "lin", 150, "lin")
    P = add_actuarial(P, a, "head_T", "exp", 150, "head")
    # post-hoc diagnostics (added after seeing that the uncapped convex curve is dominated by
    # extrapolation into Model 2's physically impossible >4.3 Tcf tail)
    P = add_actuarial(P, a, "dev_T", "exp", 150, "clip", clip=True)
    P = add_actuarial(P, a, "dev_T", "exp", 150, "cap42", cap=4200.0)
    N_VARIANTS["actuarial"] = 5 * 4

    # Model 2 summer bias
    m2b = m2.assign(bias=m2.p50 - m2.actual, bbias=m2.benchmark - m2.actual,
                    month=m2.origin.dt.month, year=m2.origin.dt.year)
    res["bias_year"] = m2b.groupby("year")[["bias", "bbias", "sd"]].mean()
    res["bias_month"] = m2b.groupby("month")[["bias", "bbias", "sd"]].mean()
    res["bias_all"] = {"m2_mean": m2b.bias.mean(), "m2_mae": m2b.bias.abs().mean(),
                       "bench_mean": m2b.bbias.mean(), "bench_mae": m2b.bbias.abs().mean(),
                       "pit_mean": float(np.mean([(m2b.loc[i, PCT_COLS].values < m2b.loc[i, "actual"]).mean()
                                                  for i in m2b.index]))}

    # premium by month (decay) and by year
    P["month"] = P.mdate.dt.month
    prem_cols = [c for c in P.columns if c.startswith("prem_")]
    res["prem_month"] = P.groupby(["spread", "month"])[["market", "act_rob_main", "prem_rob_main",
                                                        "prem_raw_main", "drift"]].mean()
    res["prem_year"] = P.groupby(["spread", "year"])[["market", "act_rob_main", "prem_rob_main",
                                                      "prem_raw_main", "prem_pt5_main", "drift", "expiry"]].mean()

    # 4. drift regressions
    dr = [drift_regression(P, c, s) for s in SPREADS for c in prem_cols]
    res["drift_reg"] = pd.DataFrame(dr)
    P20 = P[P.year != 2020]
    res["drift_reg_ex2020"] = pd.DataFrame([drift_regression(P20, c, "XV") for c in prem_cols])
    res["prem_month_median_XV"] = P[P.spread == "XV"].groupby("month")[
        ["market", "act_rob_main", "prem_rob_main", "prem_rob_clip", "prem_rob_cap42", "prem_pt5_main", "drift"]].median()
    res["prem_year_XV_variants"] = P[P.spread == "XV"].groupby("year")[[c for c in prem_cols]].mean()
    # benchmark: drift on -market (does the level of contango alone predict drift?)
    for s in SPREADS:
        d = P[P.spread == s].copy()
        d["neg_mkt"] = -(d.market - d.market.mean())
        lo, hi = block_boot_slope(d.rename(columns={"neg_mkt": "x"}), "x", "drift")
        res.setdefault("drift_bench", []).append({"spread": s, "slope_on_-market": np.polyfit(d.neg_mkt, d.drift, 1)[0],
                                                  "ci_lo": lo, "ci_hi": hi,
                                                  "r2": np.corrcoef(d.neg_mkt, d.drift)[0, 1] ** 2})
    res["drift_bench"] = pd.DataFrame(res["drift_bench"])
    N_VARIANTS["drift_regs"] = len(dr) + 3

    # 5. trades
    tg = pd.concat([trade_grid(P, c, "XV", first_only=f) for c in ("prem_rob_main", "prem_raw_main",
                                                                   "prem_pt5_main", "prem_rob_head",
                                                                   "prem_rob_clip", "prem_rob_cap42")
                    for f in (False, True)])
    # unconditional benchmark: always short contango (long V-X) at every origin
    d = P[P.spread == "XV"]
    pnl = -d.drift * MULT - COST
    by = pnl.groupby(d.year).mean()
    ci = year_boot_mean(by.values)
    tg = pd.concat([tg, pd.DataFrame([{"premium": "always_short_contango", "theta": np.nan,
                                       "first_only": False, "trades": len(d), "years": len(by),
                                       "mean_$": pnl.mean(), "mean_by_year_$": by.mean(), "ci_lo": ci[0],
                                       "ci_hi": ci[1], "pct_years_pos": (by > 0).mean(),
                                       "ex2020_$": pnl[d.year != 2020].mean(),
                                       "by_year": {int(k): round(v) for k, v in by.items()}}])])
    res["trades"] = tg
    N_VARIANTS["trade_grid"] = 6 * 2 * 6

    # 6. price-only
    po = price_only(a, w)
    res["price_only"] = po
    res["price_only_summary"] = summarize_price_only(po)
    res["price_only_by_entry_month"] = price_only_by_month(a, w)

    # spring/summer 2020 detail
    res["y2020"] = P[(P.year == 2020) & (P.spread == "XV")][
        ["origin", "weeks_to_go", "market", "p50", "bench", "act_raw_main", "act_rob_main",
         "act_pt5_main", "prem_rob_main", "drift"]]

    P.drop(columns=["lev"]).to_csv(CACHE / "panel.csv", index=False)
    a.to_csv(CACHE / "autumn_table.csv", index=False)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(CACHE / f"{k}.csv")
    (CACHE / "variants.json").write_text(json.dumps(N_VARIANTS))
    return a, P, res


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    a, P, res = run()
    for k, v in res.items():
        print(f"\n=== {k}")
        print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
    print(N_VARIANTS)
