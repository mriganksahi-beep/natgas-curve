"""v4_news_channel full analysis. python -m src.research.v4_news_channel.run  (~1-2 min)

Outputs: data/processed/research_v4_news_channel/*.csv, reports/agents/figures/v4_news_channel_*.png
All data cut at 2021-12-31 (load_dev_panel). OOS = walk-forward by winter (decompose()).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import decompose as D
from .common import (prepared_panel, decompose, window, r2_oos, boot_r2, CHOOSE, CHECK, CACHE, ROOT,
                     fmt, RNG)

FIG = ROOT / "reports/agents/figures"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "bg": "#fcfcfb"}
plt.rcParams.update({"figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6})
INS = ["r_front", "r_second", "r_strip", "d_hj"]
N3 = ["news_w1", "news_w2", "news_w3"]
B = 2000
OUT = {}


def core(d, months=(11, 12, 1, 2)):
    return d[d.date.dt.month.isin(months) & d.news_part.notna()]


def both(d, fn=r2_oos):
    return {"choose": fmt(boot_r2(window(d, CHOOSE), fn, B)), "check": fmt(boot_r2(window(d, CHECK), fn, B)),
            "n_choose": len(window(d, CHOOSE)), "n_check": len(window(d, CHECK))}


# ------------------------------------------------------------------ 1. main OOS R^2 and comparisons
def main_table(p):
    rows = []
    for ins in INS:
        base = D.SPEC[ins]
        specs = {"GEFS ens (frozen spec)": base,
                 "GEFS control run": {"terms": [t + "_c00" for t in base["terms"]], "scale": None}}
        for name, sp in specs.items():
            d = decompose(p, ins, spec=sp)
            rows.append({"instrument": ins, "model": name, "months": "Nov-Feb", **both(core(d))})
            if name.startswith("GEFS ens"):
                rows.append({"instrument": ins, "model": name, "months": "Oct-Mar",
                             **both(core(d, (10, 11, 12, 1, 2, 3)))})
        # CPC block: common sample from winter 2012-13 (CPC available), OOS from 2014-15
        q = p[p.cpc610.notna() & (p.date >= "2012-11-01")]
        for name, terms in [("GEFS ens, CPC sample", base["terms"]), ("CPC only (d_cpc610, d_cpc814)",
                            ["d_cpc610", "d_cpc814"]), ("GEFS ens + CPC", base["terms"] + ["d_cpc610", "d_cpc814"])]:
            d = core(decompose(q, ins, spec={"terms": terms, "scale": None}))
            rows.append({"instrument": ins, "model": name, "months": "Nov-Feb",
                         **both(d[d.winter >= 2015])})
        # hindsight (NOT available ex ante) - upper-bound style comparisons
        for name, terms in [("HINDSIGHT perfect-at-d revision hs_w1..3", ["hs_w1", "hs_w2", "hs_w3"]),
                            ("HINDSIGHT realised next-2wk level", ["real12"]),
                            ("GEFS ens + HINDSIGHT revision", base["terms"] + ["hs_w1", "hs_w2", "hs_w3"])]:
            d = core(decompose(p, ins, spec={"terms": terms, "scale": None}))
            rows.append({"instrument": ins, "model": name, "months": "Nov-Feb", **both(d)})
        # timing diagnostic: tomorrow's 00Z news (look-ahead by construction, diagnostic only)
        q = p.copy()
        for t in N3:
            q[t + "_lead"] = q[t].shift(-1).where(pd.to_datetime(q.prev_date.shift(-1)) == q.date)
        d = core(decompose(q, ins, spec={"terms": N3 + [t + "_lead" for t in N3], "scale": None}))
        rows.append({"instrument": ins, "model": "DIAGNOSTIC + next day's 00Z news (look-ahead)",
                     "months": "Nov-Feb", **both(d)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "r2_table.csv", index=False)
    return t


# ------------------------------------------------------------------ coefficients and horizon weights
def coef_table(p):
    rows = []
    for ins in INS:
        for terms in (N3, ["news_w0"] + N3):
            sp = {"terms": terms, "scale": None}
            ok = D.eligible(p, ins, sp) & p.month.isin([11, 12, 1, 2])
            d = p[ok]
            m = sm.OLS(d[ins], sm.add_constant(d[terms])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            sd = d[terms].std()
            for t in terms:
                rows.append({"instrument": ins, "spec": "+".join(x[-2:] for x in terms), "term": t,
                             "beta": m.params[t], "t_HAC5": m.tvalues[t], "beta_x_sd": m.params[t] * sd[t],
                             "R2_insample": m.rsquared, "n": len(d)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "coefs_insample.csv", index=False)
    return t


def vintage_paths(p):
    rows = []
    for ins in INS:
        sp = D.SPEC[ins]
        ok = D.eligible(p, ins, sp) & p.month.isin([11, 12, 1, 2])
        d = p[ok]
        for w in sorted(d.W.unique())[1:]:
            tr = d[d.W < w]
            if tr.W.nunique() < 2:
                continue
            b = D.ols(D.design(tr, ins, sp).to_numpy(), tr[ins].to_numpy())
            rows.append({"instrument": ins, "applied_to_winter": w, **dict(zip(sp["terms"], b[1:]))})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "vintage_coefs.csv", index=False)
    return t


# ------------------------------------------------------------------ 2. state dependence (descriptive)
def state_table(p):
    """Full-dev, in-sample, descriptive: beta of the move on the frozen news index within terciles."""
    rows = []
    for ins in INS:
        sp = D.SPEC[ins]
        d = decompose(p, ins, mode="frozen")
        d = d[d.in_core].merge(p[["date", "hj_prev", "iv_prev", "front_days_left", "px_prev", "month"]], on="date")
        d["idx"] = d.news_part
        for st in ["hj_prev", "iv_prev", "px_prev", "front_days_left", "month"]:
            g = d.dropna(subset=[st])
            grp = g.month.map({11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb"}) if st == "month" else \
                pd.qcut(g[st], 3, labels=["low", "mid", "high"])
            for k, h in g.groupby(grp, observed=True):
                m = sm.OLS(h.move, sm.add_constant(h.idx)).fit()
                bs = []
                ws = h.winter.unique()
                for _ in range(500):
                    hh = pd.concat([h[h.winter == w] for w in RNG.choice(ws, len(ws))])
                    bs.append(np.polyfit(hh.idx, hh.move, 1)[0] if hh.idx.std() > 0 else np.nan)
                rows.append({"instrument": ins, "state": st, "group": k,
                             "state_median": float(h[st].median()) if st != "month" else np.nan,
                             "slope_on_news_index": m.params["idx"],
                             "ci": f"[{np.nanpercentile(bs, 2.5):.2f}, {np.nanpercentile(bs, 97.5):.2f}]",
                             "R2": m.rsquared, "n": len(h)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "state_dependence.csv", index=False)
    return t


# ------------------------------------------------------------------ 3. weekly
def weekly(p):
    rows, wk_all = [], {}
    for ins in INS:
        sp = D.SPEC[ins]
        dd = decompose(p, ins).merge(p[["date"] + sorted(set(N3 + ["news_w0", "news_tot", "real12"]))], on="date")
        dd = dd[dd.news_part.notna()]
        dd["wk"] = dd.date.dt.to_period("W-FRI")
        agg = {"move": "sum", "news_part": "sum", "intercept": "sum", "train_mean": "sum", "winter": "last",
               "date": "last", **{t: "sum" for t in set(N3 + ["news_w0", "news_tot"])}, "real12": "last"}
        g = dd.groupby("wk").agg(agg)
        g["ndays"] = dd.groupby("wk").size()
        g = g[(g.ndays >= 4) & g.date.dt.month.isin([11, 12, 1, 2])]
        g["premium_part"] = g.move - g.news_part
        wk_all[ins] = g
        rows.append({"instrument": ins, "model": "sum of daily walk-forward news_part", **both(g)})
        m = sm.OLS(g.move, sm.add_constant(g.news_part)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        rows[-1]["weekly_slope_on_news_part"] = f"{m.params.news_part:.2f} (t {m.tvalues.news_part:.1f})"
        # weekly regressions refitted walk-forward by winter on weekly sums
        for name, terms in [("weekly refit, GEFS news (frozen terms)", sp["terms"]),
                            ("weekly refit, HINDSIGHT realised next-2wk level (v2-style)", ["real12"])]:
            g2 = g.copy()
            pr = np.full(len(g2), np.nan); ic = pr.copy(); mu = pr.copy()
            X = g2[terms].to_numpy(); y = g2.move.to_numpy(); W = g2.winter.to_numpy()
            for w in np.unique(W):
                tr = W < w
                if len(np.unique(W[tr])) < 2:
                    continue
                b = D.ols(X[tr], y[tr]); m = W == w
                pr[m] = X[m] @ b[1:]; ic[m] = b[0]; mu[m] = y[tr].mean()
            g2["news_part"], g2["intercept"], g2["train_mean"] = pr, ic, mu
            rows.append({"instrument": ins, "model": name, **both(g2.dropna(subset=["news_part"]))})
            m = sm.OLS(g.move, sm.add_constant(g[terms])).fit()
            rows[-1]["R2_insample_all"] = round(m.rsquared, 3)
        if ins == "d_hj":       # v2 convexity, in-sample only, for comparison
            g3 = g.copy();
            hj_end = p.set_index("date").hj.reindex(g3.date).to_numpy()
            g3["S_lag"] = pd.Series(hj_end, index=g3.index).shift(1)
            g3 = g3.dropna(subset=["S_lag"])
            for nm, xcol in [("real12", "real12"), ("news_tot", "news_tot")]:
                g3["x_S"] = g3[xcol] * g3.S_lag
                m = sm.OLS(g3.move, sm.add_constant(g3[[xcol, "S_lag", "x_S"]])).fit()
                rows.append({"instrument": ins, "model": f"IN-SAMPLE weekly {nm} x lagged spread (v2 convexity)",
                             "R2_insample_all": round(m.rsquared, 3)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "weekly.csv", index=False)
    return t, wk_all


def weekly_frozen_spec(p):
    """OOS R^2 of decompose_weekly (the frozen weekly companion) and its d_hj alternatives."""
    rows = []
    for ins in INS:
        cands = {"WEEKLY_SPEC": D.WEEKLY_SPEC[ins]}
        if ins == "d_hj":
            cands["news_tot linear"] = {"terms": ["news_tot"], "scale": None}
        for nm, sp in cands.items():
            d = D.decompose_weekly(p, ins, spec=sp)
            d = d[d.in_core & d.news_part.notna()]
            rows.append({"instrument": ins, "model": nm, **both(d),
                         "by_winter": {int(w): round(r2_oos(x), 2) for w, x in d.groupby("winter")}})
    # v2-style convexity (news, S, news*S), walk-forward weekly, d_hj
    g = D.weekly_table(p, "d_hj", {"terms": ["news_tot"], "scale": None})
    g["xS"] = g.news_tot * g.hj_dp
    X = g[["news_tot", "hj_dp", "xS"]].to_numpy(); y = g.d_hj.to_numpy(); W = D.winter_of(g.date).to_numpy()
    fm = g.date.dt.month.isin([11, 12, 1, 2]).to_numpy()
    pr = np.full(len(g), np.nan); mu = pr.copy()
    for w in np.unique(W):
        tr = fm & (W < w)
        if len(np.unique(W[tr])) < 2:
            continue
        b = D.ols(X[tr], y[tr]); m = W == w
        pr[m] = X[m] @ b[1:] + b[0]; mu[m] = y[tr].mean()
    h = pd.DataFrame({"winter": W, "move": y, "news_part": pr, "intercept": 0.0, "train_mean": mu})[fm].dropna()
    rows.append({"instrument": "d_hj", "model": "v2 convexity news, S, news*S", **both(h)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "weekly_frozen_spec.csv", index=False)
    wk = D.decompose_weekly(p, "d_hj")
    wf = D.decompose_weekly(p, "r_front")
    sel = []
    for d0 in pd.to_datetime(["2014-01-24", "2014-02-21", "2018-11-16", "2018-11-23", "2021-02-12", "2021-02-19"]):
        r = wk[wk.week_end == d0]; f = wf[wf.week_end == d0]
        if len(r):
            sel.append({"week_end": d0.date(), "d_hj_move": r.move.iloc[0], "news_part": r.news_part.iloc[0],
                        "premium_part": r.premium_part.iloc[0],
                        "r_front_move": f.move.iloc[0] if len(f) else np.nan,
                        "r_front_news": f.news_part.iloc[0] if len(f) else np.nan})
    big = wk[wk.news_part.notna()].reindex(wk.move.abs().sort_values(ascending=False).index).dropna(subset=["news_part"]).head(10)
    e = pd.DataFrame(sel)
    e.to_csv(CACHE / "episodes_weekly.csv", index=False)
    big[["week_end", "move", "news_part", "premium_part", "vintage"]].to_csv(CACHE / "episodes_weekly_top10.csv", index=False)
    return t, e, big


# ------------------------------------------------------------------ 4. robustness
def robustness(p):
    rows = []
    for ins in INS:
        d = core(decompose(p, ins)).merge(p[["date", "gefs_system", "is_mon", "is_storage",
                                             "storage_surprise_5y"]], on="date")
        oos = d[d.winter >= CHOOSE[0]]
        for lab, m in [("reforecast_v12", oos.gefs_system == "reforecast_v12"),
                       ("operational (v11+v12)", oos.gefs_system != "reforecast_v12"),
                       ("Mondays", oos.is_mon == 1), ("Tue-Fri", oos.is_mon == 0),
                       ("storage release days (2015+)", (oos.is_storage == 1)),
                       ("non-release days (2015+)", (oos.is_storage == 0) & (oos.date >= "2015-01-01"))]:
            s = oos[m]
            rows.append({"instrument": ins, "subset": lab, "n": len(s), "R2_oos": fmt(boot_r2(s, B=1000))})
        # in-sample interaction tests (full dev, HAC)
        sp = D.SPEC[ins]
        q = p[D.eligible(p, ins, sp) & p.month.isin([11, 12, 1, 2])].copy()
        idx = D.design(q, ins, sp).to_numpy() @ np.asarray(json.loads(D.FROZEN.read_text())[ins]["slopes"])
        q["idx"] = idx
        q["op"] = (q.gefs_system != "reforecast_v12").astype(float)
        for lab, cols in [("idx x operational", ["idx", "op", "idx_op"]), ("idx x Monday", ["idx", "is_mon", "idx_mon"])]:
            q["idx_op"] = q.idx * q.op; q["idx_mon"] = q.idx * q.is_mon
            m = sm.OLS(q[ins], sm.add_constant(q[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            rows.append({"instrument": ins, "subset": f"IN-SAMPLE {lab}: interaction coef (t)",
                         "n": len(q), "R2_oos": f"{m.params[cols[2]]:.2f} (t {m.tvalues[cols[2]]:.1f})"})
        s = q[q.storage_surprise_5y.notna()]
        m0 = sm.OLS(s[ins], sm.add_constant(s[["idx"]])).fit()
        m1 = sm.OLS(s[ins], sm.add_constant(s[["idx", "storage_surprise_5y"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        rows.append({"instrument": ins, "subset": "IN-SAMPLE release days: + storage surprise",
                     "n": len(s), "R2_oos": f"R2 {m0.rsquared:.3f}->{m1.rsquared:.3f}; coef/Bcf {m1.params['storage_surprise_5y']:.4f} "
                                             f"(t {m1.tvalues['storage_surprise_5y']:.1f}); idx slope {m0.params['idx']:.2f}->{m1.params['idx']:.2f}"})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "robustness.csv", index=False)
    return t


# ------------------------------------------------------------------ 5. dependence of premium on next-day news
def premium_diag(p):
    rows = []
    for ins in INS:
        d = core(decompose(p, ins)).sort_values("date").reset_index(drop=True)
        nextday = dict(zip(pd.to_datetime(p.prev_date), p.date))
        consec = d.date.map(nextday) == d.date.shift(-1)
        nxt = d.news_part.shift(-1).where(consec)
        prv = d.premium_part.shift(1)
        rows.append({"instrument": ins, "corr(premium_d, news_part_d+1)": d.premium_part.corr(nxt),
                     "corr(premium_d, premium_d+1)": d.premium_part.corr(d.premium_part.shift(-1).where(consec)),
                     "corr(news_d, news_d+1)": d.news_part.corr(nxt),
                     "var share news_part": d.news_part.var() / d.move.var(),
                     "corr(news,premium)": d.news_part.corr(d.premium_part)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "premium_diag.csv", index=False)
    return t


def leadlag(p):
    """IN-SAMPLE diagnostic: slope of the move on day d+k on the walk-forward news_part of day d."""
    rows = []
    for ins in INS:
        a = decompose(p, ins).reset_index(drop=True)          # eligible rows, consecutive trading days
        cons = (a.date.map(dict(zip(pd.to_datetime(p.prev_date), p.date))) == a.date.shift(-1))
        for k in [-3, -2, -1, 0, 1, 2, 3, 5]:
            y = a.move.shift(-k)
            ok = pd.Series(True, index=a.index)
            for j in range(min(k, 0), max(k, 0)):                # chain must be unbroken
                ok &= cons.shift(-j).fillna(False).astype(bool)
            d = pd.DataFrame({"y": y, "x": a.news_part})[a.in_core & ok].dropna()
            m = sm.OLS(d.y, sm.add_constant(d.x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            rows.append({"instrument": ins, "k": k, "slope": m.params.x, "t": m.tvalues.x, "n": len(d)})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "leadlag.csv", index=False)
    return t


# ------------------------------------------------------------------ 6. episodes
def episodes(p):
    dh = decompose(p, "d_hj")
    df_ = decompose(p, "r_front")
    fz = decompose(p, "d_hj", mode="frozen")
    e = dh.merge(df_[["date", "move", "news_part"]].rename(columns={"move": "r_front", "news_part": "r_front_news"}),
                 on="date", how="left").merge(fz[["date", "news_part"]].rename(columns={"news_part": "news_frozen"}),
                                              on="date").merge(p[["date", "news_tot", "news_w1", "news_w2", "news_w3"]], on="date")
    feb14 = e[(e.date >= "2014-02-01") & (e.date <= "2014-02-28")].nlargest(1, "move").date.iloc[0]
    named = pd.to_datetime(["2014-01-24", feb14, "2018-11-14", "2018-11-15", "2021-02-12"])
    top = e.reindex(e.move.abs().sort_values(ascending=False).index).head(10).date
    sel = pd.Index(named).union(pd.Index(top))
    t = e[e.date.isin(sel)].copy()
    t["tag"] = np.where(t.date.isin(named), "named", "") + np.where(t.date.isin(top), " top10", "")
    cols = ["date", "tag", "move", "news_part", "premium_part", "news_frozen", "news_tot", "news_w1",
            "news_w2", "news_w3", "r_front", "r_front_news", "vintage"]
    t = t[cols].sort_values("date")
    # 5-day windows starting at each named date
    win = []
    for d0 in named:
        s = e[(e.date >= d0)].head(5)
        win.append({"start": d0.date(), "d_hj_5d": s.move.sum(), "news_5d": s.news_part.sum(),
                    "premium_5d": s.premium_part.sum(), "r_front_5d": s.r_front.sum(),
                    "r_front_news_5d": s.r_front_news.sum()})
    t.to_csv(CACHE / "episodes.csv", index=False)
    pd.DataFrame(win).to_csv(CACHE / "episodes_5d.csv", index=False)
    return t, pd.DataFrame(win), e


# ------------------------------------------------------------------ figures
def fig_r2(t):
    keep = [("GEFS ens (frozen spec)", "Nov-Feb", "GEFS ensemble"), ("GEFS control run", "Nov-Feb", "GEFS control"),
            ("HINDSIGHT perfect-at-d revision hs_w1..3", "Nov-Feb", "Hindsight revision"),
            ("DIAGNOSTIC + next day's 00Z news (look-ahead)", "Nov-Feb", "+ next-day news (diag.)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    cols = [C["blue"], C["aqua"], C["yellow"], C["orange"]]
    for ax, win in zip(axes, ["choose", "check"]):
        for j, (mname, mo, lab) in enumerate(keep):
            for i, ins in enumerate(INS):
                r = t[(t.instrument == ins) & (t.model == mname) & (t.months == mo)].iloc[0][win]
                v, lo, hi = [float(x) for x in r.replace("[", "").replace("]", "").replace(",", "").split()]
                x = i + (j - 1.5) * 0.19
                ax.bar(x, v, 0.17, color=cols[j], label=lab if i == 0 else None)
                ax.plot([x, x], [lo, hi], color=C["ink2"], lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(["front %", "second %", "strip %", "Mar-Apr $"])
        ax.set_title({"choose": "Choice window (OOS winters 2012-13..2016-17)",
                      "check": "Check window (2017-18..Dec 2021)"}[win], fontsize=9, loc="left")
        ax.axhline(0, color=C["ink2"], lw=0.8)
    axes[0].set_ylabel("walk-forward OOS R², daily, Nov-Feb")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "v4_news_channel_r2.png", dpi=150); plt.close(fig)


def fig_episodes(e):
    eps = [("2013-12-01", "2014-02-28", "Winter 2013-14 (polar vortex)"),
           ("2018-11-01", "2018-12-31", "Nov-Dec 2018 (squeeze)"),
           ("2021-01-15", "2021-02-24", "Feb 2021 (Uri)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (a, b, ttl) in zip(axes, eps):
        s = e[(e.date >= a) & (e.date <= b)].sort_values("date")
        ax.plot(s.date, s.move.cumsum(), color=C["ink"], lw=2, label="March-April change")
        ax.plot(s.date, s.news_part.cumsum(), color=C["blue"], lw=2, label="news part")
        ax.plot(s.date, s.premium_part.cumsum(), color=C["orange"], lw=2, label="premium part")
        ax.set_title(ttl, fontsize=9, loc="left"); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.axhline(0, color=C["ink2"], lw=0.8)
    axes[0].set_ylabel("cumulative change, $/MMBtu")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "v4_news_channel_episodes.png", dpi=150); plt.close(fig)


def fig_weekly(t, wk):
    g = wk["r_front"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ins, lab in [(axes[0], "r_front", "front month, weekly %"), (axes[1], "d_hj", "March-April, weekly $")]:
        g = wk[ins]
        ax.scatter(g.news_part, g.move, s=10, color=C["blue"], alpha=0.6, edgecolor=C["bg"], linewidth=0.4)
        lim = np.nanmax(np.abs(g.news_part)) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=C["ink2"], lw=1, ls="--", label="slope 1 (daily-fit news)")
        b = np.polyfit(g.news_part, g.move, 1)
        ax.plot([-lim, lim], np.polyval(b, [-lim, lim]), color=C["orange"], lw=2, label=f"weekly OLS slope {b[0]:.1f}")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlabel("sum of daily walk-forward news_part"); ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs summed daily news part, Nov-Feb", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "v4_news_channel_weekly.png", dpi=150); plt.close(fig)


def run():
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    p = prepared_panel()
    t = main_table(p); print(t.to_string())
    c = coef_table(p); print(c.round(4).to_string())
    v = vintage_paths(p); print(v.round(4).to_string())
    s = state_table(p); print(s.round(3).to_string())
    w, wk = weekly(p); print(w.to_string())
    wf, we, wb = weekly_frozen_spec(p); print(wf.to_string()); print(we.round(3).to_string())
    print(wb[["week_end", "move", "news_part", "premium_part"]].round(3).to_string())
    r = robustness(p); print(r.to_string())
    pdg = premium_diag(p); print(pdg.round(3).to_string())
    ll = leadlag(p); print(ll.round(3).to_string())
    ep, ep5, e = episodes(p); print(ep.round(4).to_string()); print(ep5.round(3).to_string())
    fig_r2(t); fig_episodes(e); fig_weekly(w, wk)


if __name__ == "__main__":
    run()
