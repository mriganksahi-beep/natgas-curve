"""curve_weather: read the market's 2-week weather forecast out of the futures curve.

python -m src.research.curve_weather.run   (about 2-3 minutes on 2 CPUs)
Outputs: data/processed/research_curve_weather/*, reports/agents/figures/curve_weather_*.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data, skill, extremes, storage
from .data import ROOT, CACHE, CUTOFF

FIG = ROOT / "reports/agents/figures"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "bg": "#fcfcfb"}
plt.rcParams.update({"figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6})
R = {}


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def part12(df):
    df = df.sort_values("F").reset_index(drop=True)
    lag = df["r_front"].shift(1)
    df["r_front_lag"] = np.where(df.F.diff() == pd.Timedelta(days=7), lag, np.nan)
    df, P = skill.build_predictions(df)
    fn, need = skill.calibrated(["r_front", "r_front_lag", "a0"])
    P["front+lag+pers"] = skill.walk_forward(df, fn, need)
    pd.to_pickle((df, P), CACHE / "preds.pkl")

    models = ["pers_naive", "pers", "front", "second", "strip", "marapr", "marapr_sc", "combo3",
              "combo3_sc", "gls3", "front+pers", "combo3+pers", "front+lag+pers", "S_front", "S_marapr_sc", "S_gls3"]
    E = pd.concat([df[["F", "winter", "wx2"]], P], axis=1)
    E = E[E.winter >= skill.EVAL_FIRST].dropna(subset=["wx2", "pers", "front", "combo3", "marapr_sc", "front+pers"])
    R["n_eval"] = len(E)
    R["n_winters"] = E.winter.nunique()
    R["sd_wx2"] = float(E.wx2.std())
    R["sd_S_front"] = float(E.S_front.std())

    # pooled table with bootstrap CIs (gls3 starts later: its own sample)
    rows = []
    for win_lab, wins in (("all", None), ("dev", skill.DEV), ("check", skill.CHECK)):
        sub = E if wins is None else E[E.winter.isin(wins)]
        for m in models:
            s = sub.dropna(subset=[m])
            mt = skill.metrics(s.wx2.values, s[m].values)
            rows.append(dict(window=win_lab, model=m, **mt))
    tab = pd.DataFrame(rows)
    B = skill.boot_metrics(E.dropna(subset=["gls3"]) if False else E, [m for m in models if m != "gls3"], n_boot=2000)
    cis = {}
    for m, d in B.items():
        cis[m] = {k: ci(v) for k, v in d.items()}
    tab["ci_corr"] = [cis.get(m, {}).get("corr") if w == "all" else None for w, m in zip(tab.window, tab.model)]
    tab["ci_ss"] = [cis.get(m, {}).get("ss") if w == "all" else None for w, m in zip(tab.window, tab.model)]
    tab["ci_hit"] = [cis.get(m, {}).get("hit") if w == "all" else None for w, m in zip(tab.window, tab.model)]
    tab["ci_dss_vs_pers"] = [cis.get(m, {}).get("dss") if w == "all" else None for w, m in zip(tab.window, tab.model)]
    tab["ci_dhit_vs_pers"] = [cis.get(m, {}).get("dhit") if w == "all" else None for w, m in zip(tab.window, tab.model)]
    tab.to_csv(CACHE / "skill_pooled.csv", index=False)
    R["skill_pooled"] = tab.round(3).to_dict("records")

    # incremental value: does the curve add to persistence? (bootstrap of SS difference)
    Bx = skill.boot_metrics(E, ["front+pers", "combo3+pers", "front+lag+pers"], ref="pers")
    R["front+pers_minus_pers_ss_ci"] = ci(Bx["front+pers"]["dss"])
    Bc = skill.boot_metrics(E, ["front+pers", "pers"], ref="front")
    R["front+pers_minus_front_ss_ci"] = ci(Bc["front+pers"]["dss"])
    R["pers_minus_front_ss_ci"] = ci(Bc["pers"]["dss"])
    R["pers_minus_front_hit_ci"] = ci(Bc["pers"]["dhit"])
    # partial correlation of curve move with wx2 given a0
    rr = []
    for x in ("r_front", "dS"):
        s = df[df.winter >= skill.EVAL_FIRST].dropna(subset=[x, "a0", "wx2"])
        ex = s[x] - np.polyval(np.polyfit(s.a0, s[x], 1), s.a0)
        ey = s.wx2 - np.polyval(np.polyfit(s.a0, s.wx2, 1), s.a0)
        rr.append((x, float(np.corrcoef(ex, ey)[0, 1]), float(np.corrcoef(s[x], s.a0)[0, 1])))
    R["partial_corr_given_a0"] = rr

    # calibration of the literal inversion: slope of realised on implied
    s = E.dropna(subset=["S_front"])
    R["S_front_reg_slope"] = float(np.polyfit(s.S_front, s.wx2, 1)[0])

    # by winter
    bw = []
    for w, g in E.groupby("winter"):
        r = {"winter": w, "n": len(g), "sd_wx2": g.wx2.std()}
        for m in ("pers", "front", "marapr_sc", "combo3", "front+pers"):
            mt = skill.metrics(g.wx2.values, g[m].values)
            r[f"{m}_corr"], r[f"{m}_ss"], r[f"{m}_hit"] = mt["corr"], mt["ss"], mt["hit"]
        bw.append(r)
    bw = pd.DataFrame(bw)
    bw.to_csv(CACHE / "skill_by_winter.csv", index=False)
    R["skill_by_winter"] = bw.round(3).to_dict("records")

    # sensitivities (full dev-to-2021 fit, for the report's units)
    s = df[df.winter >= "2010-11"].dropna(subset=["r_front", "dS", "wx2", "S_prev"])
    R["beta_front_pct_per_100hdd"] = float(100 * np.polyfit(s.wx2, s.r_front, 1)[0])
    R["beta_marapr_c_per_100hdd"] = float(100 * 100 * np.polyfit(s.wx2, s.dS, 1)[0])
    R["r2_front"] = float(np.corrcoef(s.wx2, s.r_front)[0, 1] ** 2)
    return df, P, E, bw


def part3(df, P):
    df = df.copy()
    out, evt = [], {}
    for name, x, scaled in (("second", "r_second", False), ("marapr", "dS", True), ("marapr_unsc", "dS", False)):
        imp, u = extremes.wf_structural_resid(df, x, scaled)
        df[f"imp_{name}"], df[f"u_{name}"] = imp, u
        df[f"ext_{name}"] = extremes.wf_top_decile(imp, df.F)
        df[f"extu_{name}"] = extremes.wf_top_decile(u, df.F)
        y = "fwd_second" if name == "second" else "fwd_dS"
        for h in (1, 2, 4):
            for z, ext, lab in ((x, f"ext_{name}", "move | extreme implied"),
                                (f"u_{name}", f"extu_{name}", "premium u | extreme u")):
                for win, wins in (("all", None), ("dev", skill.DEV), ("check", skill.CHECK)):
                    sub = df[df.winter >= skill.EVAL_FIRST] if wins is None else df[df.winter.isin(wins)]
                    if win != "all" and name == "marapr_unsc":
                        continue
                    try:
                        r = extremes.reversal(sub, f"{y}_{h}", z, ext, h, n_boot=2000 if win == "all" else 500)
                    except Exception as e:  # too few extreme weeks in a sub-window
                        r = dict(h=h, err=str(e))
                    r.update(inst=name, spec=lab, window=win)
                    out.append(r)
    rev = pd.DataFrame(out)
    rev.to_csv(CACHE / "reversal.csv", index=False)
    R["reversal"] = rev.round(3).to_dict("records")
    # thresholds actually used (last walk-forward value) and extreme counts
    d = df[df.winter >= skill.EVAL_FIRST]
    R["n_ext"] = {k: int(d[k].sum()) for k in ("ext_second", "ext_marapr", "extu_second", "extu_marapr")}
    R["imp_second_top_decile_median_abs"] = float(d.loc[d.ext_second == 1, "imp_second"].abs().median())
    R["wx2_max_abs"] = float(d.wx2.abs().max())
    # sign-adjusted mean forward move after extreme up/down implied weeks, Mar-Apr in $/spread
    sg = []
    for name, y in (("second", "fwd_second_4"), ("marapr", "fwd_dS_4")):
        for sgn in (1, -1):
            g = d[(d[f"ext_{name}"] == 1) & (np.sign(d[f"imp_{name}"]) == sgn)].dropna(subset=[y])
            scale = 10000 if name == "marapr" else 1
            sg.append(dict(inst=name, side="up" if sgn > 0 else "down", n=len(g),
                           mean_fwd4=float(g[y].mean() * scale) if len(g) else np.nan,
                           mean_wx2=float(g.wx2.mean()) if len(g) else np.nan,
                           frac_reverse=float((np.sign(g[y]) == -sgn).mean()) if len(g) else np.nan))
    R["ext_sign"] = sg
    # events
    ev = {"Jan 2014": ("2013-12-27", "2014-02-14"), "Nov 2018": ("2018-11-02", "2018-12-14"),
          "Feb 2021": ("2021-01-22", "2021-02-26")}
    cols = ["F", "r_front", "r_second", "dS", "a0", "wx2", "imp_second", "imp_marapr", "ext_second", "ext_marapr",
            "u_marapr", "extu_marapr", "fwd_second_1", "fwd_second_4", "fwd_dS_1", "fwd_dS_4"]
    Pi = P.copy()
    for k, (a, b) in ev.items():
        g = df[(df.F >= a) & (df.F <= b)][cols].copy()
        g["cal_front+pers"] = Pi.loc[g.index, "front+pers"]
        g["cal_front"] = Pi.loc[g.index, "front"]
        g["pers"] = Pi.loc[g.index, "pers"]
        for c in ("dS", "u_marapr", "fwd_dS_1", "fwd_dS_4"):
            g[c] = g[c] * 100  # cents
        g["F"] = g.F.dt.strftime("%Y-%m-%d")
        evt[k] = g.round(1)
        g.to_csv(CACHE / f"event_{k.replace(' ', '_')}.csv", index=False)
    R["events"] = {k: v.to_dict("records") for k, v in evt.items()}
    pd.to_pickle(df, CACHE / "panel_part3.pkl")
    return df, rev, evt


def part4(df, P):
    wx = data.load_weather()
    nc = storage.nowcast(df, P, wx)
    nc.to_csv(CACHE / "nowcast.csv", index=False)
    res = {}
    for col in ("bench5", "clim", "pers", "market", "market_front", "market+pers", "oracle"):
        s = nc.dropna(subset=[col, "actual"])
        e = s.actual - s[col]
        r = dict(n=len(s), mae=float(e.abs().mean()), rmse=float(np.sqrt((e ** 2).mean())))
        if col != "bench5":
            r["dm_vs_bench5"] = storage.dm(e, s.actual - s.bench5)
        if col not in ("clim", "bench5"):
            r["dm_vs_clim"] = storage.dm(e, s.actual - s.clim)
        if col in ("market", "market+pers"):
            r["dm_vs_pers"] = storage.dm(e, s.actual - s.pers)
        # winter bootstrap of MAE difference vs bench5
        rng = np.random.default_rng(1)
        gk = {k: g for k, g in s.groupby("winter")}
        keys = list(gk)
        diffs = []
        for _ in range(2000):
            sub = pd.concat([gk[k] for k in rng.choice(keys, len(keys))])
            diffs.append((sub.actual - sub[col]).abs().mean() - (sub.actual - sub.bench5).abs().mean())
        r["dmae_vs_bench5_ci"] = ci(diffs)
        res[col] = r
    R["nowcast"] = res
    R["nowcast_winters"] = sorted(nc.winter.unique().tolist())
    m1 = storage.model1_with_curve(df, P)
    m1.to_csv(CACHE / "model1_curve.csv")
    res = {}
    for lab, mask in (("panel winter weeks", m1.in_panel), ("all weeks", m1.index == m1.index)):
        s = m1[mask & (m1.index >= "2016-01-01")]
        r = {"n": int(len(s))}
        for col in ("bench5", "model1", "model1+curve_wx", "model1+curve_wx+front"):
            r[col] = float((s.actual - s[col]).abs().mean())
        for col in ("model1+curve_wx", "model1+curve_wx+front"):
            r[f"dm_{col}"] = storage.dm(s.actual - s[col], s.actual - s.model1)
        res[lab] = r
    R["model1_curve"] = res
    return nc, m1


def figures(df, P, E, bw, rev, evt):
    # Fig 1: literal inversion vs realised; skill by winter
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    s = E.dropna(subset=["S_front"])
    a = ax[0]
    a.scatter(s.S_front, s.wx2, s=14, color=C["blue"], alpha=0.7, edgecolor=C["bg"], linewidth=0.5,
              label="literal inversion (front month)")
    a.scatter(s.front, s.wx2, s=14, color=C["orange"], alpha=0.7, edgecolor=C["bg"], linewidth=0.5,
              label="calibrated (front month)")
    lim = 750
    a.plot([-lim, lim], [-lim, lim], color=C["ink2"], lw=1, ls="--")
    a.set_xlim(-lim, lim); a.set_ylim(-200, 200)
    a.set_xlabel("market-implied HDD anomaly, next 2 weeks (walk-forward)")
    a.set_ylabel("realised HDD anomaly, next 2 weeks")
    a.set_title("Literal inversion is ~3.5x too volatile", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a = ax[1]
    x = np.arange(len(bw))
    wdt = 0.26
    for j, (m, col, lab) in enumerate((("pers", C["ink2"], "persistence (a0)"), ("front", C["blue"], "curve: front month"),
                                       ("front+pers", C["orange"], "curve + persistence"))):
        a.bar(x + (j - 1) * wdt, bw[f"{m}_ss"].clip(lower=-0.6), wdt * 0.92, color=col, label=lab)
    a.axhline(0, color=C["ink"], lw=0.8)
    a.set_xticks(x); a.set_xticklabels(bw.winter, rotation=45, fontsize=8)
    a.set_ylabel("skill score vs climatology (1 - MSE/MSE_clim)")
    a.set_title("Skill by winter (walk-forward; clipped at -0.6)", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "curve_weather_skill.png", dpi=150); plt.close(fig)

    # Fig 2: events
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for a, (k, g) in zip(ax, evt.items()):
        t = pd.to_datetime(g.F)
        a.plot(t, g.wx2, color=C["ink"], lw=2, marker="o", ms=4, label="realised")
        a.plot(t, g["cal_front"], color=C["blue"], lw=2, marker="o", ms=4, label="curve (front, calibrated)")
        a.plot(t, g["cal_front+pers"], color=C["orange"], lw=2, marker="o", ms=4, label="curve + persistence")
        a.plot(t, g["pers"], color=C["aqua"], lw=2, marker="o", ms=4, label="persistence")
        a.axhline(0, color=C["ink2"], lw=0.8)
        a.set_title(k, loc="left", fontsize=10)
        a.tick_params(axis="x", rotation=45, labelsize=7)
    ax[0].set_ylabel("HDD anomaly, next 2 weeks")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "curve_weather_events.png", dpi=150); plt.close(fig)

    # Fig 3: reversal slopes
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, inst, ttl in ((ax[0], "second", "Second month (% moves)"), (ax[1], "marapr", "March-April ($ moves)")):
        g = rev[(rev.inst == inst) & (rev.window == "all")]
        for j, (spec, col) in enumerate((("move | extreme implied", C["blue"]), ("premium u | extreme u", C["orange"]))):
            gg = g[g.spec == spec]
            xs = np.array([1, 2, 4]) + (j - 0.5) * 0.25
            lo = [c[0] for c in gg.ci_ext]; hi = [c[1] for c in gg.ci_ext]
            a.errorbar(xs, gg.slope_ext, yerr=[gg.slope_ext - lo, np.array(hi) - gg.slope_ext], fmt="o",
                       color=col, ms=6, capsize=0, lw=1.5, label=f"extreme weeks: {spec.split(' |')[0]}")
            lo = [c[0] for c in gg.ci_norm]; hi = [c[1] for c in gg.ci_norm]
            a.errorbar(xs + 0.06, gg.slope_norm, yerr=[gg.slope_norm - lo, np.array(hi) - gg.slope_norm], fmt="s",
                       color=col, ms=4, mfc=C["bg"], capsize=0, lw=1, alpha=0.8, label=f"other weeks: {spec.split(' |')[0]}")
        a.axhline(0, color=C["ink2"], lw=0.8); a.axhline(-1, color=C["ink2"], lw=0.8, ls=":")
        a.set_xticks([1, 2, 4]); a.set_xlabel("horizon h (weeks)")
        a.set_title(ttl, loc="left", fontsize=10)
    ax[0].set_ylabel("slope of later move on this week's move\n(-1 = full reversal)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_ylim(-3, 3)
    fig.tight_layout(); fig.savefig(FIG / "curve_weather_reversal.png", dpi=150); plt.close(fig)


def main():
    df = data.build(force=True)
    assert df.d0.max() <= CUTOFF
    df, P, E, bw = part12(df)
    df3, rev, evt = part3(df, P)
    nc, m1 = part4(df, P)
    figures(df3, P, E, bw, rev, evt)
    with open(CACHE / "results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    print(json.dumps({k: v for k, v in R.items() if k not in ("skill_pooled", "skill_by_winter", "events", "reversal")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
