"""regimes_weather: run everything (Part A news decomposition, Part B regimes, events, rule).

    python -m src.research.regimes_weather.run

All data are cut at 2021-12-31 on load (see data.py). Outputs:
  data/processed/research_regimes_weather/*.csv, results.json
  reports/agents/figures/regimes_weather_*.png
"""
import json
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CACHE, CUTOFF, ROOT, build, load_spread_daily
from .regimes import TRAIN_WINTERS, filtered_probs, fit_hmm, params_table
from .stats import ci, ols, r2, slope, winter_boot

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
FIG = ROOT / "reports/agents/figures"
NB = 2000
COST = 0.008          # 4 ticks per leg x 2 legs, $/MMBtu, round trip
MULT = 10_000         # MMBtu per contract -> $ per spread
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
RES = {"n_specs": {}}


def fmt_ci(a):
    lo, hi = ci(a)
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------- Part A
def part_a(d):
    out = {}
    base = d.dropna(subset=["surp5", "wx2"]).copy()          # primary sample (surp5 exists from Jan 2012)
    full = d.dropna(subset=["surpW", "wx2"]).copy()
    base["wx2_S"], base["s5_S"] = base.wx2 * base.S_prev, base.surp5 * base.S_prev
    specs = {
        "A1 storage (5y) only": (base, ["surp5"]),
        "A2 + next-2-wk weather": (base, ["surp5", "wx2"]),
        "A3 storage + a0,a1,a2 separately": (base, ["surp5", "a0", "a1", "a2"]),
        "A4 + report-week weather": (base, ["surp5", "aR", "a0", "a1", "a2"]),
        "A5 weather-adj storage only": (full, ["surpW"]),
        "A6 weather-adj storage + next-2-wk weather": (full, ["surpW", "wx2"]),
        "A7 A2 with x lagged spread level (convexity)": (base, ["surp5", "s5_S", "wx2", "wx2_S"]),
    }
    rows = []
    for name, (df, xs) in specs.items():
        m, dd = ols(df, "dS", xs, hac_lags=2)
        boot = winter_boot(dd, lambda s, xs=xs: r2(s, "dS", xs), NB)
        rows.append({"spec": name, "n": int(m.nobs), "R2": round(m.rsquared, 3), "R2_CI": fmt_ci(boot),
                     **{f"b_{k}": round(v, 5) for k, v in m.params.items() if k != "const"},
                     **{f"t_{k}": round(v, 2) for k, v in m.tvalues.items() if k != "const"}})
    RES["n_specs"]["A_explain"] = len(specs)
    out["explain"] = rows

    # incremental R2 of forward weather over storage, winter bootstrap
    def dr2(s, sv):
        return r2(s, "dS", [sv, "wx2"]) - r2(s, "dS", [sv])
    for sv, df in (("surp5", base), ("surpW", full)):
        b = winter_boot(df, lambda s: dr2(s, sv), NB)
        out[f"dR2_wx2_over_{sv}"] = {"point": round(dr2(df, sv), 3), "CI": fmt_ci(b)}
    b = winter_boot(base, lambda s: r2(s, "dS", ["surp5", "s5_S", "wx2", "wx2_S"]) - r2(s, "dS", ["surp5", "s5_S"]), NB)
    out["dR2_wx2_convex"] = {"point": round(r2(base, "dS", ["surp5", "s5_S", "wx2", "wx2_S"]) - r2(base, "dS", ["surp5", "s5_S"]), 3),
                             "CI": fmt_ci(b)}

    # residual (unexplained) move, primary spec A2 fitted in-sample; plus a train-only fit
    m, dd = ols(base, "dS", ["surp5", "wx2"], 2)
    base.loc[dd.index, "e"] = m.resid
    base["fit"] = base.dS - base.e
    tr = base[base.winter.isin(TRAIN_WINTERS)]
    mt, _ = ols(tr, "dS", ["surp5", "wx2"], 2)
    base["e_oos"] = base.dS - mt.predict(np.column_stack([np.ones(len(base)), base.surp5, base.wx2]))

    rev = []
    for h in range(1, 5):
        ctl = [f"a{k}" for k in range(1, h + 3)]          # all weather from t+1 to t+h+2
        for x in ("dS", "e", "fit"):
            for c in ([], ctl):
                if x == "fit" and c:
                    continue  # fit is a function of a1,a2: collinear with the controls
                mm, dd = ols(base, f"fwd_{h}", [x] + c, hac_lags=h)
                bb = winter_boot(dd, lambda s, x=x, c=c: slope(s, f"fwd_{h}", x, c), NB)
                rev.append({"h": h, "x": x, "weather_controls": bool(c), "n": int(mm.nobs),
                            "slope": round(mm.params[x], 3), "HAC_t": round(mm.tvalues[x], 2),
                            "HAC_CI": [round(v, 3) for v in mm.conf_int().loc[x]], "boot_CI": fmt_ci(bb)})
        # train-fit residual on 2017-18+ only
        oo = base[~base.winter.isin(TRAIN_WINTERS)]
        mm, dd = ols(oo, f"fwd_{h}", ["e_oos"], hac_lags=h)
        bb = winter_boot(dd, lambda s: slope(s, f"fwd_{h}", "e_oos"), NB)
        rev.append({"h": h, "x": "e_oos (2017-21, coefs from 2011-17)", "weather_controls": False, "n": int(mm.nobs),
                    "slope": round(mm.params["e_oos"], 3), "HAC_t": round(mm.tvalues["e_oos"], 2),
                    "HAC_CI": [round(v, 3) for v in mm.conf_int().loc["e_oos"]], "boot_CI": fmt_ci(bb)})
    RES["n_specs"]["A_reversal"] = len(rev)
    out["reversal"] = rev
    pd.DataFrame(rev).to_csv(CACHE / "reversal.csv", index=False)
    pd.DataFrame(rows).to_csv(CACHE / "explain.csv", index=False)
    return out, base


# ---------------------------------------------------------------- Part B
def part_b(d):
    out = {}
    d = d.sort_values("F").reset_index(drop=True)
    fits = {}
    for k in (2, 3):
        f = fit_hmm(d, k)
        fits[k] = f
        t, P = params_table(f)
        out[f"hmm{k}"] = {"loglik": round(f["ll"], 1), "BIC": round(f["bic"], 1), "AIC": round(f["aic"], 1),
                          "n_train": f["n"], "states": t.round(4).to_dict(orient="index"),
                          "transmat": np.round(P, 3).tolist()}
        m, o = f["model"], f["order"]
        (CACHE / f"hmm{k}_params_frozen.json").write_text(json.dumps({
            "states_ordered_calm_to_stressed": True, "mean": m.means_.ravel()[o].tolist(),
            "sd": np.sqrt(m.covars_.ravel()[o]).tolist(), "startprob": m.startprob_[o].tolist(),
            "transmat": m.transmat_[np.ix_(o, o)].tolist(), "fit_winters": TRAIN_WINTERS}, indent=1))
        fp, pp = filtered_probs(d, f)
        d[f"pS{k}"] = fp[:, -1]            # filtered P(most volatile state | data to t)
        d[f"pS{k}_pre"] = pp[:, -1]        # predicted P(most volatile state at t | data to t-1)
    RES["n_specs"]["B_hmm"] = 2
    d["stress"] = (d.pS2 > 0.5).astype(int)
    d["stress_pre"] = (d.pS2_pre > 0.5).astype(int)
    out["share_weeks_stressed_by_winter"] = d.groupby("winter").stress.mean().round(2).to_dict()
    # how well does the filtered regime describe next-week volatility out of sample?
    oos = d[~d.winter.isin(TRAIN_WINTERS)].copy()
    oos["absnext"] = oos.groupby("winter").dS.shift(-1).abs()
    out["oos_next_week_abs_dS_by_regime"] = oos.groupby("stress").absnext.agg(["count", "mean", "median"]).round(4).to_dict(orient="index")
    d.to_csv(CACHE / "panel_with_regimes.csv", index=False)
    return out, d, fits


def regime_reversal(d, base):
    """fwd_h on (dS or e) x regime. Regime = filtered prob at t (includes the move) or
    predicted prob from t-1 (before the move)."""
    rows = []
    b = base.merge(d[["F", "stress", "stress_pre", "pS2"]], on="F", how="left")
    for reg in ("stress", "stress_pre"):
        for x, df in (("dS", d), ("e", b)):
            for h in (1, 4):
                for sval in (0, 1):
                    for sample, sub in (("all 2010-21", df), ("2017-21 only", df[~df.winter.isin(TRAIN_WINTERS)])):
                        s = sub[sub[reg] == sval].dropna(subset=[x, f"fwd_{h}"])
                        if len(s) < 8:
                            rows.append({"regime_def": reg, "x": x, "h": h, "stressed": sval, "sample": sample, "n": len(s)})
                            continue
                        mm, dd = ols(s, f"fwd_{h}", [x], hac_lags=h)
                        bb = winter_boot(dd, lambda q: slope(q, f"fwd_{h}", x), NB)
                        rows.append({"regime_def": reg, "x": x, "h": h, "stressed": sval, "sample": sample,
                                     "n": int(mm.nobs), "slope": round(mm.params[x], 3), "HAC_t": round(mm.tvalues[x], 2),
                                     "boot_CI": fmt_ci(bb)})
    # leave-one-winter-out stability of the headline regime slopes (h=4, regime from t-1)
    lowo = {}
    for sval in (0, 1):
        g = d[d.stress_pre == sval].dropna(subset=["dS", "fwd_4"])
        v = [slope(g[g.winter != w], "fwd_4", "dS") for w in g.winter.unique()]
        lowo[f"stress_pre={sval}"] = [round(min(v), 3), round(max(v), 3)]
    RES["regime_reversal_lowo_range_h4_dS"] = lowo
    RES["n_specs"]["B_regime_reversal"] = len(rows)
    pd.DataFrame(rows).to_csv(CACHE / "regime_reversal.csv", index=False)
    return rows


# ---------------------------------------------------------------- spike rule
def exit_change(r, H):
    for h in range(H, 0, -1):
        v = r.get(f"fwd_{h}")
        if v is not None and not np.isnan(v):
            return v, h
    return np.nan, 0


def run_rule(d, X, H=4, regime=None):
    """Short the spread (short March, long April) at the settlement of any week whose
    spread change exceeds +X; exit H weeks later (or the last weekly settlement before
    March expiry / 2021-12-31). One position at a time."""
    trades, busy_until = [], pd.Timestamp("1900-01-01")
    for _, r in d.sort_values("F").iterrows():
        if r.F <= busy_until or not (r.dS > X):
            continue
        if regime is not None and r.stress_pre != regime:  # regime known BEFORE the jump
            continue
        ch, h = exit_change(r, H)
        if h == 0:
            continue
        pnl = (-ch - COST) * MULT
        trades.append({"F": r.F, "winter": r.winter, "dS": r.dS, "hold_wk": h, "change": ch, "pnl": pnl})
        busy_until = r.F + pd.Timedelta(days=7 * h)
    return pd.DataFrame(trades)


def summarize_trades(t):
    if t.empty:
        return {"trades": 0}
    b = winter_boot(t, lambda s: s.pnl.mean(), NB) if t.winter.nunique() > 1 else np.array([np.nan])
    return {"trades": len(t), "winters": int(t.winter.nunique()), "mean_pnl_$": round(t.pnl.mean(), 0),
            "total_pnl_$": round(t.pnl.sum(), 0), "hit_rate": round((t.pnl > 0).mean(), 2),
            "CI_mean_$": [round(v, 0) for v in ci(b)] if np.isfinite(b).any() else None}


def spike_rule(d):
    out = {}
    tr = d[d.winter.isin(TRAIN_WINTERS)]
    te = d[~d.winter.isin(TRAIN_WINTERS)]
    grid = [0.05, 0.10, 0.15, 0.20, 0.30]
    sel = []
    for X in grid:
        t = run_rule(tr, X)
        sel.append({"X": X, **summarize_trades(t)})
    out["train_grid_H4"] = sel
    ok = [s for s in sel if s["trades"] >= 3]
    Xs = max(ok, key=lambda s: s["mean_pnl_$"])["X"]
    out["X_selected"] = Xs
    out["test_H4"] = summarize_trades(run_rule(te, Xs))
    out["test_H1"] = summarize_trades(run_rule(te, Xs, H=1))
    out["test_H4_stressed_only"] = summarize_trades(run_rule(te, Xs, regime=1))
    out["test_H4_calm_only"] = summarize_trades(run_rule(te, Xs, regime=0))
    out["train_H4_stressed_only"] = summarize_trades(run_rule(tr, Xs, regime=1))
    out["train_H4_calm_only"] = summarize_trades(run_rule(tr, Xs, regime=0))
    out["test_grid_H4_for_reference"] = [{"X": X, **summarize_trades(run_rule(te, X))} for X in grid]
    tt = run_rule(te, Xs)
    tt.to_csv(CACHE / "rule_trades_test.csv", index=False)
    run_rule(tr, Xs).to_csv(CACHE / "rule_trades_train.csv", index=False)
    RES["n_specs"]["rule"] = len(grid) + 4
    return out, Xs


# ---------------------------------------------------------------- events
EVENTS = {"Polar vortex Jan-Feb 2014": ("2014-01-01", "2014-02-28"),
          "Nov 2018": ("2018-11-01", "2018-11-30"),
          "Feb 2021 (Uri)": ("2021-02-01", "2021-02-28")}


def events(d, Xs):
    """For each episode: every week with dS >= 0.10 $/MMBtu (and the largest week if none),
    with the next 1-4 weekly changes (stopping at March expiry / 2021-12-31) and what a short
    from that settlement earned at the selected rule's 4-week hold."""
    out = {}
    for name, (a, b) in EVENTS.items():
        w = d[(d.F >= a) & (d.F <= b)].sort_values("F")
        pick = w[w.dS >= 0.10]
        if pick.empty:
            pick = w.loc[[w.dS.idxmax()]]
        weeks = []
        for _, s in pick.iterrows():
            ch, h = exit_change(s, 4)
            weeks.append({
                "F": str(s.F.date()), "dS": round(s.dS, 3),
                "dS_pct_of_prior_spread": round(100 * s.dS / abs(s.S_prev), 0),
                "storage_surprise_bcf": None if np.isnan(s.surp5) else round(s.surp5, 1),
                "wx2_hdd_anom": None if np.isnan(s.wx2) else round(s.wx2, 1),
                "P_stressed_before": round(s.pS2_pre, 2), "P_stressed_after": round(s.pS2, 2),
                "fwd_1_4": [None if np.isnan(s[f"fwd_{k}"]) else round(s[f"fwd_{k}"], 3) for k in range(1, 5)],
                "weeks_to_expiry": round(s.weeks_to_expiry, 1),
                "short_pnl_$": None if h == 0 else round((-ch - COST) * MULT, 0), "hold_weeks": h,
                "rule_fires": bool(s.dS > Xs)})
        out[name] = {"window_max_dS": round(w.dS.max(), 3), "window_sum_dS": round(w.dS.sum(), 3), "weeks": weeks}
    return out


def fig_events_spikes(ev):
    """Daily path after the largest tradeable spike (one with >= 1 week to expiry) per episode."""
    pick = {}
    for name, e in ev.items():
        ok = [w for w in e["weeks"] if w["weeks_to_expiry"] >= 1.5] or e["weeks"]
        pick[name] = max(ok, key=lambda w: w["dS"])["F"]
    return pick


# ---------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def fig_part_a(base, rev):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    tight = base.S_prev >= base.S_prev.quantile(0.75)
    ax.scatter(base.wx2[~tight], base.dS[~tight], s=22, color=C1, alpha=0.75, edgecolor="white", lw=0.6,
               label="spread below its 75th pct")
    ax.scatter(base.wx2[tight], base.dS[tight], s=26, color=C2, alpha=0.9, edgecolor="white", lw=0.6,
               label="spread in top quartile (tight)")
    ax.axhline(0, color="#bdbcb6", lw=0.8); ax.axvline(0, color="#bdbcb6", lw=0.8)
    ax.set_xlabel("HDD anomaly, next 2 weeks (realised minus 10-yr normal)", color=INK2, fontsize=9)
    ax.set_ylabel("Weekly change in Mar-Apr spread ($/MMBtu)", color=INK2, fontsize=9)
    ax.set_title("A. Spread moves vs the weather that followed", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    ax = axs[1]
    r = pd.DataFrame(rev)
    combos = [("dS", False, "raw move", C1), ("e", False, "unexplained move", C2), ("e", True, "unexplained, + future-weather controls", C3)]
    for i, (x, c, lab, col) in enumerate(combos):
        s = r[(r.x == x) & (r.weather_controls == c)].sort_values("h")
        xs = s.h + (i - 1) * 0.22
        lo = np.array([v[0] for v in s.boot_CI]); hi = np.array([v[1] for v in s.boot_CI])
        ax.vlines(xs, lo, hi, color=col, lw=2)
        ax.scatter(xs, s.slope, color=col, s=40, zorder=3, edgecolor="white", lw=1.5, label=lab)
    ax.axhline(0, color="#bdbcb6", lw=0.8)
    ax.axhline(-1, color="#bdbcb6", lw=0.8, ls=":")
    ax.text(4.45, -1, "full reversal", fontsize=7, color=INK2, va="center")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Horizon h (weeks)", color=INK2, fontsize=9)
    ax.set_ylabel("Slope of next-h-week change on this week's move", color=INK2, fontsize=9)
    ax.set_title("B. Do moves reverse? (95% winter-bootstrap CI)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "regimes_weather_news_reversal.png", dpi=150)
    plt.close(fig)


def fig_regimes(d):
    d = d.sort_values("F").reset_index(drop=True)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(d))
    ax = axs[0]
    ax.bar(x, d.dS, color=np.where(d.stress == 1, C2, C1), width=0.8)
    ax.set_ylabel("Weekly Δ spread ($/MMBtu)", color=INK2, fontsize=9)
    ax.set_title("Weekly Mar-Apr spread changes, coloured by filtered regime (orange = stressed, P>0.5)",
                 loc="left", fontsize=10, color=INK)
    style(ax)
    ax = axs[1]
    ax.plot(x, d.pS2, color=C2, lw=1.5)
    ax.set_ylabel("P(stressed)\nfiltered", color=INK2, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    style(ax)
    starts = d.groupby("winter").apply(lambda g: g.index.min())
    for w, i in starts.items():
        for a in axs:
            a.axvline(i - 0.5, color="#d8d7d0", lw=0.6)
        axs[1].text(i + 0.5, -0.45, w, fontsize=7, color=INK2, rotation=0)
    split = starts["2017-18"] - 0.5
    for a in axs:
        a.axvline(split, color=INK, lw=1.2, ls="--")
    axs[0].text(split + 0.5, axs[0].get_ylim()[1] * 0.9, "out of sample →", fontsize=8, color=INK)
    axs[1].set_xticks([])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(FIG / "regimes_weather_regimes.png", dpi=150)
    plt.close(fig)


def fig_events(ev):
    sp = load_spread_daily()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pick = fig_events_spikes(ev)
    for (name, Fs), col in zip(pick.items(), (C1, C2, C3)):
        F = pd.Timestamp(Fs)
        Y = F.year + 1 if F.month >= 11 else F.year
        s = sp[(sp.year == Y) & (sp.date >= F - pd.Timedelta(days=10))].sort_values("date")
        s = s[s.date < s.h_last]
        d0 = s[s.date >= F].date.min()
        s0 = s.loc[s.date == d0, "spread"].iloc[0]
        s = s[s.date <= d0 + pd.Timedelta(days=28)]
        td = (s.date - d0).dt.days
        ax.plot(td, s.spread - s0, color=col, lw=2, label=f"{name} (spike week ending {F.date()})")
        ax.text(td.iloc[-1] + 0.5, (s.spread - s0).iloc[-1], name.split(" (")[0], fontsize=7, color=INK2, va="center")
    ax.axhline(0, color="#bdbcb6", lw=0.8); ax.axvline(0, color="#bdbcb6", lw=0.8)
    ax.set_xlabel("Calendar days after the spike-week settlement", color=INK2, fontsize=9)
    ax.set_ylabel("Spread change since spike settlement ($/MMBtu)", color=INK2, fontsize=9)
    ax.set_title("Mar-Apr spread after the three spikes (lines end at March expiry)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "regimes_weather_events.png", dpi=150)
    plt.close(fig)


def main():
    d = build(force=True)
    assert d.obs_date.max() <= CUTOFF
    RES["sample"] = {"weeks": len(d), "winters": d.winter.nunique(), "first": str(d.F.min().date()),
                     "last": str(d.F.max().date()), "sd_dS": round(d.dS.std(), 4)}
    a, base = part_a(d)
    RES["partA"] = a
    b, d2, fits = part_b(d)
    RES["partB"] = b
    RES["regime_reversal"] = regime_reversal(d2, base)
    rule, Xs = spike_rule(d2)
    RES["rule"] = rule
    RES["events"] = events(d2, Xs)
    fig_part_a(base, a["reversal"])
    fig_regimes(d2)
    fig_events(RES["events"])
    RES["n_specs_total"] = sum(RES["n_specs"].values())
    (CACHE / "results.json").write_text(json.dumps(RES, indent=1, default=str))
    print(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
