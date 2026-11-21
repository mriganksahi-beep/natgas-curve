"""Frozen holdout tests H2 and H5a for regimes_weather (see reports/synthesis_v2.md).

    python -m src.holdout.regimes_weather check     # end 2021-12-31: must reproduce 2017-21 validation
    python -m src.holdout.regimes_weather holdout   # run ONCE: end 2026-11-17, scores Jan 2022 - Feb 2026

Wraps src/research/regimes_weather without editing it: the module-level CUTOFF and CACHE
of data.py and COST of run.py are overridden here.

H2  Frozen 2-state HMM (hmm2_params_frozen.json, not refitted), filtered forward, restarted each
    winter at startprob. stress_pre_t = 1 if P(stressed at t | dS up to t-1) > 0.5.
    (1) OLS slope of fwd_4 on dS_t in stress_pre weeks, HAC 4 lags; pass if slope < 0 and
        one-sided p < 0.05.
    (2) Short 1 Mar-Apr spread when dS_t > $0.05 and stress_pre_t = 1, hold 4 weeks (or exit at the
        last weekly settle before March expiry), one position at a time. Costs 4 ticks per
        contract per side: 2 contracts x 2 sides x 4 ticks x $10 = $160 per round trip.
        Pass if mean net P&L per trade > 0.
H5a R^2 of dS on surp5 (actual change minus 5-year-average change); pass if <= 0.05.
    Also reported: the R^2 added by next-2-week realised HDD anomaly (wx2), as in Part A.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import src.research.regimes_weather.data as D
import src.research.regimes_weather.run as RUN
from src.research.regimes_weather.stats import ci, ols, r2, slope, winter_boot

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data/processed/research_regimes_weather/hmm2_params_frozen.json"
VALID_PANEL = ROOT / "data/processed/research_regimes_weather/panel_with_regimes.csv"
COST_PER_SIDE = 0.016            # $/MMBtu per spread round trip = $160 per spread
NB = 2000


def stress_probs(d, params):
    mu, sd = np.array(params["mean"]), np.array(params["sd"])
    P, pi = np.array(params["transmat"]), np.array(params["startprob"])
    d = d.sort_values("F").reset_index(drop=True)
    pre, post = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for _, g in d.groupby("winter", sort=False):
        prior = pi.copy()
        for i in g.index:
            pre[i] = prior[1]
            p = prior * norm.pdf(d.at[i, "dS"], mu, sd)
            p /= p.sum()
            post[i] = p[1]
            prior = p @ P
    d["pS2_pre"], d["pS2"] = pre, post
    d["stress_pre"] = (d.pS2_pre > 0.5).astype(int)
    d["stress"] = (d.pS2 > 0.5).astype(int)
    return d


def h2(d, cost):
    out = {}
    s = d[d.stress_pre == 1].dropna(subset=["dS", "fwd_4"])
    m, dd = ols(s, "fwd_4", ["dS"], hac_lags=4)
    t = m.tvalues["dS"]
    bb = winter_boot(dd, lambda q: slope(q, "fwd_4", "dS"), NB)
    out["slope"] = {"n": int(m.nobs), "winters": int(dd.winter.nunique()), "slope": round(m.params["dS"], 3),
                    "HAC_t": round(t, 2), "p_one_sided": round(float(norm.cdf(t)), 4),
                    "HAC_CI": [round(v, 3) for v in m.conf_int().loc["dS"]],
                    "boot_CI": [round(v, 3) for v in ci(bb)]}
    RUN.COST = cost
    tr = RUN.run_rule(d, 0.05, H=4, regime=1)
    out["rule"] = RUN.summarize_trades(tr)
    out["trades"] = [{"F": str(r.F.date()), "winter": r.winter, "jump_$": round(r.dS, 3), "hold_wk": int(r.hold_wk),
                      "change_$": round(r.change, 3), "net_$": round(r.pnl, 0)} for r in tr.itertuples()]
    out["pass_slope"] = bool(out["slope"]["slope"] < 0 and out["slope"]["p_one_sided"] < 0.05)
    out["pass_pnl"] = bool(len(tr) > 0 and tr.pnl.mean() > 0)
    out["PASS"] = out["pass_slope"] and out["pass_pnl"]
    return out


def h5a(d):
    b = d.dropna(subset=["surp5", "wx2"]).copy()
    b["wx2_S"], b["s5_S"] = b.wx2 * b.S_prev, b.surp5 * b.S_prev
    out = {"n": len(b)}
    for name, xs in (("storage", ["surp5"]), ("storage+wx2", ["surp5", "wx2"]),
                     ("convex", ["surp5", "s5_S", "wx2", "wx2_S"]), ("convex_storage_only", ["surp5", "s5_S"])):
        m, dd = ols(b, "dS", xs, hac_lags=2)
        bt = winter_boot(dd, lambda q, xs=xs: r2(q, "dS", xs), NB)
        out[name] = {"R2": round(m.rsquared, 3), "boot_CI": [round(v, 3) for v in ci(bt)],
                     "coefs": {k: round(v, 5) for k, v in m.params.items()},
                     "t": {k: round(v, 2) for k, v in m.tvalues.items()}}
    bt = winter_boot(b, lambda q: r2(q, "dS", ["surp5", "wx2"]) - r2(q, "dS", ["surp5"]), NB)
    out["dR2_wx2"] = {"point": round(out["storage+wx2"]["R2"] - out["storage"]["R2"], 3),
                      "boot_CI": [round(v, 3) for v in ci(bt)]}
    out["PASS"] = bool(out["storage"]["R2"] <= 0.05)
    return out


def build(end, tag):
    D.CUTOFF = pd.Timestamp(end)
    D.CACHE = ROOT / f"data/processed/research_regimes_weather/{tag}"
    D.CACHE.mkdir(parents=True, exist_ok=True)
    return D.build(force=True)


def check():
    d = build("2021-12-31", "holdout_check")
    params = json.loads(FROZEN.read_text())
    d = stress_probs(d, params)
    v = pd.read_csv(VALID_PANEL, parse_dates=["F"])
    j = d.merge(v[["F", "pS2_pre", "dS"]], on="F", suffixes=("", "_v"))
    print(f"rows {len(d)} vs validation {len(v)}; max|pS2_pre diff| {np.abs(j.pS2_pre - j.pS2_pre_v).max():.2e}; "
          f"max|dS diff| {np.abs(j.dS - j.dS_v).max():.2e}")
    te = d[(d.F >= "2017-11-01")]
    res = {"cost_80 (validation convention)": h2(te, 0.008), "cost_160 (holdout convention)": h2(te, COST_PER_SIDE),
           "H5a_2017_21": h5a(te), "H5a_2012_21": h5a(d)}
    print(json.dumps(res, indent=1, default=str))


def holdout():
    d = build("2026-11-17", "holdout")
    params = json.loads(FROZEN.read_text())
    d = stress_probs(d, params)          # winters filtered from their own November start
    d.to_csv(D.CACHE / "panel_holdout.csv", index=False)
    h = d[(d.F >= "2022-01-07") & (d.F <= "2026-02-28")].copy()
    res = {"sample": {"weeks": len(h), "first": str(h.F.min().date()), "last": str(h.F.max().date()),
                      "winters": sorted(h.winter.unique().tolist()),
                      "stressed_pre_share": round(h.stress_pre.mean(), 2),
                      "stressed_pre_share_by_winter": h.groupby("winter").stress_pre.mean().round(2).to_dict()},
           "H2": h2(h, COST_PER_SIDE), "H5a": h5a(h)}
    RUN.COST = COST_PER_SIDE
    # secondary, not pass/fail: calm-before slope; rule at $80 for comparison with validation
    s = h[h.stress_pre == 0].dropna(subset=["fwd_4"])
    m, _ = ols(s, "fwd_4", ["dS"], hac_lags=4)
    res["secondary_calm_slope"] = {"n": int(m.nobs), "slope": round(m.params["dS"], 3), "HAC_t": round(m.tvalues["dS"], 2)}
    RUN.COST = 0.008
    res["secondary_rule_at_80"] = RUN.summarize_trades(RUN.run_rule(h, 0.05, H=4, regime=1))
    # out-of-sample volatility by regime
    h["absnext"] = h.groupby("winter").dS.shift(-1).abs()
    res["next_week_abs_dS_by_regime"] = h.groupby("stress_pre").absnext.agg(["count", "mean", "median"]).round(4).to_dict(orient="index")
    # weeks with |dS| >= 0.15 and the regime around them
    big = h[h.dS.abs() >= 0.15]
    res["big_weeks"] = [{"F": str(r.F.date()), "winter": r.winter, "dS": round(r.dS, 3), "P_pre": round(r.pS2_pre, 2),
                         "P_post": round(r.pS2, 2), "fwd_4": None if np.isnan(r.fwd_4) else round(r.fwd_4, 3),
                         "wx2": None if np.isnan(r.wx2) else round(r.wx2, 1)} for r in big.itertuples()]
    (D.CACHE / "holdout_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    fig(d, h)


def fig(d, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    w = d[d.F >= "2021-11-01"].sort_values("F").reset_index(drop=True)
    w = w[w.F <= "2026-02-28"].reset_index(drop=True)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(w))
    axs[0].bar(x, w.dS, color=np.where(w.stress_pre == 1, C2, C1), width=0.8)
    axs[0].set_ylabel("Weekly Δ spread ($/MMBtu)", color=INK2, fontsize=9)
    axs[0].set_title("Holdout: weekly Mar-Apr spread changes, coloured by regime known the week before "
                     "(orange = stressed)", loc="left", fontsize=10, color=INK)
    axs[1].plot(x, w.pS2_pre, color=C2, lw=1.5)
    axs[1].axhline(0.5, color="#bdbcb6", lw=0.8, ls=":")
    axs[1].set_ylabel("P(stressed)\nfrom t−1", color=INK2, fontsize=9)
    for _, g in w.groupby("winter"):
        i = g.index.min()
        for a in axs:
            a.axvline(i - 0.5, color="#d8d7d0", lw=0.6)
        axs[1].text(i + 0.5, -0.22, g.winter.iloc[0], fontsize=7, color=INK2, clip_on=False)
    i0 = w.index[w.F >= "2022-01-07"].min()
    for a in axs:
        a.axvline(i0 - 0.5, color=INK, lw=1.2, ls="--")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
        a.grid(axis="y", color="#e6e5e0", lw=0.6)
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_xticks([])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(ROOT / "reports/holdout/figures_regimes_weather_regimes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    {"check": check, "holdout": holdout}[sys.argv[1]]()
