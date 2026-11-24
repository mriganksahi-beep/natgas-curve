"""Part 2: surface factors. Contemporaneous structure, lead-lag, squeeze event study, ex-ante tail test.

  1. Share of each daily factor change explained by the same-day futures move, and by forecast news
     (Oct-Apr rows): R^2 of d_factor ~ r_front, and incremental R^2 of + news_tot, news_w3.
  2. Lead-lag correlations corr(d_skew(t), r_front(t+j)), j = -5..+5 (whole sample, descriptive).
  3. Up-squeeze event study: events = first day of a 5-day front move >= +15% (non-overlapping).
     Skew (front, second) z-scored against the trailing 250 days, days -20..+10 around the start.
  4. Ex-ante tail test: P(next-5-day front return >= +10%) by walk-forward tercile of skew2 level,
     skew1 5-day change, call_spike; 21-day block-bootstrap CI of the hit-rate difference top - bottom.
  5. Last walk-forward PCA loadings.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.research.v5_greeks_signals import common as C
from src.research.v5_greeks_signals.features import build

FACT = ["d_level", "d_term", "d_skew1", "d_skew2", "d_curv1", "pc1", "pc2", "pc3"]


def r2(y, X):
    d = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit().rsquared, len(d)


def main():
    f = build()
    out = {}
    # 1. contemporaneous
    rows = []
    for k in FACT:
        win = f.news_tot.notna()
        a, n = r2(f.loc[win, k], f.loc[win, ["r_front"]])
        b, _ = r2(f.loc[win, k], f.loc[win, ["r_front", "news_tot", "news_w3"]])
        c, _ = r2(f.loc[win, k], f.loc[win, ["news_tot", "news_w3"]])
        bb, tt, _ = C.hac_ols(f.loc[win, k], f.loc[win, ["r_front", "news_tot"]], 5)
        rows.append({"factor": k, "R2_futures": a, "R2_futures+news": b, "R2_news_only": c,
                     "t_news|fut": tt[2], "t_fut": tt[1], "n": n})
    out["contemp"] = pd.DataFrame(rows)
    # 2. lead-lag
    ll = {}
    for k in ["d_skew1", "d_skew2", "d_level", "pc1"]:
        ll[k] = {j: f[k].corr(f.r_front.shift(-j)) for j in range(-5, 6)}
    out["leadlag"] = pd.DataFrame(ll)
    # 3. event study
    f["fwd5"] = C.fwd_sum(f.r_front, 5)
    f["skew1_z"] = (f.skew1 - f.skew1.rolling(250, min_periods=120).mean().shift(1)) / \
        f.skew1.rolling(250, min_periods=120).std().shift(1)
    f["skew2_z"] = (f.skew2 - f.skew2.rolling(250, min_periods=120).mean().shift(1)) / \
        f.skew2.rolling(250, min_periods=120).std().shift(1)
    # event start = day t such that the move t+1..t+5 >= 15%; we align day 0 = t (last pre-move settle)
    ev, last = [], None
    for d in f.index[f.fwd5 >= 15]:
        if last is None or (f.index.get_loc(d) - f.index.get_loc(last)) > 10:
            ev.append(d)
        last = d
    pos = [f.index.get_loc(d) for d in ev]
    es = {}
    for col in ("skew1_z", "skew2_z", "skew1", "skew2"):
        mat = []
        for p in pos:
            if p - 20 >= 0 and p + 10 < len(f):
                mat.append(f[col].iloc[p - 20:p + 11].values)
        es[col] = pd.DataFrame(mat, columns=range(-20, 11))
    out["events"] = pd.DataFrame({"date": ev,
                                  "fwd5": f.fwd5.reindex(ev).values,
                                  "skew2_z_d0": f.skew2_z.reindex(ev).values,
                                  "skew1_z_d0": f.skew1_z.reindex(ev).values,
                                  "skew2_z_d-5": [f.skew2_z.iloc[p - 5] for p in pos],
                                  "skew2_z_d+5": [f.skew2_z.iloc[min(p + 5, len(f) - 1)] for p in pos]})
    # named episodes
    for name, d0 in (("2014-01", "2014-01-17"), ("2018-11", "2018-11-07"), ("2021-02", "2021-02-05")):
        p = f.index.searchsorted(pd.Timestamp(d0))
        out[f"episode_{name}"] = f.iloc[p - 15:p + 12][["r_front", "skew1", "skew2", "skew1_z", "skew2_z",
                                                       "atm1", "call_spike", "pc_oi"]]
    # 4. ex-ante tail hit rates by walk-forward tercile
    f["big_up"] = (f.fwd5 >= 10).astype(float).where(f.fwd5.notna())
    trows = []
    for k in ("skew2", "skew1", "skew1_ch5", "call_spike", "spike_diff", "pc_oi", "atm1", "d_skew2"):
        q = qwf(f[k])
        for wn, w in (("choose", C.CHOOSE), ("check", C.CHECK)):
            sel = (f.index >= w[0]) & (f.index <= w[1])
            g = f.loc[sel, "big_up"]
            qq = q[sel]
            top, bot = g[qq >= 2 / 3], g[qq < 1 / 3]
            diff = (g * (qq >= 2 / 3) / max((qq >= 2 / 3).mean(), 1e-9) -
                    g * (qq < 1 / 3) / max((qq < 1 / 3).mean(), 1e-9))
            lo, hi = C.block_boot_ci(diff.dropna(), 21)
            trows.append({"signal": k, "window": wn, "p_top": top.mean(), "p_bot": bot.mean(),
                          "p_all": g.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi, "n_top": top.count()})
    out["tail"] = pd.DataFrame(trows)
    loads = pd.read_pickle(C.CACHE / "pca_loadings.pkl")
    lastk = sorted(loads)[-1]
    out["pca_last"] = pd.DataFrame(loads[lastk], index=["datm1", "datm2", "datm3", "datm4", "datm5", "datm6",
                                                         "div_m20_1", "div_m10_1", "div_p10_1", "div_p20_1"],
                                   columns=["pc1", "pc2", "pc3"])
    for k, v in out.items():
        v.to_csv(C.CACHE / f"part2_{k}.csv")
    figures(f, es, out)
    return out


def qwf(s: pd.Series, min_obs=250) -> pd.Series:
    """Walk-forward percentile of s_t among the values before the current month (monthly refit)."""
    per = s.index.to_period("M")
    out = pd.Series(np.nan, index=s.index)
    for m in per.unique():
        h = np.sort(s[per < m].dropna().values)
        if len(h) < min_obs:
            continue
        cur = s[per == m]
        out[cur.index] = np.searchsorted(h, cur.values) / len(h)
    return out.where(s.notna())


def figures(f, es, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for col, c in (("skew1_z", "#1f6fb4"), ("skew2_z", "#d1495b")):
        m = es[col].mean()
        se = es[col].std() / np.sqrt(es[col].count())
        ax[0].plot(m.index, m.values, color=c, label=col.replace("_z", " (z vs trailing yr)"))
        ax[0].fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.15)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axvspan(0, 5, color="0.9")
    ax[0].set_title(f"Call skew around {len(es['skew1_z'])} up-squeezes (5-day move >= +15%)\nday 0 = last settle before the move", fontsize=9)
    ax[0].set_xlabel("trading days")
    ax[0].legend(fontsize=8)
    ll = out["leadlag"]
    for col, c in (("d_skew1", "#1f6fb4"), ("d_skew2", "#d1495b"), ("pc1", "#555555")):
        ax[1].bar(ll.index + {"d_skew1": -0.25, "d_skew2": 0, "pc1": 0.25}[col], ll[col], width=0.25,
                  color=c, label=col)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("corr(factor change at t, front return at t+j)\nj<0: futures lead; j>0: options lead", fontsize=9)
    ax[1].set_xlabel("j (days)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG / "v5_greeks_signals_skew_events.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    o = main()
    for k, v in o.items():
        print("==", k)
        print(v.round(3).to_string())
