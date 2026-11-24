"""Key tables (with 21-day block-bootstrap CIs) and the Greek-attribution figure for the report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.v5_greeks_signals import common as C

KEY = [("resid_rel123", "front", 1, "thr1"), ("resid_rel123", "front", 1, "sign"),
       ("vega_pnl123", "front", 1, "thr1"), ("vshock123", "front", 1, "thr1"),
       ("pc1", "front", 1, "thr1"), ("d_level", "front", 1, "thr1"), ("d_skew2", "front", 5, "thr1"),
       ("skew2", "front", 5, "thr1"), ("call_spike", "front", 5, "thr1"), ("pc_oi", "front", 5, "thr1"),
       ("vshock123", "straddle", 5, "thr1"), ("vshock123", "straddle", 1, "thr1"),
       ("r_front", "front", 1, "thr1"), ("r_front", "straddle", 1, "thr1"),
       ("short_straddle", "always", 0, None), ("long_front", "always", 0, None)]


def main():
    pn = pd.read_pickle(C.CACHE / "pnls.pkl")
    rows = []
    for k in KEY:
        key = k if k[3] else k[:3]
        p = pn[key]
        rec = {"signal": k[0], "target": k[1], "h": k[2], "rule": k[3]}
        for wn, w in (("choose", C.CHOOSE), ("check", C.CHECK)):
            x = p[(p.index >= w[0]) & (p.index <= w[1])]
            lo, hi = C.block_boot_ci(x, 21, 1000)
            rec[f"{wn}_ann"] = x.mean() * 252
            rec[f"{wn}_ci"] = f"[{lo*252:,.0f}, {hi*252:,.0f}]"
            rec[f"{wn}_t"] = C.hac_mean_t(x, max(5, k[2]))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(C.CACHE / "summary_key.csv", index=False)
    sc = pd.read_csv(C.CACHE / "scores.csv")
    opt = sc[sc.family.isin(["greek", "surface", "flow"])]
    info = opt[opt.rule == "sign"]
    counts = {
        "tradable_variants": len(opt),
        "tradable_raw_p05_choose": int((opt.p_choose < 0.05).sum()),
        "tradable_holm05_choose": int((opt.holm_choose < 0.05).sum()),
        "tradable_raw_and_check": int(((opt.p_choose < 0.05) & (opt.p_check_1s < 0.05)).sum()),
        "tradable_bonferroni_t": float(norm.isf(0.05 / len(opt))),
        "info_variants": len(info),
        "info_raw_p05": int((info.p_info < 0.05).sum()),
        "info_holm05": int((info.holm_info < 0.05).sum()),
        "info_holm_survivors": info[info.holm_info < 0.05][["signal", "target", "inc_choose_t", "inc_check_t"]].round(2).values.tolist(),
        "info_raw_replicated": info[(info.p_info < 0.05) & (np.sign(info.inc_check_t) == np.sign(info.inc_choose_t))
                                    & (info.inc_check_t.abs() > 1.645)][["signal", "target", "h", "inc_choose_t", "inc_check_t"]].round(2).values.tolist(),
        "spearman_gross_t": float(opt[["gross_choose_t", "gross_check_t"]].corr("spearman").iloc[0, 1]),
        "spearman_info_t": float(info[["inc_choose_t", "inc_check_t"]].corr("spearman").iloc[0, 1]),
    }
    pd.Series(counts).to_json(C.CACHE / "summary_counts.json", indent=1)
    # figure: cumulative net P&L of the Greek-attribution rules, + year t-stats
    yrs = pd.read_csv(C.CACHE / "part1_years.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, c in ((("resid_rel123", "front", 1, "thr1"), "#1f6fb4"), (("vega_pnl123", "front", 1, "thr1"), "#d1495b"),
                 (("pc1", "front", 1, "thr1"), "#888888")):
        p = pn[k]
        ax[0].plot(p.index, p.cumsum() / 1e3, color=c, label=f"{k[0]} (|z|>1, 1-day, net)")
    ax[0].axvline(pd.Timestamp("2017-01-01"), color="k", ls=":", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("$k per 1 front contract")
    ax[0].set_title("Option-attribution signals traded in front futures\n(direction fixed on 2010-16; dotted = start of check)", fontsize=9)
    ax[0].legend(fontsize=7)
    x = yrs[["vega_pnl123", "resid_rel123", "pc1"]].dropna(how="all")
    for i, (col, c) in enumerate((("vega_pnl123", "#d1495b"), ("resid_rel123", "#1f6fb4"), ("pc1", "#888888"))):
        ax[1].bar(x.index + (i - 1) * 0.27, x[col], 0.27, color=c, label=col)
    ax[1].axhline(1.96, color="k", ls=":", lw=0.6)
    ax[1].axhline(-1.96, color="k", ls=":", lw=0.6)
    ax[1].set_title("Year-by-year HAC t of the next-day front-return slope\n(controls: today's and 5-day front return)", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "v5_greeks_signals_attribution.png", dpi=130)
    plt.close(fig)
    return t, counts


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    t, c = main()
    print(t.round(2).to_string())
    for k, v in c.items():
        print(k, v)
