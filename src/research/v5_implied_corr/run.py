"""v5_implied_corr: run everything.  python -m src.research.v5_implied_corr.run   (~2 min, 1 CPU)"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import crp, indicator, trade
from .common import CACHE, FIG, MMBTU, boot_blockmean, gas_year

OUT = {}
EPIS = {"2013-14 (Dec-Feb)": ("2013-12-01", "2014-02-28"), "Nov 2018": ("2018-10-25", "2018-11-30"),
        "Feb 2021": ("2021-02-01", "2021-02-26")}


def save(df, name):
    df.to_csv(CACHE / f"{name}.csv", index=False)
    return df


def trades_tables():
    rows, per = [], []
    base = {}
    for hs in (0.03, 0.04, 0.05, 0.08):
        T, D = trade.build_trades(cso_hs=hs) if hs == 0.04 else trade.build_trades(cso_hs=hs)
        base[hs] = (T, D)
    T, D = base[0.04]
    T = T.copy()
    T["blk"] = gas_year(T.entry)
    T["window"] = np.where(T.entry.dt.year <= 2016, "choose", "check")
    for hs, (Th, _) in base.items():
        Th = Th.copy()
        Th["blk"] = gas_year(Th.entry)
        Th["window"] = np.where(Th.entry.dt.year <= 2016, "choose", "check")
        for hm in ("daily_leg", "daily_spread", "band"):
            ocS, ocN, ocF = Th.ocS, Th.ocN, Th.ocF
            A = Th[f"gS_{hm}"] - Th[f"hS_{hm}"] - ocS
            N_net = Th[f"gN_{hm}"] - Th[f"hN_{hm}"] - ocN           # long 1 near straddle, net
            F_net = Th[f"gF_{hm}"] - Th[f"hF_{hm}"] - ocF
            N_short = -Th[f"gN_{hm}"] - Th[f"hN_{hm}"] - ocN
            F_short = -Th[f"gF_{hm}"] - Th[f"hF_{hm}"] - ocF
            strat = {
                "A_short_CSO": A,
                "B_short_near_prem_matched": (Th.PS / Th.PN) * N_short,
                "B2_short_legs_gamma_matched": Th.wG_n * N_short + Th.wG_f * F_short,
                "C_corr_gamma_matched": A + Th.wG_n * N_net + Th.wG_f * F_net,
                "D_corr_rho_neutral_vega": A + np.where(Th.wV_n >= 0, Th.wV_n * N_net, -Th.wV_n * N_short)
                + np.where(Th.wV_f >= 0, Th.wV_f * F_net, -Th.wV_f * F_short),
            }
            for sname, v in strat.items():
                X = Th.assign(v=v * MMBTU, pct=v / Th.PS)
                for E in (*trade.ENTRY_T, "all"):
                    XE = X if E == "all" else X[X.E == E]
                    for w in ("choose", "check", "all"):
                        q = XE if w == "all" else XE[XE.window == w]
                        if q.empty:
                            continue
                        bm = q.groupby("blk").v.mean()
                        mu, lo, hi = boot_blockmean(bm.values)
                        rows.append({"cso_hs": hs, "hedge": hm, "strategy": sname, "E": E, "window": w, "n": len(q),
                                     "winters": len(bm), "mean_usd": q.v.mean(), "winter_mean_usd": mu, "lo": lo, "hi": hi,
                                     "median_usd": q.v.median(), "win_rate": (q.v > 0).mean(), "worst_usd": q.v.min(),
                                     "pct_premium_mean": q.pct.mean(), "pct_premium_median": q.pct.median(),
                                     "t_like": q.v.mean() / (q.v.std(ddof=1) / np.sqrt(len(q))) if len(q) > 2 else np.nan})
                if hs == 0.04 and hm == "band":
                    per.append(X[["cid", "cm", "E", "entry", "window", "blk"]].assign(strategy=sname, usd=X.v))
    tab = save(pd.DataFrame(rows), "step3_trades_grid")
    per = save(pd.concat(per), "step3_trades_per_trade")
    # gross capture as % of premium (no costs): spread vs outright vol
    gross = T.assign(cso=T.gS_band / T.PS, near=-T.gN_band / T.PN, far=-T.gF_band / T.PF,
                     rv_ratio=T.rv_S / T.nv0)
    g = []
    for w, q in (("choose", gross[gross.window == "choose"]), ("check", gross[gross.window == "check"]), ("all", gross)):
        r = {"window": w, "n": len(q)}
        for c in ("cso", "near", "far", "rv_ratio"):
            mu, lo, hi = boot_blockmean(q.groupby("blk")[c].mean().values)
            r.update({c: mu, c + "_lo": lo, c + "_hi": hi, c + "_med": q[c].median()})
        r["hedge_cost_pct_prem"] = (T.loc[q.index, "hS_band"] / T.loc[q.index, "PS"]).mean()
        r["opt_cost_pct_prem"] = (T.loc[q.index, "ocS"] / T.loc[q.index, "PS"]).mean()
        g.append(r)
    save(pd.DataFrame(g), "step3_gross_capture")
    return tab, per, T, D


def portfolio(T, D, E_sel):
    T, D = trade.combine(T, D)
    sel = T[T.E == E_sel]
    out, curves = [], {}
    for sname, col, ccol in (("A_short_CSO", "A_short_cso", None), ("C_corr_gamma_matched", "C_corr_G", None),
                             ("D_corr_rho_neutral_vega", "D_corr_V", None)):
        d = D.merge(sel[["cid", "E"]], on=["cid", "E"])
        d["A_short_cso"] = d.pS_band
        d["C_corr_G"] = d.pS_band + d.wG_n * d.pN_band + d.wG_f * d.pF_band
        d["D_corr_V"] = d.pS_band + d.wV_n * d.pN_band + d.wV_f * d.pF_band
        daily = d.groupby("date")[col].sum() * MMBTU
        # trading costs booked on the entry day (band hedge, 4% CSO)
        if sname == "A_short_CSO":
            cost = sel.hS_band + sel.ocS
        elif sname == "C_corr_gamma_matched":
            cost = sel.hS_band + sel.ocS + sel.wG_n * (sel.hN_band + sel.ocN) + sel.wG_f * (sel.hF_band + sel.ocF)
        else:
            cost = sel.hS_band + sel.ocS + sel.wV_n.abs() * (sel.hN_band + sel.ocN) + sel.wV_f.abs() * (sel.hF_band + sel.ocF)
        cser = pd.Series((cost * MMBTU).values, index=sel.entry.values).groupby(level=0).sum()
        daily = daily.add(-cser.reindex(daily.index.union(cser.index)).fillna(0), fill_value=0).sort_index()
        cum = daily.cumsum()
        curves[sname] = cum
        r = {"strategy": sname, "E": E_sel}
        ent = sel.set_index(["cid"]).entry
        win = pd.Series(np.where(sel.entry.dt.year <= 2016, "choose", "check"), index=sel.cid.values)
        dwin = d.assign(w=d.cid.map(win)).groupby(["w", "date"])[col].sum() * MMBTU
        for w in ("choose", "check"):
            cw = sel[win.reindex(sel.cid).values == w]
            cc = pd.Series((cost.loc[cw.index] * MMBTU).values, index=cw.entry.values).groupby(level=0).sum()
            dd_ = dwin.loc[w] if w in dwin.index.get_level_values(0) else pd.Series(dtype=float)
            dd_ = dd_.add(-cc.reindex(dd_.index.union(cc.index)).fillna(0), fill_value=0).sort_index()
            c = dd_.cumsum()
            r[f"{w}_total"] = c.iloc[-1] if len(c) else np.nan
            r[f"{w}_maxdd"] = (c - c.cummax()).min() if len(c) else np.nan
        for ep, (a, b) in EPIS.items():
            r[f"ep {ep}"] = daily[a:b].sum()
        out.append(r)
    return save(pd.DataFrame(out), "step3_portfolio"), curves


def episodes_by_trade(per):
    rows = []
    p = per[(per.strategy.isin(["A_short_CSO", "D_corr_rho_neutral_vega", "C_corr_gamma_matched"]))]
    for ep, (a, b) in EPIS.items():
        q = p[(p.entry <= b) & (p.entry >= pd.Timestamp(a) - pd.Timedelta(days=160))]
        for r in q.itertuples():
            rows.append({"episode": ep, "cid": r.cid, "E": r.E, "entry": r.entry.date(), "strategy": r.strategy,
                         "usd": r.usd})
    return save(pd.DataFrame(rows), "step3_episode_trades")


def figures(s, curves, m):
    # Fig 1: implied vs realised correlation (1 - rho, log scale) by tdays bucket and group
    o, b = crp.by_cuts(s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bb = o["bucket"]
    x = np.arange(len(bb))
    ax[0].bar(x - 0.2, bb.impl_decoupling_1mr, 0.4, label="implied 1-ρ (R1)", color="#4C78A8")
    ax[0].bar(x + 0.2, bb.real_decoupling_1mr, 0.4, label="realised 1-ρ (life)", color="#F58518")
    ax[0].set_xticks(x, [str(v) for v in bb.bucket], fontsize=8)
    ax[0].set_xlabel("trading days to CSO expiry")
    ax[0].set_title("All adjacent pairs: priced vs realised decoupling (medians)", fontsize=9)
    ax[0].legend(fontsize=8)
    hj = b[b.cm == 3].groupby("blk").agg(ri=("ri", "median"), rr=("rho_r_life", "median"))
    ax[1].plot(hj.index, (1 - hj.ri).clip(lower=1e-3), "o-", label="implied 1-ρ (floored at 0.001)", color="#4C78A8")
    ax[1].plot(hj.index, (1 - hj.rr).clip(lower=1e-3), "s-", label="realised 1-ρ", color="#F58518")
    ax[1].set_yscale("log")
    ax[1].set_title("March−April (cm=3): median by winter (T ≤ 126d)", fontsize=9)
    ax[1].set_xlabel("winter (March year)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v5_implied_corr_rho.png", dpi=120)
    plt.close(fig)
    # Fig 2: cumulative portfolio P&L
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, c in curves.items():
        if k.startswith("C_"):
            continue      # gamma-matched version is off-scale (-$560k on the check window): see table
        ax.plot(c.index, c.values / 1000, label=k)
    for ep, (a, bnd) in EPIS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(bnd), color="grey", alpha=0.2)
    ax.axvline(pd.Timestamp("2017-01-01"), color="k", ls="--", lw=0.8)
    ax.set_ylabel("$k, 1 CSO straddle per trade")
    ax.set_title("Cumulative net P&L, entry 42 trading days before expiry, all pairs (choose | check)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v5_implied_corr_pnl.png", dpi=120)
    plt.close(fig)
    # Fig 3: H-J rho_impl around Nov 2018 & 2013-14
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a_, (y, lo, hi) in zip(ax, ((2019, "2018-09-15", "2018-12-15"), (2014, "2013-11-01", "2014-02-25"))):
        q = m[(m.hj_year == y) & (m.date >= lo) & (m.date <= hi)]
        a_.plot(q.date, q.rc_R1, "o-", ms=3, color="#4C78A8", label="ρ_impl H−J (R1)")
        a2 = a_.twinx()
        a2.plot(q.date, q.hj, color="#E45756", lw=1, label="H−J spread")
        a_.set_title(f"winter {y - 1}-{str(y)[2:]}", fontsize=9)
        a_.set_ylabel("ρ_impl")
        a2.set_ylabel("H−J $/MMBtu")
        a_.tick_params(axis="x", labelsize=7, rotation=30)
    fig.tight_layout()
    fig.savefig(FIG / "v5_implied_corr_episodes.png", dpi=120)
    plt.close(fig)


def main():
    s = crp.prep()
    v, tot, diag = crp.validity(s)
    save(v, "step1_validity")
    OUT["validity_all_rows"] = tot
    OUT["rho_gt1_diag"] = {k: float(x) for k, x in diag.items()}
    q = s[s.tdays.between(5, 126) & ~s.stale]
    OUT["q_hat_vs_real_by_bucket"] = s.groupby("bucket")[["q_real", "q_hat"]].median().round(3).to_dict()
    OUT["rho_variant_medians_T5_126"] = q[["rho_R0", "rho_R1", "rho_R2"]].median().round(4).to_dict()
    t = save(crp.crp_table(s), "step2_crp")
    o, _ = crp.by_cuts(s)
    for k, d in o.items():
        save(d, f"step2_by_{k}")
    save(crp.var_decomp(s), "step2_var_decomp")
    # step 3
    tab, per, T, D = trades_tables()
    ch = tab[(tab.cso_hs == 0.04) & (tab.hedge == "band") & (tab.window == "choose") & (tab.E != "all")
             & (tab.strategy == "D_corr_rho_neutral_vega")]
    E_sel = int(ch.sort_values("t_like").E.iloc[-1])
    OUT["E_selected_on_choose"] = E_sel
    port, curves = portfolio(T, D, E_sel)
    episodes_by_trade(per)
    # step 4
    m = indicator.build_weekly()
    m.to_pickle(CACHE / "hj_weekly.pkl")
    save(indicator.describe(m), "step4_indicator")
    ov = indicator.overlay(m)
    save(ov, "step4_overlay")
    ovs = ov[ov.hj_year <= 2021].groupby(["rule", "window"]).agg(mean=("pnl", "mean"), worst_dd=("maxdd", "min"),
                                                                 days_flat=("days_flat", "sum")).reset_index()
    save(ovs, "step4_overlay_summary")
    figures(s, curves, m)
    # variant count
    OUT["variants"] = {"rho horizon treatments": 3, "drop/cap": 2, "CRP targets": 2,
                       "trade grid": "4 cso costs x 3 hedges x 5 strategies x 5 entry sets",
                       "indicator": "3 signals x 3 targets x 2 benchmarks + 5 overlay rules + 2 twins"}
    with open(CACHE / "summary.json", "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str))


if __name__ == "__main__":
    main()
