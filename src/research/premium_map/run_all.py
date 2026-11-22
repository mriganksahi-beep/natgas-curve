"""Premium map of the Henry Hub curve, 2010-06 .. 2021-12 (prices + public storage only).

    python -m src.research.premium_map.run_all

Writes tables to data/processed/research_premium_map/ and figures to
reports/agents/figures/premium_map_*.png. Takes ~1-2 minutes on 1 CPU.
"""
import io
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from .data import CACHE, FIG, CUTOFF, load_wide, storage_state
from .spreads import PAIR_ORDER, weekly_panel, drift_panel
from .stats import cell_table, boot_mean, ols_cluster
from . import portfolio as pfm

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
SPLIT = pd.Timestamp("2015-12-31")
DIV = LinearSegmentedColormap.from_list("div", ["#b8302f", "#e34948", "#f3a3a2", "#f0efec",
                                                "#86b6ef", "#2a78d6", "#1c5cab"])
INK, INK2 = "#0b0b0b", "#52514e"
out = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=out)


def build():
    W, expiry = load_wide()
    assert W.index.max() < CUTOFF
    fr = W.index[W.index.weekday == 4]
    wk = weekly_panel(W, expiry, fr)
    st = storage_state(fr)
    wk = wk.merge(st[["storage", "dev", "proj_endmar", "proj_endoct"]], left_on="t", right_index=True, how="left")
    assert wk.t.max() < CUTOFF
    steps = pfm.monthly_steps(W, expiry, fr)
    wk.to_pickle(CACHE / "weekly_panel.pkl")
    steps.to_pickle(CACHE / "monthly_steps.pkl")
    return wk, steps


# ----------------------------------------------------------------------------- 1. map
def premium_map(P):
    C = cell_table(P, "drift")
    Cp = cell_table(P, "pct")
    C.to_csv(CACHE / "map_usd.csv", index=False)
    Cp.to_csv(CACHE / "map_pct.csv", index=False)
    say("\n=== 1. PREMIUM MAP: mean drift V(t)-V(T), cents/MMBtu, by pair x lead L (months) ===")
    say((C.pivot(index="pair", columns="L", values="mean").reindex(PAIR_ORDER) * 100).round(1).to_string())
    say("\nbootstrap t (gas-year blocks)")
    say(C.pivot(index="pair", columns="L", values="t").reindex(PAIR_ORDER).round(1).to_string())
    say("\nshare of years with positive drift")
    say(C.pivot(index="pair", columns="L", values="frac_pos").reindex(PAIR_ORDER).round(2).to_string())
    say("\n% of average leg price")
    say(Cp.pivot(index="pair", columns="L", values="mean").reindex(PAIR_ORDER).round(1).to_string())
    say(f"\ncells: {len(C)}; CI excludes 0 ($): {((C.lo > 0) | (C.hi < 0)).sum()}; (%): {((Cp.lo > 0) | (Cp.hi < 0)).sum()}")
    return C, Cp


def heatmap(C, Cp):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw=dict(wspace=0.28))
    for ax, tab, scale, unit, lim in [(axes[0], C, 100, "¢/MMBtu", 30), (axes[1], Cp, 1, "% of price", 9)]:
        M = tab.pivot(index="pair", columns="L", values="mean").reindex(PAIR_ORDER) * scale
        sig = tab.assign(s=(tab.lo > 0) | (tab.hi < 0)).pivot(index="pair", columns="L", values="s").reindex(PAIR_ORDER)
        im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if np.isnan(v):
                    continue
                strong = abs(v) > 0.55 * lim
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if strong else INK, fontweight="bold" if sig.values[i, j] else "normal")
                if sig.values[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_xticks(range(M.shape[1]), M.columns)
        ax.set_yticks(range(M.shape[0]), M.index)
        ax.set_xlabel("lead time L: months before the near (first) leg expires", color=INK2)
        ax.axhline(11.5, color="white", lw=3)
        ax.set_title(f"Mean drift to expiry, {unit}", loc="left", color=INK, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
    fig.suptitle("Where the Henry Hub curve's premia sit, 2010-06 to 2021-12 (weekly Fridays)\n"
                 "drift = spread at lead L − spread at expiry; blue = spread falls into expiry (short earns), "
                 "red = rises (long earns); boxed/bold = 95% gas-year-bootstrap CI excludes 0",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.005, "Strips (bottom two rows): Win = Nov–Mar avg, Sum = Apr–Oct avg; terminal value = "
             "each leg's own final settle; lead measured to the first leg's expiry.", fontsize=8, color=INK2)
    fig.savefig(FIG / "premium_map_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 3a. stability
def stability(P):
    early = cell_table(P[P.expiry <= SPLIT], "drift")
    late = cell_table(P[P.expiry > SPLIT], "drift")
    m = early.merge(late, on=["pair", "L"], suffixes=("_e", "_l"))
    m.to_csv(CACHE / "stability_cells.csv", index=False)
    r = np.corrcoef(m.mean_e, m.mean_l)[0, 1]
    rs = m[["mean_e", "mean_l"]].rank().corr().iloc[0, 1]
    sig_e = m[(m.t_e.abs() >= 2)]
    agree = (np.sign(sig_e.mean_e) == np.sign(sig_e.mean_l)).mean()
    say("\n=== 3a. STABILITY: 2010-15 expiries vs 2016-21 expiries, cell means ($) ===")
    say(f"cells {len(m)}; Pearson r = {r:.2f}; Spearman = {rs:.2f}; cells with |t_early|>=2: {len(sig_e)}, "
        f"sign agrees late: {agree:.0%}; late |t|>=2 same sign: {((sig_e.t_l.abs() >= 2) & (np.sign(sig_e.mean_e) == np.sign(sig_e.mean_l))).sum()}")
    # per-pair summary at L = 3..10 average
    g = m[m.L.between(3, 10)].groupby("pair")[["mean_e", "mean_l"]].mean().reindex(PAIR_ORDER) * 100
    say("mean drift over L=3..10 (cents): early vs late")
    say(g.round(1).to_string())
    # without Mar-Apr / strips
    sub = m[~m.pair.isin(["Mar-Apr", "Win-nextSum", "Win-prevSum"])]
    say(f"excluding Mar-Apr and strips: r = {np.corrcoef(sub.mean_e, sub.mean_l)[0, 1]:.2f} (n={len(sub)})")
    # percent units
    e2, l2 = cell_table(P[P.expiry <= SPLIT], "pct"), cell_table(P[P.expiry > SPLIT], "pct")
    m2 = e2.merge(l2, on=["pair", "L"], suffixes=("_e", "_l"))
    say(f"percent units: Pearson r = {np.corrcoef(m2.mean_e, m2.mean_l)[0, 1]:.2f}; ex Mar-Apr/strips "
        f"{np.corrcoef(*m2[~m2.pair.isin(['Mar-Apr', 'Win-nextSum', 'Win-prevSum'])][['mean_e', 'mean_l']].values.T)[0, 1]:.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = 0.42
    ax.axhline(0, color="#d6d5d0", lw=1, zorder=0)
    ax.axvline(0, color="#d6d5d0", lw=1, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color="#d6d5d0", lw=1, ls="--", zorder=0)
    groups = [("Mar-Apr", "#2a78d6"), ("strips", "#eb6834"), ("other pairs", "#8a8984")]
    for lab, col in groups:
        if lab == "strips":
            s = m[m.pair.isin(["Win-nextSum", "Win-prevSum"])]
        elif lab == "Mar-Apr":
            s = m[m.pair == "Mar-Apr"]
        else:
            s = m[~m.pair.isin(["Mar-Apr", "Win-nextSum", "Win-prevSum"])]
        ax.scatter(s.mean_e, s.mean_l, s=26, color=col, edgecolor="white", lw=0.8, label=lab, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("mean drift, 2010–15 expiries ($/MMBtu)", color=INK2)
    ax.set_ylabel("mean drift, 2016–21 expiries ($/MMBtu)", color=INK2)
    ax.set_title(f"Is the map stable? One dot per pair × lead cell\nPearson r = {r:.2f} "
                 f"(ex Mar−Apr & strips: {np.corrcoef(sub.mean_e, sub.mean_l)[0, 1]:.2f})", loc="left", fontsize=10.5)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "premium_map_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return m


# ----------------------------------------------------------------------------- 2. conditioning
FAMILIES = [("Mar-Apr", range(3, 13)), ("Win-nextSum", range(1, 13)), ("Win-prevSum", range(1, 11)),
            ("Apr-May", range(3, 13)), ("Dec-Jan", range(1, 13)), ("Oct-Nov", range(6, 13)),
            ("Oct-Jan", range(6, 13)), ("Nov-Dec", range(1, 13))]


def conditioning(P):
    P = P.copy()
    P["state"] = np.where(P.proj_endmar < 1500, "scarce", np.where(P.proj_endoct > 3950, "full", "normal"))
    P.loc[P.dev.isna(), "state"] = np.nan
    say("\n=== 2. CONDITIONING (storage known at t, 2013+; states from the 5-yr-average path: "
        "scarce = projected end-Mar < 1,500 Bcf, full = projected end-Oct > 3,950 Bcf) ===")
    rows = []
    for fam, Ls in FAMILIES:
        g = P[(P.pair == fam) & P.L.isin(Ls) & P.dev.notna() & P.rv13.notna()].copy()
        r = dict(family=f"{fam} L{min(Ls)}-{max(Ls)}", n=len(g), years=g.gy.nunique())
        for s in ["scarce", "normal", "full"]:
            h = g[g.state == s]
            if h.gy.nunique() >= 2:
                mm, lo, hi, _ = boot_mean(h.pct.values, h.gy.values)
                r[s] = f"{mm:+.1f} [{lo:+.1f},{hi:+.1f}] ({h.gy.nunique()}y)"
            else:
                r[s] = f"n/a ({h.gy.nunique()}y)"
        fe = [(g.L == l).astype(float) for l in sorted(g.L.unique())[1:]]
        b, lo, hi, _ = ols_cluster(g.pct.values, np.column_stack([g.dev / 100] + fe), g.gy.values, nboot=1000)
        r["slope_dev_%per100Bcf"] = f"{b[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]"
        z = lambda x: (x - x.mean()) / x.std()
        X = np.column_stack([z(g.dev), z(g.apr_px), z(np.log(g.rv13 + 1e-4))] + fe)
        b, lo, hi, _ = ols_cluster(g.pct.values, X, g.gy.values, nboot=1000)
        for i, nm in enumerate(["dev", "aprpx", "logrv"], start=1):
            r[f"{nm}_per_sd"] = f"{b[i]:+.1f} [{lo[i]:+.1f},{hi[i]:+.1f}]"
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(CACHE / "conditioning.csv", index=False)
    say("drift in % of price; [95% CI, gas-year cluster bootstrap]; regressions include lead fixed effects")
    say(T[["family", "n", "years", "scarce", "normal", "full", "slope_dev_%per100Bcf"]].to_string(index=False))
    say("\nmultivariate (standardised regressors; % drift per 1 sd):")
    say(T[["family", "dev_per_sd", "aprpx_per_sd", "logrv_per_sd"]].to_string(index=False))
    return T


# ----------------------------------------------------------------------------- 3b. robustness of top cells
def robustness(P, C):
    C2 = C.copy()
    C2["abst"] = C2.t.abs()
    top = C2.sort_values("abst", ascending=False).drop_duplicates("pair").head(4)
    say("\n=== 3b. ROBUSTNESS of the 4 biggest cells (largest |t| per pair) ===")
    yrs = []
    for _, c in top.iterrows():
        g = P[(P.pair == c.pair) & (P.L == c.L)].sort_values("gy")
        yy = g.set_index(g.expiry.dt.year)[["drift", "pct"]].copy()
        yy["drift"] *= 100
        yy.columns = [f"{c.pair} L{c.L} ¢", f"{c.pair} L{c.L} %"]
        yrs.append(yy)
        loo = [g.drop(i).drift.mean() * 100 for i in g.index]
        e, l = g[g.expiry <= SPLIT], g[g.expiry > SPLIT]
        me, loe, hie, _ = boot_mean(e.drift.values * 100) if len(e) > 1 else (np.nan,) * 4
        ml, lol, hil, _ = boot_mean(l.drift.values * 100) if len(l) > 1 else (np.nan,) * 4
        mp, lop, hip, tp = boot_mean(g.pct.values, g.gy.values)
        say(f"{c.pair} L{c.L}: mean {c['mean'] * 100:+.1f}¢ [{c.lo * 100:+.1f},{c.hi * 100:+.1f}], t {c.t:.1f}, "
            f"{c.frac_pos:.0%} of {c.n} years >0 | LOO range {min(loo):+.1f} to {max(loo):+.1f}¢ | "
            f"2010-15 {me:+.1f}¢ [{loe:+.1f},{hie:+.1f}] n={len(e)}; 2016-21 {ml:+.1f}¢ [{lol:+.1f},{hil:+.1f}] n={len(l)} | "
            f"% units {mp:+.1f}% [{lop:+.1f},{hip:+.1f}] t {tp:.1f}")
    Y = pd.concat(yrs, axis=1).sort_index()
    Y.to_csv(CACHE / "top_cells_by_year.csv")
    say("\nyear-by-year (expiry year of the near leg)")
    say(Y.round(1).to_string())
    return top


# ----------------------------------------------------------------------------- 4. portfolio
def portfolio(steps):
    say("\n=== 4. PREMIUM-HARVEST PORTFOLIO (selection on 2010-15, check 2016-21; 4 ticks/contract/side) ===")
    cellsA = pfm.select_cells(steps)
    cellsA.to_csv(CACHE / "portfolio_A_cells.csv", index=False)
    say(f"Variant A (monthly cells, |t|>=2): {cellsA.sel.sum()} of {len(cellsA)} cells selected")
    say(cellsA[cellsA.sel][["pair", "L", "mean", "t", "n", "sign"]].round(4).to_string(index=False))
    pfA = pfm.run_portfolio(steps, cellsA)
    candB, bestB = pfm.select_pairs(steps)
    bestB.to_csv(CACHE / "portfolio_B_pairs.csv", index=False)
    say("\nVariant B (hold from L* to expiry; L* = argmax |t| on 2010-15):")
    say(bestB[["pair", "L", "drift", "lo", "hi", "t", "n", "sel", "sign"]].round(3).to_string(index=False))
    pfB = pfm.run_hold(steps, bestB)
    bench = pd.DataFrame([dict(pair="Mar-Apr", L=4, sign=-1.0)])          # short Mar-Apr from ~Nov to expiry
    pfM = pfm.run_hold(steps, None, only=bench)
    bench12 = pd.DataFrame([dict(pair="Mar-Apr", L=12, sign=-1.0)])
    pfM12 = pfm.run_hold(steps, None, only=bench12)
    series = {}
    rows = []
    for lab, pf in [("A cells monthly", pfA), ("B hold-to-expiry", pfB),
                    ("Mar-Apr short from L4 (Nov)", pfM), ("Mar-Apr short from L12", pfM12)]:
        net, gross = pfm.monthly_series(pf), pfm.monthly_series(pf, "gross")
        series[lab] = net
        for per, sl in [("2010-15", net.index <= SPLIT), ("2016-21", net.index > SPLIT)]:
            s = pfm.summarize(net[sl], lab)
            sg = pfm.summarize(gross[sl], lab)
            rows.append(dict(strategy=lab, period=per, sharpe_net=s["sharpe"], ci=f"[{s['sharpe_lo']:.2f},{s['sharpe_hi']:.2f}]",
                             sharpe_gross=sg["sharpe"], total_net=s["total"], cost=(gross[sl] - net[sl]).sum(),
                             worst_year=f"{s['worst_year']:,.0f} ({s['worst_year_id']})",
                             worst_month=f"{s['worst_month']:,.0f} ({s['worst_month_id'][:7]})", pos_years=s["pos_years"]))
    R = pd.DataFrame(rows)
    R.to_csv(CACHE / "portfolio_results.csv", index=False)
    say("\nrisk budget $1,000 per spread per month ex ante; $ figures are totals per period")
    say(R.round(2).to_string(index=False))
    S = pd.DataFrame(series)
    S.to_csv(CACHE / "portfolio_monthly.csv")
    say("\ncorrelation of monthly net P&L (2010-07..2021-12 / 2016-21 only):")
    for lab in ["A cells monthly", "B hold-to-expiry"]:
        for b in ["Mar-Apr short from L4 (Nov)", "Mar-Apr short from L12"]:
            say(f"  {lab} vs {b}: {S[lab].corr(S[b]):+.2f} / {S[S.index > SPLIT][lab].corr(S[S.index > SPLIT][b]):+.2f}")
    # gross P&L share from winter-related pairs
    for lab, pf in [("A", pfA), ("B", pfB)]:
        by = pf.groupby("pair")[["gross", "cost", "net"]].sum().round(0)
        say(f"\n{lab}: P&L by pair, full sample\n{by.to_string()}")
    # gross-only sensitivity, and A with t>=1.5 / 3 (variants)
    for tsel in [1.5, 3.0]:
        cA = pfm.select_cells(steps, t_sel=tsel)
        net = pfm.monthly_series(pfm.run_portfolio(steps, cA))
        say(f"A with |t|>={tsel}: {cA.sel.sum()} cells; 2016-21 net Sharpe {pfm.summarize(net[net.index > SPLIT], '')['sharpe']:.2f}")
    for tk in [0, 1, 2]:
        net = pfm.monthly_series(pfm.run_hold(steps, bestB, ticks=tk))
        netA = pfm.monthly_series(pfm.run_portfolio(steps, cellsA, ticks=tk))
        say(f"{tk} ticks: 2016-21 Sharpe A {pfm.summarize(netA[netA.index > SPLIT], '')['sharpe']:.2f}, "
            f"B {pfm.summarize(net[net.index > SPLIT], '')['sharpe']:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = {"A cells monthly": "#2a78d6", "B hold-to-expiry": "#eb6834",
            "Mar-Apr short from L4 (Nov)": "#1baf7a", "Mar-Apr short from L12": "#8a8984"}
    for lab, col in cols.items():
        c = S[lab].cumsum() / 1000
        ax.plot(c.index, c.values, color=col, lw=2, label=lab)
        ax.text(c.index[-1] + pd.Timedelta(days=40), c.values[-1], lab.split(" (")[0], color=INK2, fontsize=8, va="center")
    ax.axvline(pd.Timestamp("2016-01-01"), color=INK2, lw=1, ls="--")
    ax.text(pd.Timestamp("2016-02-01"), ax.get_ylim()[1] * 0.92, "selection ← | → check", fontsize=8.5, color=INK2)
    ax.axhline(0, color="#d6d5d0", lw=1)
    ax.set_ylabel("cumulative net P&L, $000 (4 ticks/side)", color=INK2)
    ax.set_title("Premium-harvest portfolios vs the short widowmaker ($1,000 ex-ante monthly risk per spread)",
                 loc="left", fontsize=10.5)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_xlim(right=pd.Timestamp("2023-06-01"))
    fig.savefig(FIG / "premium_map_portfolio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return R, S, pfA, pfB


def main():
    wk, steps = build()
    P = drift_panel(wk, extra=("dev", "proj_endmar", "proj_endoct"))
    P.to_pickle(CACHE / "drift_panel.pkl")
    say(f"panel: {len(wk)} pair-Fridays, {len(P)} pair-year-lead cells, Fridays {wk.t.min().date()}..{wk.t.max().date()}; "
        f"near-leg expiries {P.expiry.min().date()}..{P.expiry.max().date()}")
    C, Cp = premium_map(P)
    heatmap(C, Cp)
    conditioning(P)
    stability(P)
    robustness(P, C)
    portfolio(steps)
    (CACHE / "results.txt").write_text(out.getvalue())


if __name__ == "__main__":
    main()
