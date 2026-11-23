"""Report tables, episode log, frozen constants and figures for v4_widowmaker (run after run.py).

    PYTHONPATH=. python -m src.research.v4_widowmaker.tables
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.v4_widowmaker import data as D

C = D.CACHE
KEY = {  # label -> rule name (selected cells per entry come from selected.csv)
    "G w th1.0 f0.0": "G: weekly tight-news cut (post-hoc)",
    "Gwarm w th1.0 f0.0": "control: warm-news cut",
}


def fmt(v, d=0):
    return "–" if pd.isna(v) else f"{v:,.{d}f}"


def main():
    S = pd.read_csv(C / "summary.csv")
    R = pd.read_csv(C / "winter_results.csv")
    sel = pd.read_csv(C / "selected.csv")
    out = []

    # --- Table 1: headline per entry
    for entry in ("nov", "dec"):
        names = ["base", "half"] + sel[sel.entry == entry].name.tolist() + ["G w th1.0 f0.0", "Gwarm w th1.0 f0.0"]
        if entry == "dec":
            names += ["E th0.5"]
        lines = ["| Rule | Train mean / worst DD | R train | Check mean / worst DD | R check | All 11: mean [CI] | worst DD | worst winter | R all | P(beat half) | R lag 1 | R 4 ticks |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            lag = 1 if n.startswith("D ") else 0
            def g(w, lg=lag, tk=2):
                x = S[(S.entry == entry) & (S.name == n) & (S.window == w) & (S.lag == lg) & (S.ticks == tk)]
                return x.iloc[0] if len(x) else None
            tr, ck, al = g("train"), g("check"), g("all")
            l1 = g("all", lg=1) if n not in ("base", "half") else None
            t4 = g("all", tk=4)
            lines.append(f"| {n} | {fmt(tr['mean'])} / {fmt(tr.worst_dd)} | {fmt(tr.ratio, 2)} | "
                         f"{fmt(ck['mean'])} / {fmt(ck.worst_dd)} | {fmt(ck.ratio, 2)} | "
                         f"{fmt(al['mean'])} [{fmt(al.mean_lo)}, {fmt(al.mean_hi)}] | {fmt(al.worst_dd)} | "
                         f"{fmt(al.worst_winter)} | {fmt(al.ratio, 2)} | {fmt(al.p_beat_half, 2)} | "
                         f"{fmt(l1.ratio, 2) if l1 is not None else '–'} | {fmt(t4.ratio, 2)} |")
        out.append(f"### Table 1{entry[0]}: entry = {entry}\n" + "\n".join(lines))

    # --- Table 2: per-winter P&L (Nov entry, 2 ticks)
    for entry in ("nov", "dec"):
        names = ["base", "half"] + sel[sel.entry == entry].name.tolist() + ["G w th1.0 f0.0"]
        x = R[(R.entry == entry) & (R.ticks == 2) & R.name.isin(names)
              & ((R.lag == 0) | (R.family == "D_addon"))]
        pv = x.pivot_table(index="Y", columns="name", values="pnl")[names]
        dd = x.pivot_table(index="Y", columns="name", values="dd")[names]
        lines = ["| Winter | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for Y in pv.index:
            lines.append(f"| {D.wl(Y)}{'*' if Y == 2022 else ''} | " +
                         " | ".join(f"{fmt(pv.at[Y, n])} ({fmt(dd.at[Y, n])})" for n in names) + " |")
        out.append(f"### Table 2{entry[0]}: per-winter P&L (worst MTM drawdown), entry = {entry}\n" + "\n".join(lines))

    # --- share above half
    sh = pd.read_csv(C / "share_above_half.csv")
    out.append("### Share of cells with R > 1\n" + sh.round(2).to_markdown(index=False))

    # --- episodes
    P = pd.read_pickle(C / "paths.pkl")
    ep = []
    dates = {2014: ["2014-01-17", "2014-01-24", "2014-01-29", "2014-02-14", "2014-02-20", "2014-02-26"],
             2019: ["2018-11-09", "2018-11-12", "2018-11-13", "2018-11-14", "2018-11-15", "2019-02-26"],
             2022: ["2021-11-05", "2021-11-15", "2021-12-01", "2021-12-31"]}
    for entry in ("nov", "dec"):
        names = ["base"] + sel[sel.entry == entry].name.tolist() + ["G w th1.0 f0.0"]
        for Y, ds in dates.items():
            for n in names:
                lag = 1 if n.startswith("D ") else 0
                p = P[(entry, n, lag, Y)]
                c = np.r_[0, p.cum.values]
                row = dict(entry=entry, winter=D.wl(Y), rule=n, final=p.cum.iloc[-1],
                           worst_dd=float(np.min(c - np.maximum.accumulate(c))))
                for d in ds:
                    d = pd.Timestamp(d)
                    if d in p.index:
                        row[f"pos {d:%d%b}"] = p.at[d, "pos"]
                        row[f"cum {d:%d%b}"] = p.at[d, "cum"]
                ep.append(row)
    ep = pd.DataFrame(ep)
    ep.to_csv(C / "episodes.csv", index=False)
    for Y, ds in dates.items():
        sub = ep[ep.winter == D.wl(Y)].dropna(axis=1, how="all")
        out.append(f"### Episode {D.wl(Y)}\n" + sub.round(1).to_markdown(index=False))

    # --- frozen constants for the locked / live test (full dev, Oct-Apr, <= 2021-12-31)
    sg = D.signals()
    m = sg.index.month.isin([10, 11, 12, 1, 2, 3, 4])
    const = {"RMS_L23_dev": float(np.sqrt(np.nanmean(sg.L23[m] ** 2))),
             "RMS_L123_dev": float(np.sqrt(np.nanmean(sg.L123[m] ** 2))),
             "n_days": int(sg.L23[m].notna().sum())}
    json.dump(const, open(C / "frozen_constants.json", "w"), indent=1)
    out.append("### Frozen constants\n" + json.dumps(const))
    # check: E th1.0 / th0.5 with the frozen full-dev RMS instead of walk-forward (descriptive)
    from src.research.v4_widowmaker import rules as RU
    zf = (sg.L23 / const["RMS_L23_dev"]).rolling(5, min_periods=5).mean()
    chk = []
    for entry in ("dec", "nov"):
        W = D.winters(entry)
        for th in (0.5, 1.0):
            tot = []
            for Y, w in W.items():
                w2 = w.copy()
                w2["zL23_5"] = zf.reindex(w2.index)
                pos_f = RU.delay(w2, th=th)
                pos_w = RU.delay(w, th=th)
                tot.append((Y, RU.pnl(w2, pos_f, D.SIDE[2]).sum(), RU.pnl(w, pos_w, D.SIDE[2]).sum()))
            t = pd.DataFrame(tot, columns=["Y", "frozen", "walkfwd"])
            chk.append(dict(entry=entry, th=th, n_diff=int((abs(t.frozen - t.walkfwd) > 1).sum()),
                            mean_frozen=t[t.Y < 2022].frozen.mean(), mean_wf=t[t.Y < 2022].walkfwd.mean()))
    out.append("### Frozen-RMS vs walk-forward check (E rule; frozen uses in-sample RMS, descriptive)\n"
               + pd.DataFrame(chk).to_markdown(index=False, floatfmt=".1f"))

    # --- conditional drift by forecast state (Nov entry live days, next-day P&L of 1 short spread)
    W = D.winters("nov")
    rows = []
    for Y, w in W.items():
        if Y == 2022:
            continue
        nxt = -w.S.diff().shift(-1) * D.MULT
        rows.append(pd.DataFrame({"Y": Y, "zl": w.zL23_5, "zn": w.zNS_w, "S": w.S, "p": nxt}))
    x = pd.concat(rows).dropna()
    big = x.nsmallest(20, "p")
    tab = []
    for lab, mask in (("zL23_5 <= 0", x.zl <= 0), ("0 < zL23_5 <= 1", (x.zl > 0) & (x.zl <= 1)),
                      ("zL23_5 > 1", x.zl > 1), ("zNS_w > 1", x.zn > 1), ("zNS_w <= 1", x.zn <= 1),
                      ("S > 0.40", x.S > 0.4)):
        tab.append(dict(state=lab, share_days=mask.mean(), mean_next_day=x.p[mask].mean(),
                        sd_next_day=x.p[mask].std(), share_of_20_worst_days=mask.loc[big.index].mean()))
    rng = np.random.default_rng(7)
    ys = sorted(x.Y.unique())
    diffs = {"zL23_5 >1 minus <=0": [], "zNS_w >1 minus <=1": []}
    for _ in range(4000):
        xb = pd.concat([x[x.Y == y] for y in rng.choice(ys, len(ys))])
        diffs["zL23_5 >1 minus <=0"].append(xb.p[xb.zl > 1].mean() - xb.p[xb.zl <= 0].mean())
        diffs["zNS_w >1 minus <=1"].append(xb.p[xb.zn > 1].mean() - xb.p[xb.zn <= 1].mean())
    ci = {k: np.nanpercentile(v, [2.5, 50, 97.5]).round(0).tolist() for k, v in diffs.items()}
    out.append("Winter-bootstrap CI of differences in mean next-day P&L: " + json.dumps(ci))
    out.append("### Next-day short P&L by forecast state (Nov-entry live days, 2010-11..2020-21)\n"
               + pd.DataFrame(tab).round(2).to_markdown(index=False))

    open(C / "tables.md", "w").write("\n\n".join(out))
    figures(S, P, sel)
    print("\n\n".join(out))


def figures(S, P, sel):
    blue, orange, aqua, ink, grey = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
    # Fig 1: frontier
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, entry in zip(axs, ("nov", "dec")):
        x = S[(S.entry == entry) & (S.window == "all") & (S.ticks == 2)
              & ((S.lag == 0) | (S.family == "D_addon")) & (S.family != "bench")]
        pre = x[x.family.str[0].isin(list("ABCDE"))]
        post = x[x.family.str[0].isin(list("FG"))]
        ctl = x[x.family == "Z_control"]
        ax.plot([0, 1.6], [0, 1.6], color=grey, lw=1.2, ls="--")
        ax.text(1.45, 1.36, "half / any\nfixed size", color="#52514e", fontsize=8, ha="right")
        ax.scatter(pre.dd_kept, pre.kept, s=22, color="#c3c2b7", label="pre-registered cells (A–E)", zorder=2)
        ax.scatter(post.dd_kept, post.kept, s=22, color=orange, label="post-hoc cells (F, G)", zorder=3,
                   edgecolor="white", lw=0.6)
        ax.scatter(ctl.dd_kept, ctl.kept, s=26, marker="x", color=ink, label="warm-news control", zorder=3)
        sx = x[x.name.isin(sel[sel.entry == entry].name)]
        ax.scatter(sx.dd_kept, sx.kept, s=46, color=blue, edgecolor="white", lw=0.8,
                   label="train-selected cells", zorder=4)
        for _, r in sx.iterrows():
            ax.annotate(r["name"], (r.dd_kept, r.kept), fontsize=7, color="#52514e",
                        xytext=(4, 3), textcoords="offset points")
        ax.scatter([1, 0.5], [1, 0.5], s=40, color=ink, zorder=5)
        ax.annotate("base", (1, 1), fontsize=8, xytext=(-26, 4), textcoords="offset points")
        ax.annotate("half", (0.5, 0.5), fontsize=8, xytext=(-24, 4), textcoords="offset points")
        ax.set_xlim(0, 1.6); ax.set_ylim(-0.1, 2.3)
        ax.set_title(f"Entry: {'first Friday of Nov' if entry == 'nov' else '1 Dec'} — 11 winters 2010-11..2020-21",
                     fontsize=9, loc="left")
        ax.set_xlabel("worst drawdown kept (rule ÷ base)")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("premium kept (mean P&L, rule ÷ base)")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Above the dashed line = beats holding a smaller fixed size", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v4_widowmaker_frontier.png", dpi=150)
    plt.close(fig)

    # Fig 2: episodes, Nov entry
    a_sel = sel[(sel.entry == "nov") & (sel.family == "A_cold_cut")].name.iloc[0]
    d_sel = sel[(sel.entry == "nov") & (sel.family == "D_addon")].name.iloc[0]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex="col",
                            gridspec_kw={"height_ratios": [2.2, 1]})
    for j, Y in enumerate((2014, 2019)):
        ax, az = axs[0, j], axs[1, j]
        b = P[("nov", "base", 0, Y)]
        ax.plot(b.index, b.cum, color=ink, lw=1.8, label="base (1 spread)")
        ax.plot(b.index, b.cum * 0.5, color=grey, lw=1.5, ls="--", label="half size")
        for n, lag, col, lab in ((a_sel, 0, aqua, f"A level cut ({a_sel[2:]})"),
                                 ("G w th1.0 f0.0", 0, blue, "G weekly tight-news cut (post-hoc)"),
                                 (d_sel, 1, orange, f"D add-on ({d_sel[2:]})")):
            p = P[("nov", n, lag, Y)]
            ax.plot(p.index, p.cum, color=col, lw=1.5, label=lab)
        ax.axhline(0, color=grey, lw=0.6)
        ax.set_title(f"Winter {D.wl(Y)}: cumulative P&L per spread ($), Nov entry", fontsize=9, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        az.plot(b.index, b.zL23_5, color=aqua, lw=1.3, label="forecast level z (days 6–15, 5-d mean)")
        zw = P[("nov", "G w th1.0 f0.0", 0, Y)]
        az.step(b.index, D.winters("nov")[Y].zNS_w.reindex(b.index), color=blue, lw=1.3, where="post",
                label="tight-scaled weekly news z")
        az.axhline(1, color=grey, lw=0.8, ls="--")
        az.axhline(0, color=grey, lw=0.6)
        az.set_ylabel("z (sd)", fontsize=8)
        az.grid(alpha=0.25, lw=0.5)
        for a in (ax, az):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=7)
            a.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    axs[1, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(D.FIG / "v4_widowmaker_episodes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
