"""Run every overlay variant, select on 2010-11..2016-17, check on 2017-18..2021-22 (truncated).

python -m src.research.premium_risk_management.run   (from repo root; ~1 min)
Outputs: data/processed/research_premium_risk_management/*.csv, run_log.txt
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

from src.research.premium_risk_management import data as D
from src.research.premium_risk_management import overlays as O

RNG = np.random.default_rng(20261122)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ------------------------------------------------------------------ variants
def hedge_betas(W):
    """Pooled OLS slope of daily dS on d(hedge) over live days; per winter and pooled on TRAIN."""
    rows = []
    for Y, w in W.items():
        lw = w[w.live]
        for col in ("N", "HO"):
            z = pd.concat([lw.S.diff(), lw[col].diff()], axis=1).dropna()
            b = np.polyfit(z.iloc[:, 1], z.iloc[:, 0], 1)[0]
            rows.append(dict(Y=Y, winter=D.wlabel(Y), inst=col, beta=b, corr=z.corr().iloc[0, 1],
                             n=len(z)))
    bt = pd.DataFrame(rows)
    pooled = {}
    for col in ("N", "HO"):
        zs = []
        for Y in D.TRAIN:
            lw = W[Y][W[Y].live]
            zs.append(pd.concat([lw.S.diff(), lw[col].diff()], axis=1).dropna())
        z = pd.concat(zs)
        pooled[col] = np.polyfit(z.iloc[:, 1], z.iloc[:, 0], 1)[0]
    return bt, pooled


def variant_specs(betas):
    V = [("base", "base", {})]
    for X, R in itertools.product([2000, 3000, 4000, 5000, 6000, 8000], [None, 2, 4]):
        V.append(("a_stop", f"stop X={X} re={R}", dict(X=X, R=R)))
    for X, R in itertools.product([2000, 3000, 4000, 5000, 6000, 8000], [None, 2, 4]):
        V.append(("a0_stop", f"stop0 X={X} re={R}", dict(X=X, R=R)))
    for X, R in itertools.product([2000, 3000, 4000, 5000, 6000, 8000], [None, 2, 4]):
        V.append(("b_trail", f"trail X={X} re={R}", dict(X=X, R=R)))
    for k, re in itertools.product([1.5, 2, 3, 4, 6], [False, True]):
        V.append(("c_vol", f"vol k={k} re={int(re)}", dict(k=k, re=re)))
    for m, re in itertools.product([1.5, 2, 2.5, 3], [False, True]):
        V.append(("d_phys", f"phys m={m} re={int(re)}", dict(m=m, re=re)))
    for L, re in itertools.product([0.25, 0.4, 0.6, 0.8], [False, True]):
        V.append(("d2_level", f"level L={L} re={int(re)}", dict(L=L, re=re)))
    for n, off in itertools.product([1, 2, 4, 6, 8], [0, 2, 4, 6]):
        if (n, off) != (1, 0):
            V.append(("e_stagger", f"stagger n={n} off={off}w", dict(n=n, off=off)))
    for inst, h in itertools.product(["N", "HO"], [0.25, 0.5, 1.0, 1.5]):
        V.append(("f_hedge", f"hedge {inst} h={h}xbeta", dict(inst=inst, units=h * betas[inst])))
    for st, wk in itertools.product([(11, 10), (11, 20), (12, 1), (12, 15), (1, 2)], [2, 4, 6]):
        V.append(("g_cal", f"flat {st[0]}/{st[1]} +{wk}w", dict(st=st, wk=wk)))
    return V


def run_variant(fam, p, W, alarm):
    out = {}
    for Y, w in W.items():
        hedge = None
        if fam == "base":
            pos = O.base(w)
        elif fam in ("a_stop", "b_trail"):
            pos = O.stop(w, p["X"], p["R"], trailing=fam == "b_trail")
        elif fam == "a0_stop":
            pos = O.stop(w, p["X"], p["R"], lag=0)
        elif fam == "c_vol":
            pos = O.vol_trigger(w, p["k"], p["re"])
        elif fam == "d_phys":
            if Y < 2017:
                continue  # Model 2 era only
            pos = O.phys_trigger(w, alarm, p["m"], p["re"])
        elif fam == "d2_level":
            pos = O.level_trigger(w, p["L"], p["re"])
        elif fam == "e_stagger":
            pos = O.stagger(w, p["n"], p["off"])
        elif fam == "f_hedge":
            pos = O.base(w)
            hedge = O.hedge_leg(w, p["units"])
        elif fam == "g_cal":
            pos = O.calendar(w, Y, p["st"], p["wk"])
        if hedge is None:
            daily = O.pnl(w, pos)
        else:
            daily = O.pnl(w, pos, hedge, p["inst"], D.SIDE_COST if p["inst"] == "N" else D.SIDE_COST / 2)
        out[Y] = pd.DataFrame({"pnl": daily, "pos": np.r_[np.asarray(pos)[:-1], 0.0]})
    return out


# ------------------------------------------------------------------ metrics
def winter_metrics(res):
    rows = []
    for Y, r in res.items():
        cum0 = np.r_[0.0, r.pnl.cumsum().values]
        dd = (cum0 - np.maximum.accumulate(cum0)).min()   # peak-to-trough of winter equity
        live = W_LIVE[Y]
        rows.append(dict(Y=Y, pnl=cum0[-1], dd=dd, worst_cum=cum0.min(),
                         exposure=r.pos.values[live].mean()))
    return pd.DataFrame(rows).set_index("Y")


W_LIVE = {}


def sortino(res, years):
    parts = []
    for Y in years:
        if Y in res:
            p = res[Y].pnl[W_LIVE[Y] | (np.arange(len(W_LIVE[Y])) == len(W_LIVE[Y]) - 1)]
            parts.append(p.resample("W-FRI").sum())   # all season weeks, flat weeks count as 0
    x = pd.concat(parts).values
    dn = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
    return np.mean(x) / dn * np.sqrt(52) if dn > 0 else np.nan


def boot_mean(x, B=4000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan, np.nan
    m = RNG.choice(x, (B, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summarise(wm, res, years, basewm):
    w = wm.loc[[y for y in years if y in wm.index]]
    b = basewm.loc[w.index]
    lo, hi = boot_mean(w.pnl)
    dlo, dhi = boot_mean(w.pnl - b.pnl)
    return dict(n=len(w), mean=w.pnl.mean(), lo=lo, hi=hi, worst_dd=w.dd.min(), mean_dd=w.dd.mean(),
                worst_winter=w.pnl.min(), sortino=sortino(res, w.index),
                kept=w.pnl.mean() / b.pnl.mean(), dd_cut=1 - w.dd.min() / b.dd.min(),
                mdd_cut=1 - w.dd.mean() / b.dd.mean(),
                diff=(w.pnl - b.pnl).mean(), diff_lo=dlo, diff_hi=dhi,
                calmar=w.pnl.mean() / abs(w.dd.min()) if w.dd.min() < 0 else np.inf)


def placebo(v, all_res, W, R, basewm, B=2000, sample="all"):
    yrs = [Y for Y in D.YEARS if Y in all_res[v]]
    gross, flat = {}, {}
    for Y in yrs:
        w = W[Y]
        i0, iend = int(np.argmax(w.live.values)), len(w) - 1
        dS = np.r_[0.0, np.diff(w.S.values)]
        gross[Y] = -dS[i0 + 1: iend + 1] * D.MULT          # P&L of holding over each day after entry
        pos = all_res[v][Y].pos.values[i0:iend]            # position held after close i0..iend-1
        flat[Y] = int((pos < 0.5).sum())
    obs = R[(R.variant == v) & (R["sample"] == sample)].iloc[0]
    kept, cut = np.empty(B), np.empty(B)
    bmean = basewm.loc[yrs].pnl.mean()
    bdd = basewm.loc[yrs].dd.min()
    for b in range(B):
        pn, dd = [], []
        for Y in yrs:
            g = gross[Y]
            n, k = len(g), flat[Y]
            hold = np.ones(n)
            if k > 0:
                st = RNG.integers(0, n - k + 1)
                hold[st: st + k] = 0.0
            daily = hold * g
            nchg = np.abs(np.diff(np.r_[1.0, hold, 0.0])).sum() + 1   # entry + changes + final exit
            daily = np.r_[-D.SIDE_COST, daily]
            daily[-1] -= 0  # costs added in total below
            cum = np.cumsum(daily)
            tot = cum[-1] - (nchg - 1) * D.SIDE_COST
            cum0 = np.r_[0.0, cum]
            dd.append((cum0 - np.maximum.accumulate(cum0)).min())
            pn.append(tot)
        kept[b], cut[b] = np.mean(pn) / bmean, 1 - min(dd) / bdd
    return dict(variant=v, kept=obs.kept, dd_cut=obs.dd_cut, plc_kept_med=np.median(kept),
                plc_cut_med=np.median(cut), p_kept=(kept >= obs.kept).mean(), p_cut=(cut >= obs.dd_cut).mean(),
                p_joint=((kept >= obs.kept) & (cut >= obs.dd_cut)).mean())


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    W = D.winters()
    for Y, w in W.items():
        W_LIVE[Y] = w.live.values.copy()
        W_LIVE[Y][-1] = False
    alarm = D.phys_m2_alarm().alarm.dropna()
    bt, betas = hedge_betas(W)
    V = variant_specs(betas)
    log(f"Variants: {len(V)} (families: {pd.Series([v[0] for v in V]).value_counts().to_dict()})")
    log(f"Hedge betas (pooled OLS dS on dX, train winters): {betas}")
    all_res, rows, wrows = {}, [], []
    for fam, name, p in V:
        res = run_variant(fam, p, W, alarm)
        all_res[name] = res
        wm = winter_metrics(res)
        for Y, r in wm.iterrows():
            wrows.append(dict(family=fam, variant=name, Y=Y, winter=D.wlabel(Y), **r.to_dict()))
    WM = pd.DataFrame(wrows)
    basewm = WM[WM.variant == "base"].set_index("Y")
    samples = {"train": D.TRAIN, "test": D.TEST, "all": D.YEARS, "m2": list(range(2017, 2023)),
               "test4": list(range(2018, 2022)), "all11": list(range(2011, 2022))}
    for fam, name, p in V:
        wm = WM[WM.variant == name].set_index("Y")
        for sname, yrs in samples.items():
            if fam == "d_phys" and sname == "train":
                continue
            rows.append(dict(family=fam, variant=name, sample=sname, **summarise(wm, all_res[name], yrs, basewm)))
    R = pd.DataFrame(rows)
    WM.to_csv(D.CACHE / "winter_metrics.csv", index=False)
    R.to_csv(D.CACHE / "summary.csv", index=False)
    bt.to_csv(D.CACHE / "hedge_betas.csv", index=False)
    pd.to_pickle(all_res, D.CACHE / "all_res.pkl")

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    log("\n== Base, per winter ==")
    log(basewm.round(0).assign(winter=[D.wlabel(y) for y in basewm.index]).to_string())
    log("\n== Hedge betas per winter ==")
    log(bt.pivot(index="winter", columns="inst", values=["beta", "corr"]).round(2).to_string())

    # selection on train: max Calmar (mean / |worst drawdown|), per family; d_phys fixed a priori m=2, re=1
    sel = {}
    for fam in R.family.unique():
        if fam in ("base", "a0_stop"):   # a0 = execution sensitivity of a_stop, not a candidate
            continue
        if fam == "d_phys":
            sel[fam] = "phys m=2 re=1"
            continue
        tr = R[(R.family == fam) & (R["sample"] == "train")]
        sel[fam] = tr.sort_values(["calmar", "mean"], ascending=False).variant.iloc[0]
    log("\n== Selected on train (max mean/|worst DD|) ==")
    cols = ["variant", "sample", "mean", "lo", "hi", "worst_dd", "mean_dd", "worst_winter", "sortino",
            "kept", "dd_cut", "diff", "diff_lo", "diff_hi"]
    selrows = R[R.variant.isin(["base"] + list(sel.values()))]
    for s in ("train", "test", "test4", "all", "all11", "m2"):
        log(f"-- {s} --")
        log(selrows[selrows["sample"] == s][cols].round(2).to_string(index=False))
    pd.Series(sel).to_csv(D.CACHE / "selected.csv")

    # robustness: full grid in test
    log("\n== Grid robustness: test sample, per family ==")
    g = []
    bt_test = R[(R.variant == "base") & (R["sample"] == "test")].iloc[0]
    for fam in R.family.unique():
        if fam == "base":
            continue
        x = R[(R.family == fam) & (R["sample"] == "test")]
        g.append(dict(family=fam, cells=len(x), med_kept=x.kept.median(), med_dd_cut=x.dd_cut.median(),
                      share_beat_calmar=(x.calmar > bt_test.calmar).mean(),
                      share_above_scaling=((x.kept + x.dd_cut) > 1).mean(),
                      best_kept_at_dd_cut_50=x[x.dd_cut >= 0.5].kept.max()))
    log(pd.DataFrame(g).round(2).to_string(index=False))
    # "above scaling": the trivial overlay (hold f < 1 spreads) gives kept = f, dd_cut = 1 - f

    log("\n== Stop with same-settle execution (lag 0) vs next-settle (lag 1), all winters ==")
    a = R[R.family.isin(["a_stop", "a0_stop"]) & (R["sample"] == "all")][["variant", "mean", "worst_dd", "kept", "dd_cut"]]
    log(a.round(2).to_string(index=False))

    # placebo: same flat days per winter, random timing
    log("\n== Timing placebo (all winters; 2,000 draws; one random contiguous flat block per winter with "
        "the overlay's flat-day count) ==")
    cand = list(sel.values()) + ["vol k=1.5 re=0", "vol k=2 re=0", "phys m=2 re=0", "stop X=2000 re=4",
                                 "stop X=2000 re=None", "level L=0.25 re=0", "flat 11/10 +2w", "stagger n=4 off=2w"]
    prow = []
    for v in dict.fromkeys(cand):
        if v.startswith("hedge"):
            continue
        prow.append(placebo(v, all_res, W, R, basewm))
    P = pd.DataFrame(prow)
    P.to_csv(D.CACHE / "placebo.csv", index=False)
    log(P.round(3).to_string(index=False))

    # frontier
    for s in ("all", "test", "train", "all11"):
        x = R[R["sample"] == s].copy()
        x = x[np.isfinite(x.kept) & np.isfinite(x.dd_cut)]
        x = x.sort_values(["dd_cut", "kept"], ascending=[False, False])
        front, best = [], -np.inf
        for _, r in x.iterrows():
            if r.kept > best + 1e-9:
                front.append(r)
                best = r.kept
        F = pd.DataFrame(front)
        F.to_csv(D.CACHE / f"frontier_{s}.csv", index=False)
        log(f"\n== Pareto frontier ({s}): premium kept vs worst-DD cut ==")
        log(F[["variant", "mean", "worst_dd", "worst_winter", "kept", "dd_cut", "mdd_cut", "sortino"]]
            .round(2).to_string(index=False))

    # episodes, day by day
    eps = {2014: ("2013-12-02", "2014-02-26"), 2019: ("2018-11-01", "2018-12-21")}
    show = ["base"] + list(sel.values())
    for Y, (a, b) in eps.items():
        frames = {}
        for v in show:
            if Y not in all_res[v]:
                continue
            r = all_res[v][Y]
            frames[v] = pd.DataFrame({"cum": r.pnl.cumsum(), "pos": r.pos})
        E = pd.concat(frames, axis=1).loc[a:b]
        E.to_csv(D.CACHE / f"episode_{Y}.csv")
        log(f"\n== Episode {D.wlabel(Y)}: cumulative net P&L ($) and position, selected overlays ==")
        sub = E.xs("cum", axis=1, level=1).round(-1)
        posd = E.xs("pos", axis=1, level=1)
        chg = (posd.diff().abs().sum(1) > 0) | (sub["base"].diff().abs() > 1500)
        log(sub[chg | (sub.index == sub.index[0]) | (sub.index == sub.index[-1])].astype(int).to_string())
        log("positions on those days:")
        log(posd[chg].round(2).to_string())
    log(f"\nRun time {time.time() - t0:.0f}s")
    (D.CACHE / "run_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__":
    main()
