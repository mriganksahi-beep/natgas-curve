"""Run every table of the fear-vs-physics study. Usage: python -m src.research.fear_vs_physics.run"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import analysis as A
from . import data as D

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
OUT = D.CACHE


class Tee:
    def __init__(self, f):
        self.f, self.o = open(f, "w"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.o.write(s)

    def flush(self):
        self.f.flush()
        self.o.flush()


def H(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def main():
    t0 = time.time()
    sys.stdout = Tee(OUT / "run_log.txt")
    raw = D.build_panel(refresh=True)
    p = A.add_indices(raw)
    H("0. Panel")
    print(f"rows {len(p)}, winters {p.winter.nunique()}, obs {p.obs.min().date()}..{p.obs.max().date()}")
    print(p.groupby("winter")[["S", "rv4", "fvol", "dev_now", "n_dev", "n_avg5", "x_an", "p_tail", "p_tail_st",
                                 "p_tail_m2", "n_an", "Fear", "Phys", "dS_exp"]].mean().round(3))
    print("corr:\n", p[["S", "Fear", "Phys", "rv4", "fvol", "p_tail", "dev_now", "wte"]].corr().round(2))

    H("1. Level of S on fear / physics (winter-block bootstrap, 2000 draws)")
    for k, v in A.level_tables(p).items():
        print("--", k)
        print(v.round(3))

    H("2. Drift on fear / physics and the decomposition")
    res, dec, b = A.drift_tables(p)
    print("level decomposition coefs (const, wte, Fear, Phys):", np.round(b, 4))
    print("sd of components:", dec[["S_fear", "S_phys", "S_time", "S_res"]].std().round(3).to_dict())
    rows = []
    for (y, m), v in res.items():
        print("--", y, m)
        print(v.round(3))
        for c in v.index:
            rows.append({"target": y, "model": m, "term": c, **v.loc[c].to_dict()})
    pd.DataFrame(rows).to_csv(OUT / "drift_tables.csv", index=False)
    dec.to_pickle(OUT / "decomp.pkl")

    H("2b. Decomposition robustness: leave one winter out, and alternative fear/physics definitions")
    for drop in [None, "2013-14", "2018-19", "2014-15", "2021-22"]:
        q = p if drop is None else p[p.winter != drop]
        d2, _ = A.decompose(q)
        for y in ("dS_exp", "dS_4w"):
            (bb, r), _ = A.fit(d2, y, ["S_fear", "S_phys", "S_time", "S_res"])
            print(f"drop {drop}: {y} fear {bb[1]:+.2f} phys {bb[2]:+.2f} time {bb[3]:+.2f} res {bb[4]:+.2f} R2 {r:.2f}")
    alt = []
    for fc, pc in [(["rv4"], ["p_tail"]), (["fvol"], ["p_tail"]), (["fvol"], A.PHYS), (["maxrise"], A.PHYS),
                   (A.FEAR, ["neg_dev_now"]), (A.FEAR, ["p_tail"]), (["rv2", "rv4", "rv8"], A.PHYS)]:
        q = raw.assign(neg_dev_now=-raw.dev_now, neg_x_an=-raw.x_an)
        q["Fear"] = pd.concat([(q[c] - q[c].mean()) / q[c].std() for c in fc], axis=1).mean(axis=1)
        q["Phys"] = pd.concat([(q[c] - q[c].mean()) / q[c].std() for c in pc], axis=1).mean(axis=1)
        d2, bl = A.decompose(q)
        r = {"fear": "+".join(fc), "phys": "+".join(pc), "lvl_fear": bl[2], "lvl_phys": bl[3]}
        for y in ("dS_exp", "dS_4w"):
            t = A.boot(d2, y, ["S_fear", "S_phys", "S_time", "S_res"], B=1000)
            for c in ("S_fear", "S_phys"):
                r[f"{y}_{c}"] = f"{t.loc[c, 'coef']:+.2f} [{t.loc[c, 'lo']:+.2f},{t.loc[c, 'hi']:+.2f}]"
        alt.append(r)
    alt = pd.DataFrame(alt)
    print(alt.to_string())
    alt.to_csv(OUT / "decomp_variants.csv", index=False)

    H("3. Walk-forward out-of-sample drift forecasts (indices and fits on winters < Y)")
    for fy in (2013, 2015):
        for y in ("dS_exp", "dS_4w"):
            wf = A.walk_forward(raw, y, first_Y=fy)
            t = A.oos_table(wf, y)
            print(f"-- {y}, test winters from Y={fy}")
            print(t.round(3))
            t.to_csv(OUT / f"oos_{y}_{fy}.csv", index=False)

    H("4. Model 2 subsample (origins Nov 2016 - Dec 2021): does Model 2's tail add to fear + analog physics?")
    for k, v in A.m2_tables(raw).items():
        print("--", k)
        print(v.round(3))

    H("5. Rules")
    sp = pd.read_pickle(D.CACHE / "spreads.pkl")
    r = A.wf_signals(raw)
    r = A.wf_resid_signal(r, A.time_only, name="TX")
    r.to_pickle(OUT / "panel_signals.pkl")
    gA = A.rule_grid(raw)
    gA.to_csv(OUT / "grid_A_rv4_ptail.csv", index=False)
    gA13 = A.rule_grid(raw[raw.Y >= 2013])
    gB = A.rule_grid_generic(r, "FP", [0.05, 0.10, 0.15, 0.20, 0.30], [1.01, 0.10, 0.20, 0.30, 0.50], Ys_train=range(2013, 2018))
    gC = A.rule_grid_generic(r, "FX", [0.0, 0.05, 0.10, 0.15, 0.20], Ys_train=range(2013, 2018))
    gT = A.rule_grid_generic(r, "TX", [0.0, 0.05, 0.10, 0.15, 0.20], Ys_train=range(2013, 2018))
    for nm, g in (("B_FP", gB), ("C_FX", gC), ("T_TX", gT)):
        g.to_csv(OUT / f"grid_{nm}.csv", index=False)
    cols = ["fear_min", "phys_max", "period", "trades", "winters", "mean", "lo", "hi", "hit", "worst", "winters_pos"]
    sel = {}
    for nm, g in (("A rv4>=f & P_tail<=p (train 2011-17)", gA), ("A (train 2013-17)", gA13.assign(signal="rv4")),
                  ("B FP>=f & P_tail<=p", gB), ("C FX>=f", gC), ("Control TX>=f (time only)", gT)):
        for h in ("exp", "4w"):
            f_, p_ = A.select(g, h)
            gg = g[(g.hold == h) & (g.fear_min == f_) & (g.phys_max == p_)]
            sel[(nm, h)] = (f_, p_)
            print(f"-- {nm}, hold {h}: selected f={f_}, p={p_}")
            print(gg[cols].round(2).to_string(index=False))
    print("\n-- full grid A, check-period mean by cell (rows fear_min, cols phys_max), hold exp / 4w")
    for h in ("exp", "4w"):
        for per in ("train", "check"):
            print(h, per)
            print(gA[(gA.hold == h) & (gA.period == per)].pivot(index="fear_min", columns="phys_max", values="mean").round(0))
    print("\n-- grid A robustness: share of check cells with mean>0, and effect of the physics filter")
    for h in ("exp", "4w"):
        c = gA[(gA.hold == h) & (gA.period == "check") & (gA.trades > 0)]
        nf = c[c.phys_max > 1].set_index("fear_min")["mean"]
        wf_ = c[c.phys_max < 1].groupby("fear_min")["mean"].mean()
        worst_nf = c[c.phys_max > 1].set_index("fear_min")["worst"]
        worst_f = c[c.phys_max < 1].groupby("fear_min")["worst"].min()
        print(h, f"positive {int((c['mean'] > 0).sum())}/{len(c)};", "no-filter mean:", nf.round(0).to_dict(),
              "filtered mean (avg over p):", wf_.round(0).to_dict(), "worst no filter:", worst_nf.to_dict(),
              "worst filtered:", worst_f.to_dict())
    for nm, g in (("C", gC), ("TX", gT)):
        print(f"-- {nm} full grid")
        print(g[cols + ["hold"]].round(0).to_string(index=False))

    H("5b. Benchmarks")
    bm = A.benchmarks(raw, sp)
    print(bm[["rule", "hold", "period", "trades", "winters", "mean", "lo", "hi", "hit", "worst", "winters_pos"]].round(0).to_string(index=False))
    bm.to_csv(OUT / "benchmarks.csv", index=False)
    bm13 = A.benchmarks(raw[raw.Y >= 2013], sp)
    print("-- same, from 2012-13 (common train window for B, C, TX)")
    print(bm13[bm13.period == "train"][["rule", "hold", "trades", "winters", "mean", "worst"]].round(0).to_string(index=False))
    n1 = A.nov1_to_expiry(sp)
    print("-- 1 Nov to expiry, per winter ($):", n1.set_index("Y").pnl.to_dict(), "mean", round(n1.pnl.mean()))

    H("5c. Per-winter P&L of the main rules (sum and count of weekly trades)")
    rules = {"every week": pd.Series(True, index=r.index), "S>0.15": r.S > 0.15,
             "A rv4>=0.02 & Ptail<=0.5": (r.rv4 >= 0.02) & (r.p_tail <= 0.5),
             "A rv4>=0.03": r.rv4 >= 0.03, "C FX>=0.10": r.FX >= 0.10, "TX>=0.10": r.TX >= 0.10}
    tabs = []
    for h in ("exp", "4w"):
        for nm, m in rules.items():
            t = A.trades(r, m, h)
            w = t.groupby("winter").pnl.agg(["count", "sum"]).rename(columns={"count": "n", "sum": "pnl"})
            w["rule"], w["hold"] = nm, h
            tabs.append(w.reset_index())
    pw = pd.concat(tabs)
    pw.to_csv(OUT / "per_winter.csv", index=False)
    for h in ("exp", "4w"):
        print("hold", h)
        print(pw[pw.hold == h].pivot(index="winter", columns="rule", values="pnl").round(0).fillna(0).to_string())

    H("5d. Robustness of the FX rule: physics definition, start year, threshold, costs")
    rows = []
    variants = {"composite (primary)": [("dev_now", -1), ("x_an", -1), ("p_tail", 1)],
                "p_tail only": [("p_tail", 1)], "dev_now only": [("dev_now", -1)],
                "x_an only": [("x_an", -1)], "storage-draw analog x_st": [("x_st", -1)],
                "p_tail_300": [("p_tail_300", 1)], "p_tail_800": [("p_tail_800", 1)]}
    for vn, cl in variants.items():
        for fy in (2013, 2014):
            q = A.wf_resid_signal(raw.assign(x_st=raw.x_st.fillna(raw.x_an)), A.phys_variant(cl), first_Y=fy, name="FXv")
            for thr in (0.05, 0.10, 0.15):
                for h in ("exp", "4w"):
                    for per, Ys in (("train", range(fy, 2018)), ("check", A.CHECK_Y)):
                        qq = q[q.Y.isin(Ys)]
                        s = A.summ(A.trades(qq, qq.FXv >= thr, h))
                        rows.append({"phys": vn, "first_Y": fy, "thr": thr, "hold": h, "period": per, **s})
    rb = pd.DataFrame(rows)
    rb.to_csv(OUT / "fx_robustness.csv", index=False)
    print(rb.pivot_table(index=["phys", "first_Y", "thr"], columns=["hold", "period"], values="mean").round(0).to_string())
    print("worst trade:")
    print(rb.pivot_table(index=["phys", "first_Y", "thr"], columns=["hold", "period"], values="worst").round(0).to_string())
    print("share of robustness cells with mean>0:",
          rb.groupby(["hold", "period"])["mean"].apply(lambda s: f"{(s > 0).sum()}/{s.notna().sum()}").to_dict())
    print("-- costs (FX>=0.10, 4w / exp, check):")
    for k in (0, 2, 4, 8):
        extra = (k - 4) * 2 * 2 * 10.0
        for h in ("exp", "4w"):
            t = A.trades(r[r.Y.isin(A.CHECK_Y)], r.FX >= 0.10, h)
            t2 = A.trades(r[r.Y.isin(A.CHECK_Y)], r.TX >= 0.10, h)
            print(f"  {k} ticks {h}: FX {t.pnl.mean() - extra:,.0f}  TX {t2.pnl.mean() - extra:,.0f}  S>0.15 "
                  f"{A.trades(r[r.Y.isin(A.CHECK_Y)], r.S > 0.15, h).pnl.mean() - extra:,.0f}")

    H("6. Jan 2014 (and Nov 2018 / Nov 2021): what fear and physics said")
    cols = ["obs", "S", "rv4", "fvol", "dev_now", "x_an", "p_tail", "p_tail_st", "p_tail_m2", "FP", "PP", "FX", "TX", "dS_4w", "dS_exp"]
    for w in ("2013-14", "2018-19", "2021-22"):
        print("--", w)
        print(r[r.winter == w][cols].round(3).to_string(index=False))
    # fear percentiles relative to all earlier winter weeks
    pct = [(r.rv4.iloc[:i] < r.rv4.iloc[i]).mean() if i > 0 else np.nan for i in range(len(r))]
    r["rv4_pct_hist"] = pct
    print("rv4 percentile vs all earlier weeks, 2013-14:", r[r.winter == "2013-14"].set_index("obs").rv4_pct_hist.round(2).to_dict())

    H("7. Power")
    for nm, m, h in (("C FX>=0.10 4w", r.FX >= 0.10, "4w"), ("S>0.15 4w", r.S > 0.15, "4w"),
                     ("every week exp", pd.Series(True, index=r.index), "exp")):
        t = A.trades(r, m, h)
        wm = t.groupby("winter").pnl.mean()
        sd = wm.std()
        print(f"{nm}: winters {len(wm)}, sd of winter-mean P&L ${sd:,.0f}; with 4 holdout winters the minimum "
              f"detectable mean (80% power, 5% two-sided) is ${2.8 * sd / np.sqrt(4):,.0f}; with 11 winters ${2.8 * sd / np.sqrt(11):,.0f}")
    print(f"P(at least one 2014-type winter in 4 / 5 winters at 1-in-11): {1 - (10 / 11) ** 4:.2f} / {1 - (10 / 11) ** 5:.2f}")

    H("8. Frozen coefficients for the holdout")
    fz = A.frozen_coefs(raw)
    json.dump(fz, open(OUT / "frozen_holdout.json", "w"), indent=1, default=float)
    print(json.dumps(fz, indent=1, default=float))
    # in-sample check of the frozen spec on dev (descriptive only)
    pf = A.add_indices(raw)
    pf["PPf"] = fz["const"] + fz["wte"] * pf.wte + fz["Phys"] * pf.Phys
    pf["FXf"] = pf.S - pf.PPf
    t = A.trades(pf, pf.FXf >= 0.10, "4w")
    print(f"frozen spec in-sample 2010-21 (not a test): {len(t)} trades, {t.winter.nunique()} winters, mean ${t.pnl.mean():,.0f}, worst ${t.pnl.min():,.0f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
