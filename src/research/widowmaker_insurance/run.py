"""Run everything:  python -m src.research.widowmaker_insurance.run   (about 1-2 minutes)

Outputs: CSVs in data/processed/research_widowmaker_insurance/, figures in reports/agents/figures/,
tables printed to stdout (reports/agents/widowmaker_insurance.md quotes them).
"""
import itertools

import numpy as np
import pandas as pd

from . import analysis as A
from . import data as D
from . import payout as P
from . import figures as F

pd.set_option("display.width", 200)
OUT = D.CACHE
TRAIN = ["2016-17", "2017-18", "2018-19"]
CHECK = ["2019-20", "2020-21", "2021-22"]
XS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
YS = [0.05, 0.10, 0.15, 0.20, 0.30, 1.01]


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def main():
    sp = D.spreads()
    panel = D.weekly_panel(sp)
    m2 = D.load_m2()
    tab = A.payout_table()

    # ------------------------------------------------------------ 1. payout curve
    hdr("1. Payout curve")
    print(tab[["avg5", "n_prior", "end_march", "x", "S_exp", "S_exp5", "J_exp"]].round(3).to_string())
    cd, t = P.curves(tab, "S_exp", False, 1, B=500)
    p_full = cd[(2022, "wf")]
    print("Full-sample (2011-21) fit a, b, tau:", p_full["p0"].round(3),
          " residual sd $%.3f (inflated)" % p_full["e"].std())
    print("tau bootstrap 10/50/90:", np.quantile(p_full["ps"][:, 2], [.1, .5, .9]).round(0))
    for k in sorted(cd):
        if k[1] == "wf" and k[0] >= 2016:
            print(f"  WF fit for {D.winter_label(k[0])}: n={cd[k]['n']} a,b,tau={cd[k]['p0'].round(3)} "
                  f"resid sd={cd[k]['e'].std():.3f}")
    # variant comparison (OOS RMSE in $)
    rows = []
    for target, scaled, mp in itertools.product(["S_exp", "S_exp5"], [False, True], [1, 2]):
        c2, t2 = P.curves(tab, target, scaled, mp, B=20)
        e = P.oos_errors(t2, c2)
        for mode in ("wf", "lowo"):
            q = e[e["mode"] == mode]
            J = tab.J_exp.reindex(q.Y).values if scaled else 1.0
            r = {"target": target, "scaled": scaled, "min_prior": mp, "mode": mode, "n": len(q)}
            for c in ("pred_convex", "pred_linear", "pred_mean"):
                r[c.replace("pred_", "rmse_")] = np.sqrt(np.mean(((q[c] - q.y) * J) ** 2))
            rows.append(r)
    var = pd.DataFrame(rows)
    print(var.round(3).to_string())
    var.to_csv(OUT / "payout_variants.csv", index=False)
    e = P.oos_errors(t, cd)
    print("Point errors (primary):\n", e.round(3).sort_values(["mode", "Y"]).to_string())
    lowo = e[e["mode"] == "lowo"]
    print("LOWO 2014 point: pred %.3f actual %.3f" % tuple(lowo[lowo.Y == 2014][["pred_convex", "y"]].values[0]))

    # ------------------------------------------------------------ 2. physical distribution
    hdr("2. Actuarial value (walk-forward curve)")
    res = {}
    for centre, mode in itertools.product(["m2", "5y"], ["wf", "lowo"]):
        res[(centre, mode)] = A.actuarial(panel, m2, cd, centre, mode)
        res[(centre, mode)].to_csv(OUT / f"actuarial_{centre}_{mode}.csv", index=False)
    for key, r in res.items():
        g = r.groupby("winter").agg(n=("S", "size"), S=("S", "mean"), AV=("av", "mean"), q10=("q10", "mean"),
                                    q90=("q90", "mean"), prem=("prem", "mean"), p_above=("p_above_mkt", "mean"),
                                    p_large=("p_large", "mean"), S_exp=("S_exp", "first"))
        lo, hi = A.winter_boot(r, lambda d: d.prem.mean())
        print(f"\ncentre={key[0]} curve={key[1]}: mean premium ${r.prem.mean():.3f} (winter-boot CI {lo:.3f}..{hi:.3f}); "
              f"median premium ${r.prem.median():.3f}; share of weeks prem>0: {(r.prem > 0).mean():.2f}")
        print(g.round(3).to_string())
    # calibration of the physical distribution against realised expiry (5 winters)
    for key, r in res.items():
        d = r.dropna(subset=["S_exp"])
        pit_in = ((d.S_exp >= d.q10) & (d.S_exp <= d.q90)).mean()
        print(f"calibration {key}: realised expiry inside 10-90 band in {pit_in:.2f} of {len(d)} weeks; "
              f"mean(S_exp - AV) = {(d.S_exp - d.av).mean():+.3f}")

    # sensitivity: floor, scaled curve
    hdr("2b. Sensitivity of the mean premium")
    sens = []
    base = res[("5y", "wf")]
    for lab, kw, curves_ in [("floor 500", dict(floor=500), cd), ("floor 900", dict(floor=900), cd)]:
        for centre in ("m2", "5y"):
            r = A.actuarial(panel, m2, curves_, centre, "wf", **kw)
            sens.append({"variant": lab, "centre": centre, "mean_prem": r.prem.mean(), "mean_av": r.av.mean()})
    cds, _ = P.curves(tab, "S_exp", True, 1, B=300)
    for centre in ("m2", "5y"):
        r = A.actuarial(panel, m2, cds, centre, "wf", scaled=True)
        sens.append({"variant": "scaled by April (at t)", "centre": centre, "mean_prem": r.prem.mean(),
                     "mean_av": r.av.mean()})
    cd2, _ = P.curves(tab, "S_exp", False, 2, B=300)
    for centre in ("m2", "5y"):
        r = A.actuarial(panel, m2, cd2, centre, "wf")
        sens.append({"variant": "min 2 prior years", "centre": centre, "mean_prem": r.prem.mean(),
                     "mean_av": r.av.mean()})
    for centre in ("m2", "5y"):
        r = res[(centre, "wf")]
        sens.append({"variant": "primary", "centre": centre, "mean_prem": r.prem.mean(), "mean_av": r.av.mean()})
    print(pd.DataFrame(sens).round(3).to_string())

    # ------------------------------------------------------------ 3. premium: theta & test
    hdr("3. Premium theta and the premium test")
    for key in [("5y", "wf"), ("m2", "wf")]:
        th = A.theta_curve(res[key])
        print(f"theta, centre={key[0]}:\n", th.round(3).to_string())
        th.to_csv(OUT / f"theta_{key[0]}.csv", index=False)
    regs = []
    for key in [("5y", "wf"), ("m2", "wf"), ("5y", "lowo"), ("m2", "lowo")]:
        r = res[key].copy()
        for ycol in ("dS_exp", "dS_4w"):
            out = A.premium_regression(r, "prem", ycol)
            out.update(centre=key[0], curve=key[1])
            regs.append(out)
    r = res[("5y", "wf")].copy()
    r["S_only"] = r.S          # price-only benchmark: premium := market spread
    r["S_minus_mean_exp"] = r.S - tab.S_exp[tab.index < 2017].mean()
    for x in ("S_only",):
        for ycol in ("dS_exp", "dS_4w"):
            out = A.premium_regression(r, x, ycol)
            out.update(centre="price only", curve="-")
            regs.append(out)
    regs = pd.DataFrame(regs)
    print(regs[["centre", "curve", "x", "y", "n", "winters", "slope", "slope_hac_se", "slope_boot_lo", "slope_boot_hi",
                "intercept", "icpt_boot_lo", "icpt_boot_hi", "r2", "mean_dS", "mean_minus_prem"]].round(3).to_string())
    regs.to_csv(OUT / "premium_regressions.csv", index=False)
    # does the actuarial value add anything beyond the market spread?
    import statsmodels.api as sm
    for key in [("5y", "wf"), ("m2", "wf")]:
        d = res[key].dropna(subset=["S_exp"])
        X = sm.add_constant(np.column_stack([-d.S, d.av]))
        f = sm.OLS(d.dS_exp, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
        lo, hi = A.winter_boot(d, lambda z: np.polyfit(z.av, z.S_exp, 1)[0])
        print(f"{key}: sd(AV) {d.av.std():.3f} vs sd(S) {d.S.std():.3f}; corr(S_exp, AV) {d.S_exp.corr(d.av):.2f}, "
              f"corr(S_exp, S) {d.S_exp.corr(d.S):.2f}; dS_exp ~ -S + AV: coefs {f.params.round(3).tolist()} "
              f"HAC se {f.bse.round(3).tolist()}; slope S_exp on AV {np.polyfit(d.av, d.S_exp, 1)[0]:.2f} "
              f"(winter-boot {lo:.2f}..{hi:.2f})")
    # same regression on 2010-21 price-only panel (11 winters)
    pp = panel.dropna(subset=["S_exp"]).copy()
    pp["dS_exp"] = pp.S_exp - pp.S
    pp["dS_4w"] = pp.S_4w - pp.S
    pp["S_only"] = pp.S
    for ycol in ("dS_exp", "dS_4w"):
        o = A.premium_regression(pp, "S_only", ycol)
        print(f"price-only 2010-21 {ycol}: slope {o['slope']:.3f} boot CI {o['slope_boot_lo']:.3f}..{o['slope_boot_hi']:.3f}"
              f" icpt {o['intercept']:.3f} R2 {o['r2']:.3f} n {o['n']} winters {o['winters']}")

    # ------------------------------------------------------------ 4. conditional rule
    hdr("4. Conditional rule grid")
    allgrid = []
    for key in [("5y", "wf"), ("m2", "wf")]:
        r = res[key]
        for hold in ("exp", "4w"):
            for X, Yc in itertools.product(XS, YS):
                mask = (r.prem > X) & (r.p_large < Yc)
                tt = A.trades(r, mask, hold)
                for period, ws in (("train", TRAIN), ("check", CHECK)):
                    s = A.summarize(tt[tt.winter.isin(ws)])
                    s.update(centre=key[0], hold=hold, X=X, Y=Yc, period=period)
                    allgrid.append(s)
    grid = pd.DataFrame(allgrid)
    grid.to_csv(OUT / "rule_grid.csv", index=False)
    chosen = {}
    for (centre, hold), g in grid.groupby(["centre", "hold"]):
        tr = g[(g.period == "train") & (g.trades >= 5)]
        best = tr.sort_values(["mean_net", "X"], ascending=[False, True]).iloc[0]
        chosen[(centre, hold)] = (best.X, best.Y)
        ck = g[(g.period == "check") & (g.X == best.X) & (g.Y == best.Y)].iloc[0]
        print(f"\ncentre={centre} hold={hold}: chosen on 2016-19 X={best.X} Y={best.Y}: train {best.trades:.0f} trades, "
              f"mean net ${best.mean_net:,.0f}; CHECK {ck.trades:.0f} trades in {ck.winters:.0f} winters, mean net "
              f"${ck.mean_net:,.0f} (CI {ck.lo:,.0f}..{ck.hi:,.0f}), hit {ck.hit:.2f}, worst ${ck.worst:,.0f}")
        for period in ("train", "check"):
            pv = g[g.period == period].pivot(index="X", columns="Y", values="mean_net")
            nt = g[g.period == period].pivot(index="X", columns="Y", values="trades")
            print(f"  {period} mean net $/trade (rows X = premium threshold, cols Y = max P(large)):")
            print((pv.round(0).astype("Int64").astype(str) + " (" + nt.astype(int).astype(str) + ")").to_string())

    # benchmarks
    hdr("4b. Benchmarks (same weeks, 5y-centred panel)")
    r = res[("5y", "wf")]
    first = r.groupby("winter").obs.transform("min") == r.obs
    spike = (r.S - r.S_prev1w) > 0.05
    bench = []
    for lab, mask in [("always short, first Nov week", first), ("always short, every week", pd.Series(True, index=r.index)),
                      ("short after +$0.05 weekly spike", spike)]:
        for hold in ("exp", "4w"):
            tt = A.trades(r, mask, hold)
            for period, ws in (("train", TRAIN), ("check", CHECK), ("all", TRAIN + CHECK)):
                s = A.summarize(tt[tt.winter.isin(ws)])
                s.update(rule=lab, hold=hold, period=period)
                bench.append(s)
    for (centre, hold), (X, Yc) in chosen.items():
        rr = res[(centre, "wf")]
        tt = A.trades(rr, (rr.prem > X) & (rr.p_large < Yc), hold)
        for period, ws in (("train", TRAIN), ("check", CHECK), ("all", TRAIN + CHECK)):
            s = A.summarize(tt[tt.winter.isin(ws)])
            s.update(rule=f"chosen rule [{centre}] X={X} Y={Yc}", hold=hold, period=period)
            bench.append(s)
        if centre == "5y":
            tt["hold"] = hold
            mae = A.mae_path(sp, tt) if len(tt) else []
            print(f"chosen rule 5y hold={hold}: per-winter net totals",
                  tt.groupby("winter").net.agg(["size", "sum"]).round(0).to_dict("index"),
                  f"worst MTM ${np.min(mae):,.0f}" if len(mae) else "")
    bench = pd.DataFrame(bench)
    print(bench[["rule", "hold", "period", "trades", "winters", "mean_net", "lo", "hi", "hit", "worst", "total"]]
          .round(2).to_string())
    bench.to_csv(OUT / "rule_benchmarks.csv", index=False)
    # rule robustness summary: share of grid cells with positive check mean
    for (centre, hold), g in grid.groupby(["centre", "hold"]):
        ck = g[(g.period == "check") & (g.trades > 0)]
        print(f"grid {centre}/{hold}: {(ck.mean_net > 0).sum()} of {len(ck)} non-empty check cells positive; "
              f"median cell ${ck.mean_net.median():,.0f}")
    # per-winter P&L of trades in each premium tercile (does the premium sort?)
    r = res[("5y", "wf")].dropna(subset=["S_exp"]).copy()
    r["prem_bin"] = pd.cut(r.prem, [-9, 0.05, 0.15, 9], labels=["<=0.05", "0.05-0.15", ">0.15"])
    print("\nRealised short-to-expiry net $ by premium bin (5y-centred):")
    print(r.assign(net=-(r.S_exp - r.S) * D.MULT - D.ROUND_TRIP).groupby("prem_bin", observed=True)
          .agg(n=("net", "size"), winters=("winter", "nunique"), mean=("net", "mean"), worst=("net", "min")).round(0))

    # ------------------------------------------------------------ 5. price-only sanity check
    hdr("5. Price-only, 2010-21")
    po = A.price_only(sp)
    print(po.drop(columns=["date_max"]).round(3).to_string())
    print(f"Nov-avg spread > expiry spread in {(po.nov_minus_exp > 0).sum()} of {len(po)} winters; "
          f"mean gap ${po.nov_minus_exp.mean():.3f}, median ${po.nov_minus_exp.median():.3f}")
    lo, hi = A.winter_boot(po.reset_index(), lambda d: d.net_short_nov1.mean())
    print(f"Short from first Nov trading day to expiry: mean net ${po.net_short_nov1.mean():,.0f}/spread "
          f"(winter-boot CI {lo:,.0f}..{hi:,.0f}); positive in {(po.net_short_nov1 > 0).sum()} of {len(po)}; "
          f"sum ${po.net_short_nov1.sum():,.0f}; ex-2014 mean ${po.drop(2014).net_short_nov1.mean():,.0f}")
    for a, b in ((2011, 2016), (2017, 2021)):
        q = po.loc[a:b]
        print(f"  {a}-{b}: mean net ${q.net_short_nov1.mean():,.0f}, mean Nov-avg spread ${q.S_novavg.mean():.3f}, "
              f"mean expiry ${q.S_exp.mean():.3f}")
    po.to_csv(OUT / "price_only.csv")
    # price-only analogue of the rule (premium ~ spread + a near-constant): short when S > X, 2010-21
    pp = panel.copy()
    rows = []
    for X in (0.05, 0.10, 0.15, 0.20, 0.30):
        for hold in ("exp", "4w"):
            tt = A.trades(pp, pp.S > X, hold)
            tt["hold"] = hold
            for lab, ws in (("2010-16", [D.winter_label(y) for y in range(2011, 2017)]),
                            ("2016-21", TRAIN + CHECK), ("all", None)):
                q = tt if ws is None else tt[tt.winter.isin(ws)]
                s = A.summarize(q)
                s.update(X=X, hold=hold, sample=lab,
                         worst_mtm=A.mae_path(sp, q).min() if len(q) else np.nan,
                         net_2014=q[q.winter == "2013-14"].net.sum())
                rows.append(s)
    pr = pd.DataFrame(rows)
    print("\nPrice-only rule 'short H-J when S > X' (weekly decision points, 2010-21):")
    print(pr[["X", "hold", "sample", "trades", "winters", "mean_net", "lo", "hi", "worst", "worst_mtm", "net_2014"]]
          .round(0).to_string())
    pr.to_csv(OUT / "price_only_rule.csv", index=False)

    # ------------------------------------------------------------ figures
    F.fig_bands(res[("5y", "wf")], res[("m2", "wf")])
    F.fig_payout(t, cd)
    F.fig_theta(A.theta_curve(res[("5y", "wf")]), A.theta_curve(res[("m2", "wf")]), po)
    print("\nfigures written")


if __name__ == "__main__":
    main()
