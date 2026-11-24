"""v5_implied_storage: full pipeline (after step1_rnd has cached the daily RND fits).
python -m src.research.v5_implied_storage.step1_rnd   (~14 min, cached per winter)
python -m src.research.v5_implied_storage.run         (~1 min)
"""
import json

import numpy as np
import pandas as pd

from src.research.v5_implied_storage import analysis as A, data as D, defaults as DF, step1_rnd as S1, \
    trade as TR

OUT = {}


def wb(v, g):
    lo, hi = TR.winter_boot(v, g)
    return [round(float(np.mean(v)), 3), round(lo, 3), round(hi, 3)]


def main():
    rnd = S1.run()
    # ---------------------------------------------------------------- default settles + fit quality
    fl = DF.flag(rnd)
    fl.to_csv(D.CACHE / "defaults_flag.csv", index=False)
    g = rnd.drop(columns=["theta", "np_tail", "quotes"]).merge(fl, on=["date", "Y"])
    g["vol0"] = g.vol_day.fillna(0) == 0
    gof = g.groupby("Y").agg(days=("nq", "size"), quotes_med=("nq", "median"), comps=("ncomp", "median"),
                             rmse_med=("rmse", "median"), rmse_p90=("rmse", lambda v: v.quantile(.9)),
                             rel_rmse=("rel_rmse", "median"), bfly_viol=("bfly_viol", "sum"),
                             bfly_n=("bfly_n", "sum"), default_days=("default", "sum"),
                             pinned_share=("pin_share", "mean"), zero_vol_share=("vol0", "mean"))
    gof.to_csv(D.CACHE / "gof_by_winter.csv")
    OUT["gof"] = gof.round(4).reset_index().to_dict("records")
    OUT["defaults_total"] = int(fl.default.sum())
    OUT["days_total"] = int(len(fl))

    # ---------------------------------------------------------------- panel
    df, maps, maps_ltd = A.build_panel(rnd)
    dflt = set(map(tuple, fl.loc[fl.default, ["date", "Y"]].values))
    df["default"] = [(o, y) in dflt for o, y in zip(df.obs, df.Y)]
    df = df[~df.default].copy()
    OUT["maps"] = {int(Y): {k: round(float(v), 4) for k, v in m.items()} for Y, m in maps.items()}
    OUT["thresholds_k"] = {int(Y): {x0: round(float(A.y_of(m, x0)), 3) for x0 in A.X0} for Y, m in maps.items()}
    # robustness of the tail: mixture vs model-free BL where both exist
    for x0 in (200, 400):
        z = df[[f"Q_{x0}", f"Qnp_{x0}"]].dropna()
        OUT[f"mix_vs_np_{x0}"] = {"n": len(z), "mean_abs_diff": round(float((z.iloc[:, 0] - z.iloc[:, 1]).abs().mean()), 3),
                                  "corr": round(float(z.corr().iloc[0, 1]), 3)}
    df["lead"] = pd.cut(df.wte, [0, 4, 8, 12, 30], labels=["<4w", "4-8w", "8-12w", ">12w"])
    m = df[df.Y >= 2015].copy()   # walk-forward map exists
    # ---------------------------------------------------------------- premium tables
    prem = []
    for x0 in (200, 400, 600, 800):
        for P in ("pl_rc", "pl_raw", "m2_rc", "m2_raw"):
            z = m[[f"Q_{x0}", f"P_{P}_{x0}", f"Psp_{P}_{x0}", "Y", "window"]].dropna()
            if z.empty:
                continue
            for w in ("choose", "check", "all"):
                zz = z if w == "all" else z[z.window == w]
                if len(zz) == 0:
                    continue
                prem.append({"x0": -x0, "phys": P, "window": w, "n": len(zz), "winters": zz.Y.nunique(),
                             "Q": round(zz[f"Q_{x0}"].mean(), 3), "P": round(zz[f"P_{P}_{x0}"].mean(), 3),
                             "gap_storage": wb(zz[f"Q_{x0}"] - zz[f"P_{P}_{x0}"], zz.Y),
                             "P_spread": round(zz[f"Psp_{P}_{x0}"].mean(), 3),
                             "gap_spread": wb(zz[f"Q_{x0}"] - zz[f"Psp_{P}_{x0}"], zz.Y),
                             "share_Q_gt_P": round(float((zz[f"Q_{x0}"] > zz[f"Psp_{P}_{x0}"]).mean()), 2)})
    prem = pd.DataFrame(prem)
    prem.to_csv(D.CACHE / "premium_table.csv", index=False)
    OUT["premium"] = prem.to_dict("records")
    byw = m.groupby("Y").agg(n=("Q_400", "size"), F=("F", "mean"), Q400=("Q_400", "mean"),
                             P400_pl=("P_pl_rc_400", "mean"), Psp400_pl=("Psp_pl_rc_400", "mean"),
                             P400_m2=("P_m2_rc_400", "mean"), Psp400_m2=("Psp_m2_rc_400", "mean"),
                             Q200=("Q_200", "mean"), Q800=("Q_800", "mean"), P800_pl=("P_pl_rc_800", "mean"),
                             xQ50=("xq_Q_50", "mean"), xQ10=("xq_Q_10", "mean"), xpl10=("xq_pl_rc_10", "mean"),
                             xm2_10=("xq_m2_rc_10", "mean"), xbench=("x_bench5", "mean"), x_real=("x_real", "first"),
                             S_T=("S_T", "first"))
    byw.to_csv(D.CACHE / "by_winter.csv")
    OUT["by_winter"] = byw.round(3).reset_index().to_dict("records")
    bylead = m.groupby("lead", observed=True).apply(lambda z: pd.Series({
        "n": len(z), "Q400": z.Q_400.mean(), "Psp400_pl": z.Psp_pl_rc_400.mean(),
        "gap_sp": wb(z.Q_400 - z.Psp_pl_rc_400, z.Y), "Q800": z.Q_800.mean(), "Psp800_pl": z.Psp_pl_rc_800.mean()}))
    OUT["by_lead"] = bylead.reset_index().astype(str).to_dict("records")

    # mass of Q in the flat (non-invertible) region: S_T below y(+300) (ample storage) or below a
    from src.research.v5_implied_storage import rnd as RR
    rq = rnd.set_index(["date", "Y"])
    flat, below_a = [], []
    for _, r in m.iterrows():
        th = rq.loc[(r.obs, r.Y)].theta
        mp = maps[r.Y]
        flat.append(float(RR.mix_cdf(th, [A.y_of(mp, 300)])[0]))
        below_a.append(float(RR.mix_cdf(th, [mp["a"]])[0]))
    m["Q_flat300"], m["Q_below_a"] = flat, below_a
    OUT["Q_mass_flat"] = {"mean_P_Q(S_T<y(+300))": round(float(np.mean(flat)), 3),
                          "mean_P_Q(S_T<a)": round(float(np.mean(below_a)), 3),
                          "by_Y": m.groupby("Y")[["Q_flat300", "Q_below_a"]].mean().round(3).to_dict()}
    for w in ("choose", "check"):
        z = m[m.window == w]
        OUT[f"median_gap_bcf_{w}"] = {"xQ50_minus_5yr_path": wb(z.xq_Q_50 - z.x_bench5, z.Y),
                                      "xQ10_minus_pl_rc_10": wb(z.xq_Q_10 - z.xq_pl_rc_10, z.Y)}
        z2 = z.dropna(subset=["xq_m2_rc_10"])
        if z2.Y.nunique() > 1:
            OUT[f"median_gap_bcf_{w}"]["xQ10_minus_m2_rc_10"] = wb(z2.xq_Q_10 - z2.xq_m2_rc_10, z2.Y)
    m.to_csv(D.CACHE / "panel_weekly_final.csv", index=False)

    # ---------------------------------------------------------------- calibration
    c = m[m.x_real.notna()].copy()
    cal = []
    for x0 in (200, 400, 600, 800):
        ev = (c.x_real < -x0).astype(float)
        for nm in ["Q", "P_pl_rc", "P_pl_raw", "P_m2_rc", "P_m2_raw", "Psp_pl_rc", "clim"]:
            if nm == "Q":
                p = c[f"Q_{x0}"]
            elif nm == "clim":   # walk-forward base rate of the event over winters before Y (2011..Y-1)
                xs = D.avg5_table().x
                p = c.Y.map(lambda Y: float((xs[(xs.index < Y)] < -x0).mean()))
            else:
                p = c[f"{nm}_{x0}"]
            ok = p.notna()
            for w in ("choose", "check", "all"):
                sel = ok & ((c.window == w) | (w == "all"))
                if sel.sum() == 0:
                    continue
                br = (p[sel] - ev[sel]) ** 2
                # winter-weighted Brier
                bw = br.groupby(c.Y[sel]).mean().mean()
                cal.append({"x0": -x0, "model": nm, "window": w, "n": int(sel.sum()),
                            "winters": int(c.Y[sel].nunique()), "events_winters": int(ev[sel].groupby(c.Y[sel]).max().sum()),
                            "brier_w": round(float(bw), 3), "mean_p": round(float(p[sel].mean()), 3)})
    cal = pd.DataFrame(cal)
    cal.to_csv(D.CACHE / "calibration.csv", index=False)
    OUT["calibration"] = cal.to_dict("records")
    allq = df[df.S_T.notna() & (df.Y <= 2021)]
    pit = {"pit_Q_mean": round(allq.pit_Q.mean(), 3),
           "pit_Q_by_winter": allq.groupby("Y").pit_Q.mean().round(2).to_dict(),
           "pit_Psp_mean": round(c.pit_P_sp_pl_rc.mean(), 3),
           "pit_Psp_by_winter": c.groupby("Y").pit_P_sp_pl_rc.mean().round(2).to_dict(),
           "pitx_pl_rc_by_winter": c.groupby("Y").pitx_pl_rc.mean().round(2).to_dict(),
           "pitx_m2_rc_by_winter": c.groupby("Y").pitx_m2_rc.mean().round(2).to_dict()}
    # implied-x PIT: P_Q(x < x_real) via the map, where defined
    OUT["pit"] = pit

    # ---------------------------------------------------------------- trades
    tr = TR.option_trades(df[df.Y >= 2013], rnd)
    ft = TR.futures_trades(df[df.Y >= 2013])
    tr = tr.merge(ft, on=["obs", "Y"]).merge(
        df[["obs", "Y", "Q_400", "P_pl_rc_400", "Psp_pl_rc_400", "P_m2_rc_400", "wte"]], on=["obs", "Y"])
    tr["gap"] = tr.Q_400 - tr.P_pl_rc_400
    tr["gap_sp"] = tr.Q_400 - tr.Psp_pl_rc_400
    for cst in (0.03, 0.05):
        t2 = TR.option_trades(df[df.Y >= 2013], rnd, cost=cst)
        tr[f"pnl_cs_c{int(cst * 100)}"] = t2.pnl_cs.values
        tr[f"pnl_sc_c{int(cst * 100)}"] = t2.pnl_sc.values
    tr.to_csv(D.CACHE / "trades_weekly.csv", index=False)
    OUT["quoted_share"] = {"K1": round(tr.quoted1.mean(), 3), "K2": round(tr.quoted2.mean(), 3),
                           "K1_by_Y": tr.groupby("Y").quoted1.mean().round(2).to_dict()}
    # per-winter (all weekly entries, 1 lot each) and worst
    pw = tr.groupby("Y").agg(n=("pnl_cs", "size"), cs=("pnl_cs", "mean"), cs_worst=("pnl_cs", "min"),
                             cs_dd=("dd_cs", "min"), sc=("pnl_sc", "mean"), sc_worst=("pnl_sc", "min"),
                             sc_dd=("dd_sc", "min"), fut=("pnl_fut", "mean"), fut_worst=("pnl_fut", "min"),
                             fut_dd=("dd_fut", "min"), gap=("gap", "mean"), gap_sp=("gap_sp", "mean"))
    pw.to_csv(D.CACHE / "trades_by_winter.csv")
    OUT["trades_by_winter"] = pw.round(0).reset_index().to_dict("records")
    # signal rules: gap > theta, choose on obs <= 2016 (map exists: Y >= 2015), check on 2017-2021
    rules = []
    base = tr[tr.Y >= 2015]
    for sig in ("gap", "gap_sp"):
        for th in (-1.0, 0.0, 0.1, 0.2, 0.3):
            for inst in ("cs", "sc"):
                for w in ("choose", "check"):
                    z = base[(base.window == w) & (base[sig] > th) & (base.Y <= (2022 if w == "check" else 2017))]
                    zz = z[z.Y <= 2021]
                    rules.append({"signal": sig, "theta": th, "inst": inst, "window": w, "n": len(zz),
                                  "winters": zz.Y.nunique(),
                                  "mean": round(zz[f"pnl_{inst}"].mean(), 0) if len(zz) else np.nan,
                                  "ci": wb(zz[f"pnl_{inst}"].values, zz.Y.values) if zz.Y.nunique() > 1 else None,
                                  "worst": round(zz[f"pnl_{inst}"].min(), 0) if len(zz) else np.nan,
                                  "worst_dd": round(zz[f"dd_{inst}"].min(), 0) if len(zz) else np.nan,
                                  "mean_c3": round(zz[f"pnl_{inst}_c3"].mean(), 0) if len(zz) else np.nan,
                                  "mean_c5": round(zz[f"pnl_{inst}_c5"].mean(), 0) if len(zz) else np.nan,
                                  "n_2022": int((z.Y == 2022).sum()),
                                  "mean_2022": round(z[z.Y == 2022][f"pnl_{inst}"].mean(), 0) if (z.Y == 2022).any() else None})
    rules = pd.DataFrame(rules)
    rules.to_csv(D.CACHE / "rules.csv", index=False)
    OUT["rules"] = rules.to_dict("records")
    # once-per-winter at the first decision date (v4 comparable), winters 2013..2021 (+2022 shown)
    first = tr.sort_values("obs").groupby("Y").head(1).copy()
    first["half_fut"] = first.pnl_fut / 2
    first["half_dd"] = first.dd_fut / 2
    first.to_csv(D.CACHE / "trades_first_week.csv", index=False)
    f21 = first[first.Y <= 2021]
    once = {}
    for nm, pc, dc in (("naked_fut", "pnl_fut", "dd_fut"), ("half_fut", "half_fut", "half_dd"),
                       ("short_cs", "pnl_cs", "dd_cs"), ("short_sc", "pnl_sc", "dd_sc")):
        once[nm] = {"mean": round(f21[pc].mean(), 0), "ci": wb(f21[pc].values, f21.Y.values),
                    "worst_winter": round(f21[pc].min(), 0), "worst_dd": round(f21[dc].min(), 0),
                    "calmar": round(f21[pc].mean() / abs(f21[dc].min()), 3),
                    "y2014": round(float(f21.loc[f21.Y == 2014, pc].iloc[0]), 0),
                    "y2014_dd": round(float(f21.loc[f21.Y == 2014, dc].iloc[0]), 0),
                    "y2019_dd": round(float(f21.loc[f21.Y == 2019, dc].iloc[0]), 0),
                    "y2022": round(float(first.loc[first.Y == 2022, pc].iloc[0]), 0)}
    OUT["once_per_winter"] = once
    OUT["first_week_detail"] = first[["Y", "obs", "F", "K1", "K2", "c1", "c2", "quoted1", "quoted2", "S_T", "pnl_cs",
                                      "dd_cs", "pnl_sc", "dd_sc", "pnl_fut", "dd_fut", "gap", "gap_sp"]].round(3).astype(str).to_dict("records")
    # all-weekly unconditional per window
    for w in ("choose", "check"):
        z = tr[(tr.window == w) & (tr.Y <= 2021)]
        OUT[f"uncond_{w}"] = {i: {"n": len(z), "mean": round(z[f"pnl_{i}"].mean(), 0), "ci": wb(z[f"pnl_{i}"].values, z.Y.values),
                                  "worst": round(z[f"pnl_{i}"].min(), 0), "worst_dd": round(z[f"dd_{i}"].min(), 0)}
                              for i in ("cs", "sc", "fut")}
    # does the gap predict option P&L beyond the premium level? (weekly, Y >= 2015, <= 2021)
    z = tr[(tr.Y >= 2015) & (tr.Y <= 2021)]
    import statsmodels.api as sm
    for x in ("gap", "gap_sp", "Q_400"):
        X = sm.add_constant(z[[x]])
        r = sm.OLS(z.pnl_cs, X).fit(cov_type="HAC", cov_kwds={"maxlags": 13})
        OUT[f"reg_pnl_cs_on_{x}"] = {"b": round(r.params[x], 0), "t": round(r.tvalues[x], 2), "r2": round(r.rsquared, 3)}
    X = sm.add_constant(z[["gap", "F"]])
    r = sm.OLS(z.pnl_cs, X).fit(cov_type="HAC", cov_kwds={"maxlags": 13})
    OUT["reg_pnl_cs_on_gap_F"] = {"b_gap": round(r.params["gap"], 0), "t_gap": round(r.tvalues["gap"], 2),
                                  "b_F": round(r.params["F"], 0), "t_F": round(r.tvalues["F"], 2)}
    with open(D.CACHE / "summary.json", "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    return df, tr, OUT


if __name__ == "__main__":
    df, tr, OUT = main()
    print(json.dumps(OUT, indent=1, default=str))
