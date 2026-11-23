"""Run all v4_options steps:  python -m src.research.v4_options.run   (~2 min on 1 CPU)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.research.v4_options import analysis as A
from src.research.v4_options import data as D
from src.research.v4_options import figures as FG


def kept_cut_boot(t, vn, ys, nb=4000):
    """Winter-block bootstrap of kept + cut (mean / naked mean, 1 - worst DD / naked worst DD)."""
    q = t.loc[[y for y in ys if y in t.index and t.at[y, "complete"]]]
    out = []
    for _ in range(nb):
        b = q.iloc[A.RNG.integers(0, len(q), len(q))]
        if b.naked_pnl.mean() <= 0 or b.naked_dd.min() >= 0:
            continue
        out.append(b[f"{vn}_pnl"].mean() / b.naked_pnl.mean() + 1 - b[f"{vn}_dd"].min() / b.naked_dd.min())
    out = np.array(out)
    return {"p_gt1": float((out > 1).mean()), "lo": float(np.percentile(out, 5)), "med": float(np.median(out)),
            "hi": float(np.percentile(out, 95)), "draws_used": len(out)}


def main():
    t0 = time.time()
    W = A.winter_frames()
    opts = D.g4x_march()
    summ = {}
    # data coverage
    cov = opts.groupby("cy").agg(days=("date", "nunique"), strikes_per_day=("K", lambda k: len(k)))
    cov["strikes_per_day"] = cov.strikes_per_day / cov.days
    summ["g4x_coverage"] = cov.round(1).to_dict(orient="index")
    summ["parity_abs_err_median"] = float(opts.parity_err.abs().median())
    summ["parity_abs_err_p95"] = float(opts.parity_err.abs().quantile(.95))

    d1, s1, pw1 = A.step1(W)
    ep, pa, scale, lead = A.step2_episodes(W, opts)
    P, pred = A.step2_predict(W, opts)
    summ["step2_scale"] = scale
    summ["step2_corr_nv_S_weekly"] = float(P[["nv", "S"]].dropna().corr().iloc[0, 1])
    # frozen level -> implied-vol map (season days, winters 2013-2021), for the synthetic pre-2010 test
    x = pd.concat([s[s.index >= s.attrs["entry"]][["S", "nv_hj"]].assign(Y=Y) for Y, s in W.items()]).dropna()
    x = x[x.Y <= 2021]
    b, a = np.polyfit(x.S.clip(lower=0), x.nv_hj, 1)
    summ["nv_level_map"] = {"intercept": a, "slope": b, "corr": float(np.corrcoef(x.S.clip(lower=0), x.nv_hj)[0, 1]),
                            "n": len(x), "spec": "nv_hj = intercept + slope * max(S, 0), season days 2013-2021"}

    t3, paths, marks = A.step3(W, opts)
    s3 = A.step3_summary(t3)
    s3.to_csv(D.CACHE / "step3_summary.csv", index=False)
    t3.to_csv(D.CACHE / "step3_by_winter.csv")
    summ["step3_marks"] = marks
    summ["step3_kept_cut_boot"] = {vn: kept_cut_boot(t3, vn, D.YEARS) for vn in A.VARIANTS if vn != "naked"}
    sens = []
    for hs in (0.02, 0.04, 0.08, 0.15):
        for fs, lab in ((A.FUT_SIDE, "2tick"), (A.FUT_SIDE4, "4tick")):
            tt, _, _ = A.step3(W, opts, hs=hs, fut_side=fs)
            ss = A.step3_summary(tt)
            ss["opt_hs"], ss["fut_cost"] = hs, lab
            sens.append(ss)
    sens = pd.concat(sens)
    sens.to_csv(D.CACHE / "step3_sensitivity.csv", index=False)

    t4, phys = A.step4(W, opts)
    full = [y for y in range(2013, 2022)]
    q = t4.loc[full]
    s4 = {}
    for lab in ("abs0.50", "abs1.00", "rel+0.25", "rel+0.50", "rel+1.00"):
        s4[lab] = {"q_mean_2013_21": q[f"q_{lab}"].mean(), "q_mean_2015_21": q.loc[2015:2021, f"q_{lab}"].mean(),
                   "phys_expiry_freq_2013_21": q[f"p_{lab}"].mean(), "phys_hit_freq_2013_21": q[f"hit_{lab}"].mean(),
                   "reflection_hit_q_2013_21": 2 * q[f"q_{lab}"].mean(), "inside_share": q[f"inside_{lab}"].mean()}
    F0, ST, mx = phys.F0, phys.ST, phys["max"]
    s4["phys_2011_21"] = {"expiry_gt_F0+0.25": float((ST > F0 + .25).mean()), "expiry_gt_F0+0.5": float((ST > F0 + .5).mean()),
                          "hit_F0+0.5": float((mx > F0 + .5).mean()), "hit_F0+1.0": float((mx > F0 + 1).mean()),
                          "mean_F0_minus_ST": float((F0 - ST).mean()), "n": len(phys)}
    for dl in (0.0, 0.25, 0.5, 1.0):
        c0, pay = q[f"c0_{dl}"], q[f"pay_{dl}"]
        lo, hi = A.boot_mean((pay - c0).values * D.MULT)
        s4[f"call_{dl}"] = {"mean_price_$": c0.mean() * D.MULT, "mean_payoff_$": pay.mean() * D.MULT,
                            "payoff_over_price": pay.sum() / c0.sum(), "net_lo": lo, "net_hi": hi,
                            "winters_paid": int((pay > 0).sum())}
    summ["step4"] = s4
    summ["step4_2022_descriptive"] = t4.loc[2022, [c for c in t4.columns if c.startswith("q_")]].to_dict()

    s5 = A.step5(W)
    summ["step5"] = "skipped: news_channel file absent" if s5 is None else s5.round(4).to_dict(orient="index")

    FG.fig_vrp(pw1)
    FG.fig_episodes(pa, ep)
    FG.fig_protection(paths, s3)
    summ["runtime_s"] = time.time() - t0
    with open(D.CACHE / "summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summ[k] for k in ("step3_kept_cut_boot", "step4", "runtime_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
