"""Q2: market-implied end-of-March storage vs Model 2's scenario distribution.

Steps
  1. Model 2 percentiles per winter origin (simulate.py, cached).
  2. Map scarcity -> spread, walk-forward by winter:  y = a + b * exp(-c * x / 1000)
       y = March-April spread ($/MMBtu) or spread / April price ("pct")
       x = x_real (realised end-March minus prior-years average; ex-post) or
           x_info (storage now + average remaining draw minus the same average; ex-ante)
     Fit on winters strictly before the test winter (needs >= 2 earlier winters).
  3. Invert: x_hat = -(1000/c) ln((y - a)/b). Delta-method standard error of x_hat:
       se = sigma * 1000 / (c * (y - a)),  sigma = training residual s.d.
     Informative only if y > a and se <= SE_MAX (250 Bcf, fixed before looking at results).
  4. Locate L_hat = avg5 + x_hat in Model 2's distribution -> percentile u.
     Gap G = L_hat - Model 2 median (Bcf). Study level/stability, prediction of dS, convergence.

Run: PYTHONPATH=. python -m src.research.implied_storage.analysis   (~1 min after simulate.py)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

from src.research.implied_storage import data as D
from src.research.implied_storage.simulate import run as m2_run, PCTS

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/processed/research_implied_storage"
FIG = ROOT / "reports/agents/figures"
SE_MAX = 250.0
N_BOOT = 2000
# floor of the spread when storage is ample: about minus one month of carry. Bounded so the
# curve cannot degenerate into a near-straight line with a far-negative asymptote.
A_LO, A_HI = -0.2, 0.1
MAIN = "x_info"  # ex-ante mapping is the primary spec (see report)
warnings.filterwarnings("ignore")


def f(x, a, b, c):
    return a + b * np.exp(-c * x / 1000.0)


def fit(x, y):
    p, _ = curve_fit(f, x, y, p0=[0.0, 0.1, 1.5], bounds=([A_LO, 1e-5, 0.05], [A_HI, 20, 15]), maxfev=20000)
    sig = float(np.std(y - f(x, *p), ddof=3))
    return p, sig


def invert(y, p, sig):
    a, b, c = p
    with np.errstate(invalid="ignore", divide="ignore"):
        x = -(1000.0 / c) * np.log((y - a) / b)
        se = np.where(y > a, sig * 1000.0 / (c * (y - a)), np.inf)
    return np.where(y > a, x, np.nan), se


def walk_forward(df, xcol, units, se_max=SE_MAX):
    """Implied x for each row, using a mapping fitted only on earlier winters."""
    yv = df["spread"] if units == "usd" else df["spread"] / df["april"]
    df = df.assign(y=yv)
    out = []
    for Y in sorted(df.Y.unique()):
        tr = df[(df.Y < Y) & df[xcol].notna() & df["x_real"].notna()]
        if tr.Y.nunique() < 2:
            continue
        p, sig = fit(tr[xcol].values, tr["y"].values)
        te = df[df.Y == Y].copy()
        te["x_hat"], te["se"] = invert(te["y"].values, p, sig)
        te["informative"] = te["se"] <= se_max
        te["a"], te["b"], te["c"], te["sigma"] = *p, sig
        te["L_hat"] = te["avg5"] + te["x_hat"].clip(lower=500 - te["avg5"])  # physical floor 500 Bcf
        out.append(te)
    return pd.concat(out)


def pct_in(levels_q, L):
    return float(np.interp(L, levels_q, PCTS / 100.0, left=0.005, right=0.995))


def winter_boot(df, stat, n=N_BOOT, seed=1):
    rng = np.random.default_rng(seed)
    winters = df.winter.unique()
    if len(winters) == 0:
        return np.nan, np.nan
    df = df.reset_index(drop=True)
    groups = {w: np.flatnonzero(df.winter.values == w) for w in winters}
    vals = []
    for _ in range(n):
        pick = rng.choice(winters, len(winters))
        d = df.iloc[np.concatenate([groups[w] for w in pick])]
        try:
            vals.append(stat(d))
        except Exception:
            pass
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def ols_slope(d, ycol, xcol):
    d = d[[ycol, xcol]].dropna()
    X = np.column_stack([np.ones(len(d)), d[xcol].values])
    return np.linalg.lstsq(X, d[ycol].values, rcond=None)[0][1]


def hac(d, ycol, xcol, lags):
    d = d[[ycol, xcol]].dropna()
    r = sm.OLS(d[ycol], sm.add_constant(d[xcol])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r.params[xcol], r.tvalues[xcol], len(d), r.rsquared


def main():
    panel = D.build()
    m2 = m2_run()
    m2 = m2[m2.origin <= D.CUTOFF]
    qcols = [f"p{p}" for p in PCTS]
    m2q = {r.origin: np.array([getattr(r, c) for c in qcols]) for r in m2.itertuples()}
    log = []
    import atexit
    atexit.register(lambda: print("\n\n".join(log)))

    # ---------- step 2: choose units on pre-Model-2 test winters (2013-14..2015-16) ----------
    sel = []
    for xcol in ("x_real", "x_info"):
        for units in ("usd", "pct"):
            wf = walk_forward(panel, xcol, units)
            pre = wf[wf.Y <= 2016]
            inf = pre[pre.informative]
            err = (inf.x_hat - inf.x_real).abs()
            sel.append({"target": xcol, "units": units, "rows": len(pre), "informative": len(inf),
                        "MAE vs realised (Bcf)": err.mean(),
                        "MAE 5y-path on same rows": (inf.x_info - inf.x_real).abs().mean()})
    sel = pd.DataFrame(sel)
    log.append("## Unit choice on pre-Model-2 test winters 2013-14..2015-16\n" + sel.round(1).to_string(index=False))
    units = sel[sel.target == MAIN].sort_values("MAE vs realised (Bcf)").iloc[0]["units"]
    log.append(f"chosen units: {units}")

    # full-sample (<=2021) fit, for the figure and description only
    full = panel[panel.x_real.notna()].copy()
    full["y"] = full.spread if units == "usd" else full.spread / full.april
    fits = {xc: fit(full.dropna(subset=[xc])[xc].values, full.dropna(subset=[xc])["y"].values)
            for xc in ("x_real", "x_info")}
    for xc, (p, s) in fits.items():
        flat = -(1000 / p[2]) * np.log(s * 1000 / (p[2] * SE_MAX) / p[1])  # x where se = SE_MAX
        log.append(f"full-sample fit {xc}: a={p[0]:.4f} b={p[1]:.4f} c={p[2]:.3f} sigma={s:.4f}; "
                   f"invertible (se<={SE_MAX:.0f}) for x below {flat:.0f} Bcf")

    results = {}
    for xcol in ("x_real", "x_info"):
        for se_max in (150.0, SE_MAX, 400.0):
            wf = walk_forward(panel, xcol, units, se_max)
            d = wf[wf.Y >= 2017].merge(m2, left_on="week", right_on="origin", how="inner")
            d["u"] = [pct_in(m2q[o], L) if np.isfinite(L) else np.nan for o, L in zip(d.origin, d.L_hat)]
            d["G"] = d.L_hat - d["p50"]
            # Jensen benchmark: certainty-equivalent level if the market priced E_M2[f(X)]
            ce = []
            for r in d.itertuples():
                xs = m2q[r.origin] - r.avg5
                ey = np.mean(f(xs, r.a, r.b, r.c))
                ce.append(r.avg5 - (1000 / r.c) * np.log((ey - r.a) / r.b))
            d["L_ce"] = ce
            d["u_ce"] = [pct_in(m2q[o], L) for o, L in zip(d.origin, d.L_ce)]
            results[(xcol, se_max)] = d
    main_d = results[(MAIN, SE_MAX)]
    main_d.to_csv(OUT / "implied_weekly_main.csv", index=False)

    # ---------- step 3: coverage ----------
    cov = []
    for (xcol, se_max), d in results.items():
        g = d.groupby("winter")["informative"].agg(["size", "sum"])
        cov.append({"target": xcol, "se_max": se_max, "rows": len(d), "informative": int(d.informative.sum()),
                    **{w: f"{int(r['sum'])}/{int(r['size'])}" for w, r in g.iterrows()}})
    log.append("## Informative rows (Model 2 era)\n" + pd.DataFrame(cov).to_string(index=False))

    # ---------- step 4a: level and stability of the departure ----------
    rows = []
    for (xcol, se_max), d in results.items():
        inf = d[d.informative]
        if len(inf) < 5:
            continue
        lo, hi = winter_boot(inf, lambda z: z.u.mean())
        glo, ghi = winter_boot(inf, lambda z: z.G.mean())
        jlo, jhi = winter_boot(inf, lambda z: (z.u - z.u_ce).mean())
        rows.append({"target": xcol, "se_max": se_max, "n": len(inf), "winters": inf.winter.nunique(),
                     "mean u": inf.u.mean(), "u CI": f"{lo:.2f}..{hi:.2f}",
                     "share u<0.5": (inf.u < 0.5).mean(), "share u<0.1": (inf.u < 0.1).mean(),
                     "mean G (Bcf)": inf.G.mean(), "G CI": f"{glo:.0f}..{ghi:.0f}",
                     "mean u_ce": inf.u_ce.mean(), "mean u-u_ce": (inf.u - inf.u_ce).mean(),
                     "u-u_ce CI": f"{jlo:.2f}..{jhi:.2f}",
                     "sd u within winter": inf.groupby("winter").u.std().mean(),
                     "lag1 autocorr u": inf.sort_values("week").u.autocorr(1)})
    log.append("## 4a: percentile of market-implied level in Model 2 (informative rows)\n"
               + pd.DataFrame(rows).round(3).to_string(index=False))
    main_d["Lb"], main_d["bm"] = main_d.L_hat - main_d.benchmark, main_d.benchmark - main_d.p50
    inf = main_d[main_d.informative]
    log.append("pooled: G = (L_hat - 5y path) + (5y path - M2 median): %.0f = %.0f + %.0f; CI of L_hat - 5y path %s"
               % (inf.G.mean(), inf.Lb.mean(), inf.bm.mean(),
                  "%.0f..%.0f" % winter_boot(inf, lambda z: z.Lb.mean())))
    log.append("## 4a by winter (main spec)\n" + inf.groupby("winter").agg(
        n=("u", "size"), u_mean=("u", "mean"), u_min=("u", "min"), u_max=("u", "max"),
        G_mean=("G", "mean"), L_minus_bench=("Lb", "mean"), bench_minus_m2=("bm", "mean"), L_hat=("L_hat", "mean"), m2_med=("p50", "mean"),
        bench=("benchmark", "mean"), actual=("actual", "first")).round(2).to_string())

    fl = inf[inf.x_hat + inf.avg5 < 500]
    nf = inf[inf.x_hat + inf.avg5 >= 500]
    log.append(f"floored at 500 Bcf (implied below physical floor): {len(fl)} rows {sorted(fl.winter.unique())}; "
               f"excluding them: n={len(nf)} mean u={nf.u.mean():.3f} CI %s, mean G={nf.G.mean():.0f} CI %s, "
               f"mean L_hat-5y path={nf.Lb.mean():.0f} CI %s"
               % ("%.2f..%.2f" % winter_boot(nf, lambda z: z.u.mean()), "%.0f..%.0f" % winter_boot(nf, lambda z: z.G.mean()),
                  "%.0f..%.0f" % winter_boot(nf, lambda z: z.Lb.mean())))
    xw = inf[inf.winter != "2018-19"]
    log.append(f"excluding winter 2018-19: n={len(xw)} mean u={xw.u.mean():.3f}, mean G={xw.G.mean():.0f}, "
               f"mean L_hat-5y path={xw.Lb.mean():.0f}")

    # accuracy vs realised outcome (winters with an outcome <= 2021)
    acc = inf[inf.actual.notna()]
    accr = {"n": len(acc), "implied MAE": (acc.L_hat - acc.actual).abs().mean(),
            "M2 median MAE": (acc.p50 - acc.actual).abs().mean(),
            "5y path MAE": (acc.benchmark - acc.actual).abs().mean(),
            "implied bias": (acc.L_hat - acc.actual).mean(), "M2 bias": (acc.p50 - acc.actual).mean(),
            "realised pct in M2 (mean)": np.mean([pct_in(m2q[o], a) for o, a in zip(acc.origin, acc.actual)])}
    dd = (acc.p50 - acc.actual).abs() - (acc.L_hat - acc.actual).abs()
    accr["M2 minus implied |err| CI"] = "%.0f..%.0f" % winter_boot(acc.assign(dd=dd), lambda z: z.dd.mean())
    log.append("## Accuracy vs realised end-March (main spec, informative rows)\n" + pd.Series(accr).to_string())

    # ---------- step 4b: do departures predict the spread? ----------
    rows = []
    for (xcol, se_max), d in results.items():
        inf = d[d.informative].copy()
        inf["Gk"] = inf.G / 100.0  # per 100 Bcf
        inf["dev"] = inf.u - 0.5
        for h in (1, 4):
            y = f"dS_{h}w"
            for x in ("Gk", "dev"):
                sub = inf[[y, x, "winter"]].dropna()
                if len(sub) < 8:
                    continue
                b, t, n, r2 = hac(sub, y, x, h)
                lo, hi = winter_boot(sub, lambda z: ols_slope(z, y, x))
                big = sub[(inf.loc[sub.index, "u"] < 0.1) | (inf.loc[sub.index, "u"] > 0.9)]
                rows.append({"target": xcol, "se_max": se_max, "h": h, "regressor": x, "n": n,
                             "winters": sub.winter.nunique(), "slope": b, "CI": f"{lo:.3f}..{hi:.3f}",
                             "HAC t": t, "R2": r2, "n outside 10-90": len(big),
                             "mean dS outside": big[y].mean() if len(big) else np.nan,
                             "mean dS all": sub[y].mean()})
    tb = pd.DataFrame(rows)
    log.append("## 4b: dS(h) on departure. Gk = gap per 100 Bcf (implied - M2 median); dev = u - 0.5\n"
               + tb.round(3).to_string(index=False))

    # walk-forward OOS R2 of slope on Gk vs no change (main spec)
    inf = main_d[main_d.informative].copy()
    inf["Gk"], inf["Lbk"] = inf.G / 100, inf.Lb / 100
    oos = []
    for h in (1, 4):
        y = f"dS_{h}w"
        for x in ("Gk", "Lbk", "spread"):
            sub = inf[["Y", "winter", y, x]].dropna()
            b_is, t_is, n_is, r2 = hac(sub, y, x, h)
            ci = winter_boot(sub, lambda z: ols_slope(z, y, x))
            pr = []
            for Y in sorted(sub.Y.unique()):
                tr = sub[sub.Y < Y]
                if len(tr) < 5:
                    continue
                b = ols_slope(tr, y, x)
                a = tr[y].mean() - b * tr[x].mean()
                te = sub[sub.Y == Y]
                pr.append(pd.DataFrame({"e_m": te[y] - (a + b * te[x]), "e_0": te[y], "winter": te.winter}))
            p = pd.concat(pr)
            oos.append({"h": h, "regressor": x, "n": n_is, "slope": b_is, "CI": "%.3f..%.3f" % ci,
                        "HAC t": t_is, "in-sample R2": r2, "OOS n": len(p), "OOS winters": p.winter.nunique(),
                        "OOS R2 vs no change": 1 - (p.e_m ** 2).sum() / (p.e_0 ** 2).sum()})
        # does G add to the spread level?
        sub = inf[[y, "Gk", "spread"]].dropna()
        r = sm.OLS(sub[y], sm.add_constant(sub[["Gk", "spread"]])).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        log.append(f"h={h}: dS on Gk + spread: Gk {r.params['Gk']:.3f} (t {r.tvalues['Gk']:.2f}), "
                   f"spread {r.params['spread']:.3f} (t {r.tvalues['spread']:.2f}), n={len(sub)}")
    log.append("## 4b main spec: Gk = gap to M2 median, Lbk = gap to 5y path (per 100 Bcf), spread = level only. OOS walk-forward by winter\n" + pd.DataFrame(oos).round(3).to_string(index=False))

    # ---------- step 4c: who moves toward whom ----------
    rows = []
    for (xcol, se_max), d in results.items():
        d = d.sort_values("week")
        for h in (1, 4):
            nxt = d[["week", "L_hat", "p50", "informative"]].copy()
            nxt["week"] = nxt["week"] - pd.Timedelta(days=7 * h)
            m = d.merge(nxt, on="week", suffixes=("", "_n"))
            m = m[m.informative & m.informative_n].copy()
            if len(m) < 8:
                continue
            m["dL"], m["dM"] = m.L_hat_n - m.L_hat, m.p50_n - m.p50
            m["dG"] = m.dL - m.dM
            bL, tL, n, _ = hac(m, "dL", "G", h)
            bM, tM, _, _ = hac(m, "dM", "G", h)
            bG, tG, _, _ = hac(m, "dG", "G", h)
            ciL = winter_boot(m, lambda z: ols_slope(z, "dL", "G"))
            ciM = winter_boot(m, lambda z: ols_slope(z, "dM", "G"))
            ciG = winter_boot(m, lambda z: ols_slope(z, "dG", "G"))
            big = m[m.G.abs() >= m.G.abs().quantile(0.75)]
            sgn = np.sign(big.G)
            rows.append({"target": xcol, "se_max": se_max, "h": h, "n": n, "winters": m.winter.nunique(),
                         "b_market": bL, "CI_m": f"{ciL[0]:.2f}..{ciL[1]:.2f}",
                         "b_model2": bM, "CI_2": f"{ciM[0]:.2f}..{ciM[1]:.2f}",
                         "b_gap": bG, "CI_g": f"{ciG[0]:.2f}..{ciG[1]:.2f}", "t_gap": tG,
                         "big |G| n": len(big), "big: market toward (Bcf)": (-sgn * big.dL).mean(),
                         "big: model2 toward (Bcf)": (sgn * big.dM).mean()})
            if (xcol, se_max) == (MAIN, SE_MAX):
                m.to_csv(OUT / f"who_moves_h{h}.csv", index=False)
    log.append("## 4c: dL, dM, dG regressed on G (Bcf per Bcf). Convergence: b_gap < 0; "
               "market closes if b_market < 0, Model 2 closes if b_model2 > 0. 'big' = top quartile |G|; "
               "positive 'toward' = moved to close the gap\n" + pd.DataFrame(rows).round(3).to_string(index=False))

    text = "\n\n".join(log)
    print("written")
    (OUT / "results.txt").write_text(text)
    print(text)
    return results, fits, panel, m2, units


if __name__ == "__main__":
    main()
