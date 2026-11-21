"""Frozen holdout test H3 (implied_storage): implied-storage gap vs spread-only, 4-week horizon.

Spec (frozen in reports/agents/implied_storage.md section 6, and reports/synthesis_v2.md):
  rows      Nov-Feb storage weeks, origin >= 2022-01-01, informative (implied s.e. <= 250 Bcf),
            with a 4-week spread change before the March contract's last 25 days
  mapping   x_info, spread as % of April, a in [-0.2, 0.1], refit each winter on all earlier
            winters with a realised end-March level (from 2011-12)
  gap       G = (L_hat - Model 2 v2b median) / 100 Bcf
  forecasts gap model     dS = -0.110 + 0.0675 * G
            spread-only   dS =  0.013 - 0.633 * S
  test      one-sided Diebold-Mariano on squared errors (gap < spread-only), HAC lag 4
  pass      p < 0.05 AND OOS R2 of gap model vs no change > 0
  secondary mean(L_hat - 5-year path) with a winter-bootstrap CI (not pass/fail)

The existing research files are not edited. The panel builder is re-implemented here with an end
date as a parameter, because src/research/implied_storage/data.py hard-codes the 2021 cutoff.
The mapping, inversion and percentile code are imported unchanged from the research module.

Run:
  PYTHONPATH=. python -m src.holdout.implied_storage check     # end 2021-12-31: must reproduce validation
  PYTHONPATH=. python -m src.holdout.implied_storage holdout   # run once on 2022+
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from src.research.implied_storage import data as D
from src.research.implied_storage.analysis import pct_in, walk_forward, winter_boot
from src.research.implied_storage.simulate import PCTS

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/research_implied_storage"
REP = ROOT / "reports/holdout"
VAL_END, DATA_END = pd.Timestamp("2021-12-31"), pd.Timestamp("2026-11-17")
SE_MAX, UNITS, XCOL = 250.0, "pct", "x_info"
GAP_A, GAP_B, SPR_A, SPR_B = -0.110, 0.0675, 0.013, -0.633


# ---------------------------------------------------------------- data (end as a parameter)
def build_panel(end):
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= end].set_index("week_ending")["storage_bcf"]
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= end) & c.contract_month.isin([3, 4]) & (c.contract_year <= end.year + 1)]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    em = D.end_march(s)
    days = px.index
    rows = []
    for t in s.index:
        if t.month not in (11, 12, 1, 2) or t < pd.Timestamp("2010-11-01"):
            continue
        Y = t.year + 1 if t.month >= 11 else t.year
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        exp = D.march_expiry(Y)
        pos = days.searchsorted(t + pd.Timedelta(days=7))
        if pos >= len(days):
            continue
        obs = days[pos]
        if obs > exp - pd.Timedelta(days=D.MIN_DAYS_BEFORE_EXPIRY):
            continue
        h, j = px[(Y, 3)], px[(Y, 4)]
        if pd.isna(h.get(obs)) or pd.isna(j.get(obs)):
            continue
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index]
        if len(prior) < 2:
            continue
        avg5 = float(np.mean(prior))
        draws = []
        for k in range(1, 6):
            if (Y - k) not in em.index:
                continue
            past = D._nearest(s, t - pd.DateOffset(years=k))
            if past is not None:
                draws.append(em[Y - k] - past)
        row = {"week": t, "obs_date": obs, "Y": Y, "winter": f"{Y - 1}-{str(Y)[2:]}",
               "spread": h[obs] - j[obs], "april": j[obs], "march": h[obs],
               "weeks_to_expiry": (exp - obs).days / 7, "storage_t": s[t], "avg5": avg5,
               "n_prior": len(prior),
               "x_info": s[t] + np.mean(draws) - avg5 if len(draws) >= 2 else np.nan,
               "em_real": em.get(Y, np.nan)}
        row["x_real"] = row["em_real"] - avg5
        for hz in (1, 4):
            p2 = days.searchsorted(obs + pd.Timedelta(days=7 * hz))
            ok = p2 < len(days) and days[p2] <= exp and not pd.isna(h.get(days[p2])) and not pd.isna(j.get(days[p2]))
            row[f"dS_{hz}w"] = (h[days[p2]] - j[days[p2]]) - row["spread"] if ok else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    assert df.obs_date.max() <= end
    return df


# ---------------------------------------------------------------- Model 2 percentiles
def model2_percentiles(end):
    """Validation (end <= 2021): the cache written by simulate.py. Holdout: one Simulator(seed 0),
    all Nov-Feb origins from Nov 2016 in date order (the same random stream as the validation run),
    cached; the first rows must reproduce the validation cache."""
    val = pd.read_csv(CACHE / "model2_percentiles.csv", parse_dates=["origin", "target"])
    if end <= VAL_END:
        return val
    path = CACHE / "model2_percentiles_full_to_2026.csv"
    if not path.exists():
        from src.model2 import Simulator
        sim = Simulator()
        weeks = sim.df.index
        origins = [t for t in weeks if t >= pd.Timestamp("2016-11-01") and t <= pd.Timestamp("2026-02-28")
                   and t.month in (11, 12, 1, 2)]
        rows = []
        for t in origins:
            r = sim.simulate(t)
            if r is None:
                continue
            q = np.percentile(r["levels"], PCTS)
            row = {"origin": t, "target": r["target"], "weeks_to_go": r["weeks_to_go"],
                   "n_scenarios": r["n_scenarios"], "mean": float(r["levels"].mean()),
                   "benchmark": r["benchmark"], "actual": r["actual"]}
            row.update({f"p{p}": v for p, v in zip(PCTS, q)})
            rows.append(row)
            print(t.date(), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    full = pd.read_csv(path, parse_dates=["origin", "target"])
    a = full[full.origin <= VAL_END].reset_index(drop=True)
    diff = (a["p50"] - val["p50"]).abs().max()
    print(f"Model 2 cache check: {len(a)} validation origins re-simulated, max |median diff| = {diff:.3f} Bcf")
    return full


# ---------------------------------------------------------------- test
def dm_one_sided(e_gap, e_spr, lags=4):
    d = e_gap ** 2 - e_spr ** 2
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(np.asarray(r.params)[0] / np.asarray(r.bse)[0])
    return float(d.mean()), t, float(norm.cdf(t))  # H1: mean d < 0 (gap model better)


def run(end, window):
    panel = build_panel(end)
    m2 = model2_percentiles(end)
    m2 = m2[m2.origin <= end]
    qcols = [f"p{p}" for p in PCTS]
    m2q = {r.origin: np.array([getattr(r, c) for c in qcols]) for r in m2.itertuples()}
    wf = walk_forward(panel, XCOL, UNITS, SE_MAX)
    d = wf.merge(m2, left_on="week", right_on="origin", how="inner")
    d["u"] = [pct_in(m2q[o], L) if np.isfinite(L) else np.nan for o, L in zip(d.origin, d.L_hat)]
    d["G"] = d.L_hat - d["p50"]
    d["Lb"] = d.L_hat - d.benchmark
    if window == "validation":
        d = d[(d.Y >= 2017) & (d.week <= VAL_END)]
    else:
        d = d[d.week >= pd.Timestamp("2022-01-01")]
    d = d.copy()
    inf = d[d.informative]
    rows = inf.dropna(subset=["dS_4w"]).copy()
    rows["f_gap"] = GAP_A + GAP_B * rows.G / 100
    rows["f_spr"] = SPR_A + SPR_B * rows.spread
    e_gap, e_spr, e_0 = rows.dS_4w - rows.f_gap, rows.dS_4w - rows.f_spr, rows.dS_4w
    dbar, t, p = dm_one_sided(e_gap.values, e_spr.values)
    r2_gap = 1 - (e_gap ** 2).sum() / (e_0 ** 2).sum()
    r2_spr = 1 - (e_spr ** 2).sum() / (e_0 ** 2).sum()
    lb_ci = winter_boot(inf, lambda z: z.Lb.mean())
    g_ci = winter_boot(inf, lambda z: z.G.mean())
    cov = d.groupby("winter")["informative"].agg(["size", "sum"])
    res = {"window": window, "weeks": len(d), "readable": int(inf.shape[0]), "test rows": len(rows),
           "winters in test": rows.winter.nunique(),
           "MSE gap": float((e_gap ** 2).mean()), "MSE spread-only": float((e_spr ** 2).mean()),
           "MSE no change": float((e_0 ** 2).mean()),
           "DM mean d": dbar, "DM t (HAC 4)": t, "DM p one-sided": p,
           "OOS R2 gap vs no change": float(r2_gap), "OOS R2 spread-only vs no change": float(r2_spr),
           "PASS": bool(p < 0.05 and r2_gap > 0),
           "mean L_hat - 5y path": float(inf.Lb.mean()), "CI L_hat - 5y path": lb_ci,
           "mean G (Bcf)": float(inf.G.mean()), "CI G": g_ci, "mean u": float(inf.u.mean()),
           "share u outside 10-90": float(((inf.u < 0.1) | (inf.u > 0.9)).mean()) if len(inf) else np.nan}
    # in-sample refit on the test rows (descriptive only; never used for the verdict)
    if window == "validation":
        res["refit gap (a,b)"] = tuple(np.polyfit(rows.G / 100, rows.dS_4w, 1)[::-1].round(4))
        res["refit spread (a,b)"] = tuple(np.polyfit(rows.spread, rows.dS_4w, 1)[::-1].round(4))
    by_w = rows.groupby("winter").apply(lambda z: pd.Series({
        "n": len(z), "MSE gap": ((z.dS_4w - z.f_gap) ** 2).mean(),
        "MSE spread": ((z.dS_4w - z.f_spr) ** 2).mean(), "MSE none": (z.dS_4w ** 2).mean()}))
    return res, cov, by_w, d


def headline(d_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.research.implied_storage.figures import BLUE, GREY, INK, MUTED, ORANGE
    winters = sorted(d_all.winter.unique())
    ncol = 5
    nrow = int(np.ceil(len(winters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.4 * nrow), sharey=True)
    for ax in axes.ravel()[len(winters):]:
        ax.axis("off")
    for ax, w in zip(axes.ravel(), winters):
        g = d_all[d_all.winter == w].sort_values("obs_date")
        x = g.obs_date
        ax.fill_between(x, g.p10, g.p90, color=BLUE, alpha=0.18, lw=0, label="Model 2 10-90%")
        ax.plot(x, g.p50, color=BLUE, lw=2, label="Model 2 median")
        ax.plot(x, g.benchmark, color=GREY, lw=1.2, ls="--", label="5-yr path")
        if g.actual.notna().any():
            ax.axhline(g.actual.dropna().iloc[0], color=INK, lw=1.2, label="Realised end-March")
        inf = g[g.informative]
        ax.plot(inf.obs_date, inf.L_hat, "o", color=ORANGE, ms=6, mec="white", mew=1, label="Market-implied")
        fl = g[~g.informative]
        L_flat = fl.avg5 - (1000 / fl.c) * np.log(fl.sigma * 1000 / (fl.c * SE_MAX) / fl.b)
        ax.plot(fl.obs_date, L_flat, "^", color=ORANGE, mfc="white", ms=6,
                label="Market-implied: at least (curve too flat)")
        tag = "holdout" if g.week.min() >= pd.Timestamp("2022-01-01") else (
            "validation / holdout split" if w == "2021-22" else "validation")
        ax.set_title(f"Winter {w}  ({tag})", fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        if w == "2021-22":
            ax.axvline(pd.Timestamp("2022-01-01"), color=MUTED, lw=0.8, ls=":")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=7)
    for r in range(nrow):
        axes[r, 0].set_ylabel("End-March storage (Bcf)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("Market-implied end-March storage vs Model 2's scenario range, validation and holdout "
                 "(walk-forward mapping)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def show(res, cov, by_w):
    for k, v in res.items():
        print(f"{k:34s} {v}")
    print(cov.rename(columns={"size": "weeks", "sum": "readable"}).to_string())
    print(by_w.round(4).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        res, cov, by_w, _ = run(VAL_END, "validation")
        show(res, cov, by_w)
    elif mode == "holdout":
        flag = CACHE / "H3_holdout_ran.txt"
        if flag.exists():
            sys.exit("H3 holdout already run once; results in " + str(flag))
        res_v, _, _, d_v = run(VAL_END, "validation")
        res, cov, by_w, d_h = run(DATA_END, "holdout")
        show(res, cov, by_w)
        flag.write_text(pd.Series(res).to_string() + "\n\n" + cov.to_string() + "\n\n" + by_w.round(4).to_string())
        d_all = pd.concat([d_v[d_v.week < pd.Timestamp("2022-01-01")], d_h])
        d_all.to_csv(CACHE / "H3_weekly_all.csv", index=False)
        headline(d_all, REP / "figures/implied_storage_holdout_headline.png")
