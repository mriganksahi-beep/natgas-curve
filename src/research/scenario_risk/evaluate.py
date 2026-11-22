"""Backtests of the risk forecasts. Run: PYTHONPATH=. python -m src.research.scenario_risk.evaluate

Units: $ per spread (1 March + 1 April contract short/long, 10,000 MMBtu): x 10,000.
Samples (common dates within each, so models are compared on identical weeks):
  M2   Nov 2016 - Dec 2021, every model (PHYS-M2 needs Model 2)
  WX   Nov 2013 - Dec 2021, models with a walk-forward map (needs 2 earlier winters)
  LONG Nov 2012 - Dec 2021, (a)-(c) only (HS-win needs 1 full earlier winter)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.research.scenario_risk import data as D
from src.research.scenario_risk.models import U, run_all

USD = 10_000
N_BOOT = 1000
RNG = np.random.default_rng(7)
SAMPLES = {
    "M2": (["HS-1y", "HS-3y", "HS-win", "EWMA", "GARCH-t", "HYB", "PHYS-WX", "PHYS-M2", "PHYS-M2-pure"],
           "2016-11-01"),
    "WX": (["HS-1y", "HS-3y", "HS-win", "EWMA", "GARCH-t", "HYB", "PHYS-WX"], "2013-11-01"),
    "LONG": (["HS-1y", "HS-3y", "HS-win", "EWMA", "GARCH-t"], "2012-11-01"),
}
TAIL = U >= 0.9


def var_es(q):
    return q[int(0.95 * 1000)], q[int(0.99 * 1000)], q[U > 0.975].mean(), q[int(0.975 * 1000)]


def pinball(q, y):
    return np.where(y >= q, U * (y - q), (1 - U) * (q - y))


def kupiec(x, p):
    n, k = len(x), int(x.sum())
    if k in (0, n):
        ll1 = 0.0
    else:
        ph = k / n
        ll1 = k * np.log(ph) + (n - k) * np.log(1 - ph)
    ll0 = k * np.log(p) + (n - k) * np.log(1 - p)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)


def christoffersen(x):
    """Independence LR (first-order Markov), within-sample order."""
    x = np.asarray(x, int)
    a, b = x[:-1], x[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    if n01 + n11 == 0:
        return np.nan
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(k, n, q):
        return 0.0 if q in (0, 1) else k * np.log(q) + (n - k) * np.log(1 - q)
    l0 = ll(n01 + n11, n00 + n01 + n10 + n11, p)
    l1 = ll(n01, n00 + n01, p01) + ll(n11, n10 + n11, p11)
    return 1 - stats.chi2.cdf(-2 * (l0 - l1), 1)


def winter_boot(df, fn):
    ws = df.winter.unique()
    g = {w: df[df.winter == w] for w in ws}
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(ws, len(ws), replace=True)
        vals.append(fn(pd.concat([g[w] for w in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])


def enrich(fc):
    v = np.vstack([var_es(q) for q in fc.q])
    fc = fc.assign(VaR95=v[:, 0], VaR99=v[:, 1], ES975=v[:, 2], VaR975=v[:, 3])
    fc["crps"] = [2 * pinball(q, y).mean() for q, y in zip(fc.q, fc.loss)]
    fc["tailpb"] = [pinball(q, y)[TAIL].mean() for q, y in zip(fc.q, fc.loss)]
    fc["pit"] = [np.interp(y, q, U, left=0.0005, right=0.9995) for q, y in zip(fc.q, fc.loss)]
    for c in ("VaR95", "VaR99", "ES975", "VaR975", "loss", "crps", "tailpb"):
        fc[c] = fc[c] * USD
    return fc


def common(fc, models, start, h):
    x = fc[(fc.h == h) & fc.model.isin(models) & (fc.date >= start)]
    ok = x.groupby("date").model.nunique()
    return x[x.date.isin(ok[ok == len(models)].index)]


def z2(g):
    i = g.loss > g.VaR975
    return (g.loss * i / (0.025 * g.ES975)).mean() - 1


def coverage_table(fc):
    rows = []
    for sname, (models, start) in SAMPLES.items():
        for h in (1, 4):
            x = common(fc, models, start, h)
            for m in models:
                g = x[x.model == m].sort_values("date")
                b95, b99, b975 = g.loss > g.VaR95, g.loss > g.VaR99, g.loss > g.VaR975
                rows.append(dict(
                    sample=sname, h=h, model=m, n=len(g), winters=g.winter.nunique(),
                    br95=b95.mean(), br95_ci=winter_boot(g, lambda d: (d.loss > d.VaR95).mean()),
                    kup95=kupiec(b95.values, 0.05), chr95=christoffersen(b95.values) if h == 1 else np.nan,
                    br99=b99.mean(), n99=int(b99.sum()), kup99=kupiec(b99.values, 0.01),
                    chr99=christoffersen(b99.values) if h == 1 else np.nan,
                    n975=int(b975.sum()), Z2=z2(g), Z2_ci=winter_boot(g, z2),
                    breach_over_ES=(g.loss[b975].mean() / g.ES975[b975].mean()) if b975.any() else np.nan,
                    mean_ES=g.ES975.mean(), med_ES=g.ES975.median(),
                    crps=g.crps.mean(), tailpb=g.tailpb.mean()))
    return pd.DataFrame(rows)


def dm_table(fc, bench="EWMA"):
    rows = []
    for sname, (models, start) in SAMPLES.items():
        for h in (1, 4):
            x = common(fc, models, start, h)
            b = x[x.model == bench].set_index("date")
            for m in models:
                if m == bench:
                    continue
                g = x[x.model == m].set_index("date")
                for sc in ("crps", "tailpb"):
                    dd = (g[sc] - b[sc]).dropna()
                    ols = sm.OLS(dd.values, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 2)})
                    tmp = pd.DataFrame({"d": dd.values, "winter": g.loc[dd.index, "winter"].values})
                    ci = winter_boot(tmp, lambda d: d.d.mean())
                    rows.append(dict(sample=sname, h=h, model=m, score=sc, diff=dd.mean(),
                                     rel=dd.mean() / b[sc].mean(), t=ols.tvalues[0], lo=ci[0], hi=ci[1]))
    return pd.DataFrame(rows)


EPISODES = {"Jan 2014": ("2014-01-01", "2014-01-31"), "Nov 2018": ("2018-11-01", "2018-11-30"),
            "Nov 2021": ("2021-11-01", "2021-11-30")}


def early_warning(fc):
    """Jump week = largest 1-week rise of the spread inside each episode month (defined a priori).
    For each model: ES/VaR made on that Friday (i.e. before the jump), the loss, and how elevated the
    ES was against the model's own median ES over its earlier forecasts (alarm ratio)."""
    rows = []
    h1 = fc[fc.h == 1]
    for ep, (a, b) in EPISODES.items():
        e = h1[(h1.date >= a) & (h1.date <= b) & (h1.model == "EWMA")]
        d0 = e.loc[e.loss.idxmax(), "date"]
        for h in (1, 4):
            x = fc[(fc.h == h) & (fc.date == d0)]
            for _, r in x.iterrows():
                prior = fc[(fc.h == h) & (fc.model == r.model) & (fc.date < d0)]
                rows.append(dict(episode=ep, date=d0.date(), h=h, model=r.model, S=r.S * 1.0,
                                 loss=r.loss, VaR99=r.VaR99, ES975=r.ES975, loss_over_ES=r.loss / r.ES975,
                                 alarm=r.ES975 / prior.ES975.median() if len(prior) >= 10 else np.nan,
                                 pit=r.pit))
    return pd.DataFrame(rows)


def sizing(fc, models, start, h=1, budget=5000.0, k_ticks=4, es_floor=100.0):
    """Short-spread premium book sized each Friday to a fixed ES budget: N = budget / ES ($).
    h=1: weekly resizing, costs k ticks x $10 per contract per side on |dN| (2 legs), and on opening
    the first and closing the last position of each winter. h=4: non-overlapping 4-week holds.
    'CONST' = 1 spread every week."""
    x = common(fc, models, start, h)
    out, series = [], {}
    for m in models + ["CONST"]:
        g = x[x.model == (m if m != "CONST" else models[0])].sort_values("date").copy()
        if h == 4:  # non-overlapping blocks: first Friday of each winter then every 4th
            g = g[g.groupby("winter").cumcount() % 4 == 0]
        g["N"] = 1.0 if m == "CONST" else budget / np.maximum(g.ES975, es_floor)
        pnl = -g.N * g.loss
        cost = np.zeros(len(g))
        prevN, prevW = 0.0, None
        for i, (n, w) in enumerate(zip(g.N.values, g.winter.values)):
            if h == 4:
                cost[i] = 2 * 2 * n * k_ticks * 10  # open + close each block, 2 legs
                continue
            if w != prevW:
                cost[i] += 2 * prevN * k_ticks * 10 + 2 * n * k_ticks * 10  # close old, open new
            else:
                cost[i] += 2 * abs(n - prevN) * k_ticks * 10
            prevN, prevW = n, w
        if h == 1:
            cost[-1] += 2 * prevN * k_ticks * 10
        g["pnl"] = pnl.values - cost
        cum = g.pnl.cumsum()
        series[m] = g.set_index("date").pnl
        r4 = g.pnl.rolling(4).sum().min() if h == 1 else g.pnl.min()
        out.append(dict(model=m, weeks=len(g), mean_N=g.N.mean(), total=g.pnl.sum(),
                        per_avg_spread=g.pnl.sum() / g.N.mean(), worst=g.pnl.min(),
                        worst_per_avg=g.pnl.min() / g.N.mean(), worst4=r4,
                        maxdd=(cum.cummax() - cum).max(),
                        sharpe=g.pnl.mean() / g.pnl.std() * np.sqrt(52 / h),
                        tot_over_dd=g.pnl.sum() / max((cum.cummax() - cum).max(), 1),
                        win_pos=(g.groupby("winter").pnl.sum() > 0).mean()))
    return pd.DataFrame(out), pd.DataFrame(series)


def m2_diagnostic():
    """Ex-post check of the sqrt(h/weeks_to_go) assumption: realised s.d. of h-week changes in
    Model 2's mean vs the implied s.d. (diagnostic only; not used by any forecast)."""
    m2 = D.load_model2()
    m2["sd"] = m2[[f"p{k}" for k in range(1, 100)]].std(axis=1)
    rows = []
    for h in (1, 4):
        nxt = m2["mean"].shift(-h)
        same = m2.target == m2.target.shift(-h)
        ch = (nxt - m2["mean"]).where(same)
        imp = m2.sd * np.sqrt(h / m2.weeks_to_go)
        ok = ch.notna() & m2.index.month.isin([11, 12, 1, 2])
        rows.append(dict(h=h, n=int(ok.sum()), realised_sd=ch[ok].std(), implied_sd_rms=np.sqrt((imp[ok] ** 2).mean()),
                         ratio=ch[ok].std() / np.sqrt((imp[ok] ** 2).mean())))
    return pd.DataFrame(rows)


def main():
    fc = enrich(run_all())
    assert fc.date.max() <= D.CUTOFF
    fc.drop(columns="q").to_csv(D.CACHE / "forecasts_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cov = coverage_table(fc)
    cov.to_csv(D.CACHE / "coverage.csv", index=False)
    show = cov.copy()
    show["br95_ci"] = show.br95_ci.apply(lambda c: f"{c[0]:.2f}-{c[1]:.2f}")
    show["Z2_ci"] = show.Z2_ci.apply(lambda c: f"{c[0]:.2f}-{c[1]:.2f}")
    print(show.round(3).to_string())
    dm = dm_table(fc)
    dm.to_csv(D.CACHE / "dm.csv", index=False)
    print(dm.round(3).to_string())
    ew = early_warning(fc)
    ew.to_csv(D.CACHE / "early_warning.csv", index=False)
    print(ew.round(2).to_string())
    for sname in ("M2", "WX"):
        models, start = SAMPLES[sname]
        models = [m for m in models if m != "PHYS-M2-pure"]
        for h in (1, 4):
            for k in (4,):
                tab, ser = sizing(fc, models, start, h=h, k_ticks=k)
                tab.to_csv(D.CACHE / f"sizing_{sname}_h{h}_k{k}.csv", index=False)
                ser.to_csv(D.CACHE / f"sizing_series_{sname}_h{h}.csv")
                print(sname, "h", h, "k", k)
                print(tab.round(2).to_string())
    print(m2_diagnostic().round(2))


if __name__ == "__main__":
    main()
