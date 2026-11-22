"""Frozen holdout test V4 (= W1 in reports/agents/widowmaker_insurance.md section 6). Descriptive.

Spec (frozen; changes per reports/agents/review_v3.md and reports/synthesis_v3.md):
  rows      weekly decision settles: first trading day >= storage week F + 7 days, origins 1 Nov - 14 Feb,
            before the March contract's last trading day
  physical  Model 2 v2 p1-p99 at origin F, re-centred so the median = 5-year-path benchmark, floored at 700 Bcf;
            x = scenario - mean of the prior 5 end-March levels
  payout    frozen at the 2011-21 fit: y = a + b exp(-x/tau), 500 residual-bootstrap draws (seed 2022) + residual noise
            (identical to curves()[(2022, 'wf')] in src/research/widowmaker_insurance/payout.py)
  signal    premium = S_t - AV_t > $0.15 (W1).  Primary comparison: price-only twin S_t > $0.17 (and S_t > $0.15)
  trade     short 1 H-J, hold 4 weeks (exit at the March last trading day if sooner), 4 ticks/contract/side ($160 RT)
  stats     mean net $/trade with winter-block CI; slope of (S_exp - S_t) on -premium (winter-block 90% CI);
            worst mark-to-market drawdown per winter
  verdict   W1 "passes" if mean net > 0 and the slope's 90% CI excludes 0 (inconclusive if < 2 winters trade).
            PRIMARY: Model 2 adds value only if W1 mean net/trade beats the S > $0.17 twin by > $500.
  sample    winters 2022-23 .. 2025-26; Jan-Feb 2022 reported separately.

Existing research files are not edited; the panel is rebuilt here with the end date as a parameter.
In check mode the curve for each dev winter is the walk-forward fit used in the research report, so the
dev numbers reproduce; every holdout winter uses the frozen 2011-21 fit.

Run:
  PYTHONPATH=. python -m src.holdout.widowmaker_insurance check     # end 2021-12-31: reproduces the research numbers
  PYTHONPATH=. python -m src.holdout.widowmaker_insurance holdout   # run once on 2022+
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.widowmaker_insurance import analysis as A
from src.research.widowmaker_insurance import payout as P

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/research_widowmaker_insurance"
REP = ROOT / "reports/holdout"
VAL_END, DATA_END = pd.Timestamp("2021-12-31"), pd.Timestamp("2026-11-17")
X_W1, X_TWINS, FLOOR = 0.15, (0.17, 0.15), 700.0
MULT, RT = 10_000, 160.0
PCOLS = [f"p{k}" for k in range(1, 100)]
M2_FULL = ROOT / "data/processed/research_implied_storage/model2_percentiles_full_to_2026.csv"
M2_DEV = ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv"


def wl(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


# ---------------------------------------------------------------- data with an end date
def load(end):
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= end].set_index("week_ending")["storage_bcf"].sort_index()
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= end) & c.contract_month.isin([3, 4]) & (c.contract_year <= 2026)]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    sp = {}
    for Y in range(2011, 2027):
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        h, j = px[(Y, 3)], px[(Y, 4)]
        d = (h - j).dropna()
        d = d[d.index >= pd.Timestamp(Y - 1, 8, 1)]
        if len(d):
            sp[Y] = pd.DataFrame({"S": d, "J": j[d.index]})
    # expired = last H settle is in February and the data run beyond it
    ex = {Y: d.index.max() for Y, d in sp.items() if d.index.max().month == 2 and d.index.max() < end}
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    em = m.groupby(m.index.year).last()
    avg5 = {Y: float(np.mean([em[Y - k] for k in range(1, 6) if (Y - k) in em.index]))
            for Y in range(2011, 2027) if any((Y - k) in em.index for k in range(1, 6))}
    return s, sp, ex, em, avg5


def panel(end):
    s, sp, ex, em, avg5 = load(end)
    rows = []
    for F in s.index:
        if F.month not in (11, 12, 1, 2) or (F.month == 2 and F.day > 14):
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y not in sp:
            continue
        d = sp[Y]
        pos = d.index.searchsorted(F + pd.Timedelta(days=7))
        if pos >= len(d) or d.index[pos] > end:
            continue
        obs = d.index[pos]
        ltd = ex.get(Y, pd.NaT)
        if pd.notna(ltd) and obs >= ltd:
            continue
        r = {"origin": F, "obs": obs, "Y": Y, "winter": wl(Y), "S": d.S.iloc[pos], "J": d.J.iloc[pos],
             "S_exp": d.S.iloc[-1] if pd.notna(ltd) else np.nan}
        p4 = d.index.searchsorted(obs + pd.Timedelta(days=28))
        if p4 < len(d) and d.index[p4] <= end:
            r["S_4w"], r["date_4w"] = d.S.iloc[p4], d.index[p4]
        elif pd.notna(ltd):
            r["S_4w"], r["date_4w"] = d.S.iloc[-1], d.index[-1]
        else:
            r["S_4w"], r["date_4w"] = np.nan, pd.NaT
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("obs").reset_index(drop=True)
    assert df.obs.max() <= end
    return df, sp, em, avg5


def frozen_curves():
    """Walk-forward curves exactly as in the research module, built from 2011-21 data only."""
    _, sp, ex, em, avg5 = load(VAL_END)
    rows = []
    for Y in range(2011, 2022):
        rows.append({"Y": Y, "x": em[Y] - avg5[Y], "y": sp[Y].S.iloc[-1]})
    t = pd.DataFrame(rows).set_index("Y")
    cd = {}
    for Y in range(2011, 2023):
        tr = t[t.index < Y]
        if len(tr) < 5:
            continue
        p0, ps, e = P.bootstrap(tr.x.values, tr.y.values, B=500, seed=Y)
        cd[Y] = {"p0": p0, "ps": ps, "e": e, "n": len(tr)}
    return cd, t


def actuarial(pn, m2, cd, avg5, seed=1):
    rng = np.random.default_rng(seed)
    m2 = m2.set_index("origin")
    out = []
    for _, r in pn.iterrows():
        if r.origin not in m2.index:
            continue
        c = cd[min(r.Y, 2022)]           # holdout winters: frozen 2011-21 fit
        m = m2.loc[r.origin]
        s = np.maximum(m[PCOLS].values.astype(float) - m["p50"] + m["benchmark"], FLOOR)
        x = s - avg5[r.Y]
        y = P.predict(c["ps"], x)
        y = (y + rng.choice(c["e"], size=y.shape)).ravel()
        q = np.quantile(y, [0.1, 0.5, 0.9])
        out.append({"origin": r.origin, "av": y.mean(), "q10": q[0], "q50": q[1], "q90": q[2],
                    "p_large": (y > 0.20).mean(), "p_above": (y > r.S).mean()})
    res = pn.merge(pd.DataFrame(out), on="origin", how="inner")
    res["prem"] = res.S - res.av
    res["dS_exp"] = res.S_exp - res.S
    return res


# ---------------------------------------------------------------- stats
def trades(res, mask, sp):
    d = res[mask].dropna(subset=["S_4w"]).copy()
    d["net"] = -(d.S_4w - d.S) * MULT - RT
    mtm = []
    for _, r in d.iterrows():
        path = sp[r.Y].S
        path = path[(path.index >= r.obs) & (path.index <= r.date_4w)]
        mtm.append(-(path.max() - r.S) * MULT)
    d["worst_mtm"] = mtm
    return d


def summ(t, B=5000):
    if len(t) == 0:
        return {"trades": 0, "winters": 0, "mean_net": np.nan, "lo": np.nan, "hi": np.nan, "hit": np.nan,
                "worst": np.nan, "worst_mtm": np.nan, "total": 0.0}
    lo, hi = A.winter_boot(t, lambda d: d.net.mean(), B=B) if t.winter.nunique() > 1 else (np.nan, np.nan)
    return {"trades": len(t), "winters": t.winter.nunique(), "mean_net": t.net.mean(), "lo": lo, "hi": hi,
            "hit": (t.net > 0).mean(), "worst": t.net.min(), "worst_mtm": t.worst_mtm.min(), "total": t.net.sum()}


def slope(res, B=5000):
    d = res.dropna(subset=["dS_exp"])
    if d.winter.nunique() < 2:
        return {"n": len(d), "slope": np.nan, "lo90": np.nan, "hi90": np.nan}
    b = np.polyfit(-d.prem, d.dS_exp, 1)
    rng = np.random.default_rng(0)
    ws = d.winter.unique()
    g = {w: d[d.winter == w] for w in ws}
    vals = []
    for _ in range(B):
        z = pd.concat([g[w] for w in rng.choice(ws, len(ws))])
        if z.prem.std() > 0:
            vals.append(np.polyfit(-z.prem, z.dS_exp, 1)[0])
    lo, hi = np.quantile(vals, [0.05, 0.95])
    return {"n": len(d), "winters": len(ws), "slope": b[0], "icpt": b[1], "lo90": lo, "hi90": hi,
            "mean_prem": d.prem.mean(), "mean_dS": d.dS_exp.mean()}


def evaluate(res, sp, winters):
    r = res[res.winter.isin(winters)]
    rules = {"W1: premium > $0.15": r.prem > X_W1}
    for x in X_TWINS:
        rules[f"Twin: spread > ${x:.2f}"] = r.S > x
    tt = {k: trades(r, m, sp) for k, m in rules.items()}
    tab = pd.DataFrame({k: summ(v) for k, v in tt.items()}).T
    return tab, tt, slope(r)


# ---------------------------------------------------------------- main
def run(mode):
    end = VAL_END if mode == "check" else DATA_END
    full = pd.read_csv(M2_FULL, parse_dates=["origin", "target"])
    full = full[full.origin <= end]
    if mode == "check":
        dev = pd.read_csv(M2_DEV, parse_dates=["origin", "target"])
        a, b = full.set_index("origin")[PCOLS + ["benchmark"]], dev.set_index("origin")[PCOLS + ["benchmark"]]
        assert a.index.equals(b.index) and np.allclose(a.values, b.values), "full cache != dev cache"
        print("Full Model 2 cache matches the dev cache on all", len(a), "dev origins")
    pn, sp, em, avg5 = panel(end)
    cd, t = frozen_curves()
    print("Frozen 2011-21 curve a, b, tau:", cd[2022]["p0"].round(3), " n =", cd[2022]["n"])
    res = actuarial(pn, full, cd, avg5)

    if mode == "check":
        ws = sorted(res.winter.unique())
        print(f"mean premium {res.prem.mean():.3f} (research: 0.177)")
        tab, tt, sl = evaluate(res, sp, ws)
        print(tab.round(2).to_string())
        print("slope", {k: round(v, 3) for k, v in sl.items()}, "(research OLS slope 1.04)")
        return
    res.to_csv(OUT / "holdout_v4_weeks.csv", index=False)
    blocks = {"Jan-Feb 2022": ["2021-22"], "2022-23 to 2025-26": ["2022-23", "2023-24", "2024-25", "2025-26"]}
    res = res[res.obs >= pd.Timestamp("2022-01-01")]
    out = {}
    for lab, ws in blocks.items():
        tab, tt, sl = evaluate(res, sp, ws)
        out[lab] = (tab, tt, sl)
        print(f"\n=== {lab} ===")
        print(tab.round(2).to_string())
        print("slope", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sl.items()})
    tab, tt, sl = out["2022-23 to 2025-26"]
    ws = blocks["2022-23 to 2025-26"]
    rw = res[res.winter.isin(ws)]
    per = rw.groupby("winter").agg(weeks=("S", "size"), S=("S", "mean"), AV=("av", "mean"), prem=("prem", "mean"),
                                   p_large=("p_large", "mean"), S_exp=("S_exp", "first"), S_max=("S", "max"))
    for k, v in tt.items():
        g = v.groupby("winter").agg(n=("net", "size"), net=("net", "sum"), mtm=("worst_mtm", "min"))
        per[f"{k[:4]}{k[-5:]} trades"] = g.n
        per[f"{k[:4]}{k[-5:]} net"] = g.net
        per[f"{k[:4]}{k[-5:]} worst MTM"] = g.mtm
    # whole-winter path risk: short from first decision week to expiry, worst MTM
    path = {}
    for w in ws:
        Y = int(w[:4]) + 1
        first = rw[rw.winter == w].obs.min()
        d = sp[Y].S[sp[Y].S.index >= first]
        path[w] = -(d.max() - d.iloc[0]) * MULT
    per["hold-all-winter worst MTM"] = pd.Series(path)
    per = per.fillna({c: 0 for c in per.columns if c.endswith("trades") or c.endswith(" net")})
    print(per.round(3).T.to_string())
    per.to_csv(OUT / "holdout_v4_per_winter.csv")
    pd.concat({k: v for k, v in tt.items()}).to_csv(OUT / "holdout_v4_trades.csv")
    tab.to_csv(OUT / "holdout_v4_summary.csv")
    w1, tw = tab.loc["W1: premium > $0.15"], tab.loc["Twin: spread > $0.17"]
    diff = w1.mean_net - tw.mean_net
    print(f"\nW1 - twin(0.17) = {diff:,.0f} $/trade; Model 2 adds value: {diff > 500}")
    if w1.winters < 2:
        print("W1 verdict: INCONCLUSIVE (< 2 winters trade)")
    else:
        print(f"W1 own criterion: mean>0 {w1.mean_net > 0}; slope 90% CI excludes 0 "
              f"{(sl['lo90'] > 0) or (sl['hi90'] < 0)}")
    fig(rw)


def fig(rw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = sorted(rw.winter.unique())
    f, axes = plt.subplots(1, len(ws), figsize=(4 * len(ws), 3.8), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), ws):
        a = rw[rw.winter == w]
        ax.fill_between(a.obs, a.q10, a.q90, color="#eb6834", alpha=0.18, lw=0, label="Physical 10-90%")
        ax.plot(a.obs, a.av, color="#eb6834", lw=2, label="Actuarial value")
        ax.plot(a.obs, a.S, color="#2a78d6", lw=2, marker="o", ms=3, label="Market H-J")
        ax.plot(a.obs, a.av + X_W1, color="#52514e", lw=1, ls="--", label="W1 trigger (AV + $0.15)")
        ax.axhline(0.17, color="#6250d6", lw=1, ls=":", label="Twin trigger ($0.17)")
        if a.S_exp.notna().any():
            ax.axhline(a.S_exp.iloc[0], color="#0b0b0b", lw=0.8, ls="-.", label="Expiry spread")
        ax.set_title(f"Winter {w}", fontsize=10, loc="left")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("$/MMBtu")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    f.tight_layout(rect=(0, 0.08, 1, 1))
    f.savefig(REP / "figures/v3_widowmaker_insurance_holdout.png", dpi=120)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "check")
