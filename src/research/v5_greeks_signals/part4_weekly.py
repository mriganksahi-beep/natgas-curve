"""Part 4: storage-report event variance from weekly options (2014+; thin 2017-21).

Report days R: EIA weekly storage release = week_ending + 6 days (Thursday); if that is not a trading
day, the previous trading day (EIA moves holiday releases earlier, e.g. Thanksgiving Wednesday) -
an approximation for the few irregular weeks.

Implied (per date d, per weekly series with n = tdays trading days to expiry, n_E report days in (d, expiry]):
  V = atm_iv^2 * n / 252 (total implied variance, trading-day clock as in the panel)
  A) direct: rows with n = 1 -> one-day implied variance on a report day (n_E = 1) or non-report day
  B) per-date cross-section: V_i = a (n_i - nE_i) + b nE_i over the date's series (>= 3 series,
     both a report and a non-report day present); event excess variance = b - a  (per day units)
Realised: squared daily log change of the matched underlying futures; report-day vs other days.
Trade: at the settle before each report day, sell the shortest weekly ATM straddle that spans the
report (n <= 2), strike nearest the parity F, hold to expiry, unhedged. Cost 3% of premium on entry
plus $0.002 if the option finishes in the money (exercise -> futures exit). Benchmark: the same trade
on spans that contain no report day (n <= 2), and the short delta-hedged monthly front straddle on the
same report day (dh_straddle, 1.5% costs).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.v5_greeks_signals import common as C


def report_days(days: pd.DatetimeIndex, cut=C.CUT) -> pd.DatetimeIndex:
    s = pd.read_csv(C.ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    out = []
    for we in s.week_ending:
        r = we + pd.Timedelta(days=6)
        if r < pd.Timestamp("2014-01-01") or r > cut:
            continue
        if r in days:
            out.append(r)
        else:
            p = days.searchsorted(r)
            out.append(days[p - 1])
    return pd.DatetimeIndex(sorted(set(out)))


def main():
    px = C.curve()
    days = px.index
    R = report_days(days)
    isR = pd.Series(days.isin(R), index=days)
    w = C.weekly()
    # match underlying futures by parity F (nearest of the first 3 live contracts on that date)
    cid = []
    for d, F in zip(w.date, w.F):
        row = px.loc[d].dropna()
        row = row[row.index >= d.year * 100 + d.month][:3]
        cid.append(row.index[np.argmin(np.abs(row.values - F))] if len(row) else np.nan)
    w["cid"] = cid
    w["match_err"] = [abs(px.at[d, c] - F) if np.isfinite(c) else np.nan for d, c, F in zip(w.date, w.cid, w.F)]
    w = w[w.match_err < 0.02].copy()
    pos = {d: i for i, d in enumerate(days)}
    w["n"] = w.tdays
    w["nE"] = [int(isR.iloc[pos[d] + 1: pos[d] + 1 + n].sum()) for d, n in zip(w.date, w.n)]
    w["V"] = w.atm_iv ** 2 * w.n / 252
    w = w[(w.atm_iv > 0.05) & (w.atm_iv < 3)]
    # realised squared log changes per (date, cid)
    lr = np.log(px).diff()
    # A) one-day implied variance
    one = w[w.n == 1].copy()
    one["d1"] = [days[pos[d] + 1] for d in one.date]
    one["r2"] = [lr.at[d1, c] ** 2 for d1, c in zip(one.d1, one.cid)]
    one = one.drop_duplicates(["date", "cid", "expiry"])
    # B) per-date cross-section
    recs = []
    for d, g in w.groupby("date"):
        g = g.drop_duplicates(["n", "cid"])
        if len(g) < 3 or g.nE.nunique() < 2 and (g.n - g.nE).min() == 0:
            continue
        A = np.column_stack([g.n - g.nE, g.nE])
        if np.linalg.matrix_rank(A) < 2:
            continue
        a, b = np.linalg.lstsq(A, g.V.values, rcond=None)[0]
        recs.append({"date": d, "a": a, "b": b, "excess": b - a, "k": len(g)})
    xs = pd.DataFrame(recs).set_index("date")
    # realised by year
    fr = lr.copy()
    front = {}
    for d in days[(days >= "2014-01-01")]:
        row = px.loc[d].dropna()
        row = row[row.index >= (d + pd.offsets.BDay(4)).year * 100 + (d + pd.offsets.BDay(4)).month]
        front[d] = row.index[0] if len(row) else np.nan
    fs = pd.Series(front)
    # same-contract change into d for the contract that was front at d-1
    r_front = pd.Series({d: lr.at[d, fs.iloc[i - 1]] for i, d in enumerate(fs.index) if i > 0})
    real = pd.DataFrame({"r2": r_front ** 2, "rep": isR.reindex(r_front.index)})
    yr = []
    for y in range(2014, 2022):
        x = real[real.index.year == y]
        o1 = one[one.d1.dt.year == y]
        xb = xs[xs.index.year == y]
        yr.append({"year": y,
                   "real_rep_r2": x[x.rep].r2.mean(), "real_non_r2": x[~x.rep].r2.mean(),
                   "real_excess": x[x.rep].r2.mean() - x[~x.rep].r2.mean(),
                   "impl1_rep": o1[o1.nE == 1].V.mean(), "real1_rep": o1[o1.nE == 1].r2.mean(),
                   "n1_rep": int((o1.nE == 1).sum()),
                   "impl1_non": o1[o1.nE == 0].V.mean(), "real1_non": o1[o1.nE == 0].r2.mean(),
                   "n1_non": int((o1.nE == 0).sum()),
                   "implB_excess_med": xb.excess.median(), "implB_a_med": xb.a.median(), "nB": len(xb)})
    yr = pd.DataFrame(yr)
    # bootstrap realised excess CI (weekly blocks) 2014-16, 2017-21
    ci = {}
    for wn, (a, b) in (("2014-16", ("2014-01-01", "2016-12-31")), ("2017-21", ("2017-01-01", "2021-12-31"))):
        x = real[(real.index >= a) & (real.index <= b)].dropna()
        rng = np.random.default_rng(1)
        wk = x.index.to_period("W")
        groups = [x[wk == k] for k in wk.unique()]
        bs = []
        for _ in range(1000):
            s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            bs.append(s[s.rep].r2.mean() - s[~s.rep].r2.mean())
        ci[wn] = (x[x.rep].r2.mean() - x[~x.rep].r2.mean(), *np.percentile(bs, [2.5, 97.5]))
    # trade
    tr = trade(w, px, days, pos, isR)
    for k, v in (("weekly_rows", w), ("one_day", one), ("xsec", xs), ("by_year", yr), ("trades", tr)):
        v.to_csv(C.CACHE / f"part4_{k}.csv")
    pd.Series(ci).to_csv(C.CACHE / "part4_real_excess_ci.csv")
    fig(one, yr, tr)
    return w, one, xs, yr, tr, ci


def trade(w, px, days, pos, isR, cut=C.CUT, dh_df=None):
    raw = pd.read_csv(C.ROOT / "data/raw/options/ng_weekly_options_v2.csv.gz", parse_dates=["date"],
                      usecols=["date", "root", "cy", "cm", "cp", "k_raw", "settle"])
    raw = raw[raw.date <= cut]
    raw["series"] = raw.root + "_" + (raw.cy * 100 + raw.cm).astype(str)
    raw["K"] = raw.k_raw / 1000
    raw = raw.set_index(["date", "series"]).sort_index()
    dhd = C.dh() if dh_df is None else dh_df
    dhd = dhd[dhd["rank"] == 1].set_index("date")
    rows = []
    cand = w[(w.n <= 2)].sort_values(["date", "n"])
    for d, g in cand.groupby("date"):
        for kind in ("report", "none"):
            gg = g[g.nE >= 1] if kind == "report" else g[g.nE == 0]
            if gg.empty:
                continue
            r = gg.iloc[0]
            try:
                o = raw.loc[(d, r.series)]
            except KeyError:
                continue
            piv = o.pivot_table(index="K", columns="cp", values="settle").dropna()
            if piv.empty:
                continue
            K = piv.index[np.argmin(np.abs(piv.index - r.F))]
            prem = piv.loc[K].sum()
            if r.expiry not in px.index or r.cid not in px.columns or not np.isfinite(px.at[r.expiry, r.cid]):
                continue
            FE = px.at[r.expiry, r.cid]
            payoff = abs(FE - K)
            gross = (prem - payoff) * C.UNIT
            cost = (0.03 * prem + (C.FUT_COST if payoff > 0 else 0)) * C.UNIT
            dh_ = np.nan
            if kind == "report" and d in dhd.index:
                x = dhd.loc[d]
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[0]
                dh_ = (-x.pnl - 2 * C.OPT_HALF * x.P0 - C.FUT_COST * abs(x.gamma0 * (x.F1 - x.F0))) * C.UNIT
            rows.append({"date": d, "kind": kind, "series": r.series, "n": r.n, "nE": r.nE, "K": K, "F": r.F,
                         "prem": prem, "iv": r.atm_iv, "payoff": payoff, "gross": gross, "net": gross - cost,
                         "dh_monthly_short_net": dh_})
    return pd.DataFrame(rows)


def fig(one, yr, tr):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = yr.set_index("year")
    x = x[x.n1_rep >= 5]
    ax[0].bar(x.index - 0.2, np.sqrt(x.impl1_rep * 252), 0.4, color="#d1495b", label="implied, 1-day option spanning report")
    ax[0].bar(x.index + 0.2, np.sqrt(x.real1_rep * 252), 0.4, color="#1f6fb4", label="realised on those report days")
    ax[0].set_ylabel("annualised vol of the report day")
    ax[0].set_title("Report-day vol: implied (1-day weekly options) vs realised\n(years with >= 5 one-day report-spanning rows; none after 2016)", fontsize=9)
    ax[0].legend(fontsize=7)
    t = tr[tr.kind == "report"].sort_values("date")
    ax[1].plot(t.date, t.net.cumsum(), color="#d1495b", label=f"sell weekly straddle over report (n={len(t)})")
    t2 = tr[tr.kind == "none"].sort_values("date")
    ax[1].plot(t2.date, t2.net.cumsum(), color="#555555", label=f"same, no report in span (n={len(t2)})")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axvline(pd.Timestamp("2017-01-01"), color="k", ls=":", lw=0.8)
    ax[1].set_title("Cumulative net P&L, $ per straddle (3% cost)", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "v5_greeks_signals_eia_event.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    w, one, xs, yr, tr, ci = main()
    print(pd.crosstab(w.n.clip(upper=6), w.nE))
    print(yr.round(6).to_string())
    print("real excess CI", ci)
    print(xs.describe().round(6).to_string())
    for kind in ("report", "none"):
        t = tr[tr.kind == kind]
        for wn, (a, b) in (("2014-16", ("2014", "2016-12-31")), ("2017-21", ("2017", "2021-12-31"))):
            x = t[(t.date >= a) & (t.date <= b)]
            lo, hi = C.block_boot_ci(x.net.values, 4) if len(x) > 10 else (np.nan, np.nan)
            print(kind, wn, "n", len(x), "gross", round(x.gross.mean(), 1), "net", round(x.net.mean(), 1),
                  "CI", np.round([lo, hi], 1), "hit", round((x.net > 0).mean(), 2), "prem", round(x.prem.mean() * 1e4, 1),
                  "dh_monthly", round(x.dh_monthly_short_net.mean(), 1), "t", round(C.hac_mean_t(x.net, 2), 2))
