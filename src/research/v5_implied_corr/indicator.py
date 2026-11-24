"""Step 4: does implied H-J correlation lead realised decoupling, premium decay, or drawdowns? (ex ante only)

Sample: target-winter March-April (v4 panel hj, d_hj, dates <= 2021-12-31), trade season = first trading day
on/after the first Friday on/after 1 Nov .. 20 trading days before the H futures expiry; weekly points (every 5th
trading day of the season) on days with a valid (non-stale) H-J CSO row.
Signals at d:  rho     = rho_impl (R1 horizon-adjusted, capped at 1)
               drho5   = rho(d) - rho(last row in [d-10, d-5] trading days)
               rho_dev = rho(d) - walk-forward norm (median rho of all PAST winters' rows in the same tdays bucket)
Targets:       dec20   = 1 - rho_realised over the next 20 days (normal changes, zero mean)
               dhj20   = hj(d+20) - hj(d)       (premium decay: negative = decay)
               mae20   = max_{t in (d, d+20]} hj(t) - hj(d)   (adverse excursion for a short)
Benchmark:     hj level + trailing 20-day RV of d_hj (+ nv as a variant).  Expanding-window OLS refit each winter.
"""
import numpy as np
import pandas as pd

from .common import boot_blockmean, panel
from .crp import prep

BUCKETS = [0, 42, 84, 126, 400]


def build_weekly():
    s = prep()
    p = panel()[["date", "hj_year", "hj", "d_hj", "hj_days_left"]].dropna(subset=["hj"])
    p = p.sort_values("date").reset_index(drop=True)
    x = s[(s.cm == 3) & ~s.stale & s.rho_R1.notna() & (s.rho_R1 >= -1)].copy()
    x["hj_year"] = x.cid // 100
    x = x[["date", "hj_year", "rc_R1", "nv", "tdays", "rho_r20", "S"]]
    m = p.merge(x, on=["date", "hj_year"], how="left")
    out = []
    for y, g in m.groupby("hj_year"):
        g = g.reset_index(drop=True)
        hj = g.hj.values
        g["rv20"] = np.sqrt((g.d_hj ** 2).rolling(20, min_periods=15).mean() * 252)
        n = len(g)
        g["dhj20"] = [hj[i + 20] - hj[i] if i + 20 < n else np.nan for i in range(n)]
        g["mae20"] = [hj[i + 1:i + 21].max() - hj[i] if i + 20 < n else np.nan for i in range(n)]
        # season start: first trading day on/after first Friday on/after 1 Nov of y-1
        nov1 = pd.Timestamp(int(y) - 1, 11, 1)
        fri = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
        st = g.index[g.date >= fri]
        if not len(st):
            continue
        i0 = st[0]
        # 20-day lag of rho rows for drho
        rho = g.rc_R1.values
        dr = np.full(n, np.nan)
        for i in range(n):
            if np.isfinite(rho[i]):
                for k in range(5, 11):
                    if i - k >= 0 and np.isfinite(rho[i - k]):
                        dr[i] = rho[i] - rho[i - k]
                        break
        g["drho5"] = dr
        g["season"] = (g.index >= i0) & (g.hj_days_left > 20 * 1.4)
        g["season_full"] = g.index >= i0
        g["wk"] = ((g.index - i0) % 5 == 0) & g.season
        out.append(g)
    m = pd.concat(out, ignore_index=True)
    m["bucket"] = pd.cut(m.tdays, BUCKETS).astype(str)
    # walk-forward norm: past winters only
    norm = np.full(len(m), np.nan)
    for i, (y, b, r) in enumerate(zip(m.hj_year, m.bucket, m.rc_R1)):
        if np.isfinite(r):
            h = m.rc_R1[(m.hj_year < y) & (m.bucket == b)].dropna()
            if len(h) >= 20:
                norm[i] = r - h.median()
    m["rho_dev"] = norm
    m["dec20"] = 1 - m.rho_r20
    m["window"] = np.select([m.hj_year.between(2015, 2017), m.hj_year.between(2018, 2022)], ["choose", "check"], "pre")
    return m


def wf_r2(w, target, sig, bench=("hj", "rv20")):
    """Expanding-window OLS by winter; OOS SSE of bench vs bench+sig; returns per-window incremental R2."""
    cols = list(bench) + ([sig] if sig else [])
    d = w.dropna(subset=list(bench) + [target] + ([sig] if sig else []))
    preds_b, preds_s, ys, wins, blks = [], [], [], [], []
    for y in sorted(d.hj_year.unique()):
        tr, te = d[d.hj_year < y], d[d.hj_year == y]
        if tr.hj_year.nunique() < 2 or te.empty:
            continue
        Xb = np.c_[np.ones(len(tr)), tr[list(bench)].values]
        Xs = np.c_[np.ones(len(tr)), tr[cols].values]
        bb = np.linalg.lstsq(Xb, tr[target].values, rcond=None)[0]
        bs = np.linalg.lstsq(Xs, tr[target].values, rcond=None)[0]
        preds_b += list(np.c_[np.ones(len(te)), te[list(bench)].values] @ bb)
        preds_s += list(np.c_[np.ones(len(te)), te[cols].values] @ bs)
        ys += list(te[target].values)
        wins += list(te.window.values)
        blks += list(te.hj_year.values)
    r = pd.DataFrame({"y": ys, "pb": preds_b, "ps": preds_s, "window": wins, "blk": blks})
    out = {}
    for wname in ("choose", "check", "all"):
        q = r if wname == "all" else r[r.window == wname]
        if len(q) < 5:
            out[wname] = (np.nan, len(q))
            continue
        out[wname] = (1 - ((q.y - q.ps) ** 2).sum() / ((q.y - q.pb) ** 2).sum(), len(q))
    # bootstrap over winters of incremental R2 (all)
    g = r.assign(eb=(r.y - r.pb) ** 2, es=(r.y - r.ps) ** 2).groupby("blk")[["eb", "es"]].sum()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(4000, len(g)))
    inc = 1 - g.es.values[idx].sum(1) / g.eb.values[idx].sum(1)
    out["all_ci"] = (np.percentile(inc, 2.5), np.percentile(inc, 97.5))
    return out


def describe(m):
    w = m[m.wk & m.rc_R1.notna()]
    rows = []
    for sig in ("rc_R1", "drho5", "rho_dev"):
        for tgt in ("dec20", "dhj20", "mae20"):
            d = w.dropna(subset=[sig, tgt])
            # rank correlation by window and winter-mean of within-winter correlations
            per = d.groupby("hj_year").apply(lambda g: g[sig].corr(g[tgt], method="spearman") if len(g) >= 5 else np.nan,
                                             include_groups=False)
            mu, lo, hi = boot_blockmean(per.values)
            r = {"signal": sig, "target": tgt, "n": len(d), "spearman_pooled": d[sig].corr(d[tgt], method="spearman"),
                 "within_winter_mean": mu, "lo": lo, "hi": hi,
                 "choose_pooled": d[d.window == "choose"][sig].corr(d[d.window == "choose"][tgt], method="spearman"),
                 "check_pooled": d[d.window == "check"][sig].corr(d[d.window == "check"][tgt], method="spearman")}
            for bench in (("hj", "rv20"), ("hj", "rv20", "nv")):
                o = wf_r2(w, tgt, sig, bench)
                tag = "b2" if len(bench) == 2 else "b3"
                r.update({f"{tag}_inc_choose": o["choose"][0], f"{tag}_inc_check": o["check"][0],
                          f"{tag}_inc_all": o["all"][0], f"{tag}_ci_lo": o["all_ci"][0], f"{tag}_ci_hi": o["all_ci"][1],
                          f"{tag}_n": o["all"][1]})
            rows.append(r)
    return pd.DataFrame(rows)


def overlay(m, q=0.2, cost_side=40.0):
    """Short 1 H-J through the season; weekly decision.  Rule 'flat_low': go flat for the next 5 days when
    rho_dev < walk-forward q-quantile of past winters' rho_dev (market prices unusual decoupling);
    'flat_high' is the mirror.  Days without a fresh signal keep the last decision. $ per spread."""
    res = []
    for rule in ("always", "half", "flat_low", "flat_high", "flat_drop", "twin_nv_high", "twin_hj_high"):
        for y, g in m[m.season_full].groupby("hj_year"):
            g = g.reset_index(drop=True)
            past = m[(m.hj_year < y) & m.wk]
            thr_lo = past.rho_dev.quantile(q) if past.rho_dev.notna().sum() >= 10 else np.nan
            thr_hi = past.rho_dev.quantile(1 - q) if past.rho_dev.notna().sum() >= 10 else np.nan
            thr_dr = past.drho5.quantile(q) if past.drho5.notna().sum() >= 10 else np.nan
            pr = past[past.rho_dev.notna()]          # price-only twins: same days the rho signal exists
            thr_nv = pr.nv.quantile(1 - q) if len(pr) >= 10 else np.nan
            thr_hj = pr.hj.quantile(1 - q) if len(pr) >= 10 else np.nan
            pos = np.full(len(g), -1.0)
            cur = -1.0
            for i in range(len(g)):
                if g.wk.iat[i]:
                    rd, dr = g.rho_dev.iat[i], g.drho5.iat[i]
                    if rule == "flat_low":
                        cur = 0.0 if (np.isfinite(rd) and np.isfinite(thr_lo) and rd < thr_lo) else -1.0
                    elif rule == "flat_high":
                        cur = 0.0 if (np.isfinite(rd) and np.isfinite(thr_hi) and rd > thr_hi) else -1.0
                    elif rule == "twin_nv_high":
                        cur = 0.0 if (np.isfinite(rd) and g.nv.iat[i] > thr_nv) else -1.0
                    elif rule == "twin_hj_high":
                        cur = 0.0 if (np.isfinite(rd) and g.hj.iat[i] > thr_hj) else -1.0
                    elif rule == "flat_drop":
                        cur = 0.0 if (np.isfinite(dr) and np.isfinite(thr_dr) and dr < thr_dr) else -1.0
                pos[i] = cur
            if rule == "half":
                pos[:] = -0.5
            pnl = pos[:-1] * g.d_hj.values[1:] * 10_000
            trades = np.abs(np.diff(np.r_[0.0, pos[:-1], 0.0])).sum()
            cum = np.cumsum(np.r_[0.0, pnl]) - cost_side * np.abs(pos[0])
            dd = (cum - np.maximum.accumulate(cum)).min()
            res.append({"rule": rule, "hj_year": y, "pnl": np.nansum(pnl) - cost_side * trades, "maxdd": dd,
                        "days_flat": int((pos == 0).sum()), "window": g.window.iat[0]})
    return pd.DataFrame(res)
