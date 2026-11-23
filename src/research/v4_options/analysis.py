"""v4_options analyses (steps 1-5). Every input is cut at 2021-12-31 by data.py.

Step 1  spread / outright variance risk premium (implied vs subsequently realised vol)
Step 2  episodes (lead/lag of implied vol around the v3 blow-ups) + walk-forward prediction
Step 3  cost of protection: short H-J naked / half / + long G4X March calls
Step 4  risk-neutral tail probabilities (Breeden-Litzenberger) vs physical frequencies
Step 5  news_channel decomposition vs implied-vol jumps (only if that agent's file exists)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.v4_options import data as D

RNG = np.random.default_rng(7)
NB = 4000
FUT_SIDE = 2 * 2 * D.TICK      # 2 legs x 2 ticks x $10 = $40 per spread per side
FUT_SIDE4 = 2 * 4 * D.TICK     # 4-tick sensitivity ($80, the v3 convention)
OPT_HS = 0.04                  # half-spread on G4X options = 4% of premium (brief: 3-5%)
OPT_HS_MIN = 0.002             # ... but at least $0.002/MMBtu ($20 per option)


def boot_mean(v, nb=NB):
    """Winter-block bootstrap of the mean of per-winter values."""
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) < 2:
        return (np.nan, np.nan)
    b = RNG.choice(v, size=(nb, len(v)), replace=True).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def winter_frames():
    """dict Y -> daily frame with S, dS, rH, rJ, panel vols, realised vols (forward and trailing)."""
    sp, p = D.spread_frame(), D.panel()
    px = D.futures()
    days = px.index
    out = {}
    for Y in D.YEARS:
        s = sp[Y].copy()
        s = s[s.index >= pd.Timestamp(Y - 1, 4, 1)]      # <= ~11 months out (earlier settles stale)
        s["dS"] = s.S.diff()
        s["rH"] = np.log(s.H).diff()
        s["rJ"] = np.log(s.J).diff()
        pp = p[p.hj_year == Y]
        for c in ("nv_hj", "iv_h", "iv_j"):
            s[c] = pp[c].reindex(s.index)
        oe = s.attrs["opt_exp"]
        # J options expire a month later: their own T
        fJ = days[days.searchsorted(pd.Timestamp(Y, 4, 1)) - 3] if pd.Timestamp(Y, 4, 1) <= days[-1] else None
        live = s[s.index <= oe]
        complete = oe <= D.CUTOFF
        for nm, col in (("S", "dS"), ("H", "rH"), ("J", "rJ")):
            x2 = live[col] ** 2
            # forward realised over remaining option life (sum of squares t+1..expiry) / T
            fwd = x2[::-1].cumsum()[::-1].shift(-1)
            if nm == "J":
                s[f"rv_life_{nm}"] = np.nan
            else:
                s[f"rv_life_{nm}"] = np.sqrt(fwd / live["T"]).reindex(s.index) if complete else np.nan
            f20 = x2.rolling(20).sum().shift(-20)
            s[f"rv20_{nm}"] = np.sqrt(f20 * 252 / 20).reindex(s.index)
            s[f"rvtr20_{nm}"] = np.sqrt(x2.rolling(20).sum() * 252 / 20).reindex(s.index)
        if fJ is not None and Y <= 2021:      # J forward realised to J option expiry (late March)
            jj = px[Y * 100 + 4]
            jj = jj[(jj.index >= s.index[0]) & (jj.index <= days[days.searchsorted(fJ) - 1])]
            r2 = np.log(jj).diff() ** 2
            fwd = r2[::-1].cumsum()[::-1].shift(-1)
            TJ = (jj.index[-1] - jj.index).days / 365.25
            s["rv_life_J"] = np.sqrt(fwd / TJ).reindex(s.index)
        s.attrs.update(sp[Y].attrs)
        out[Y] = s
    return out


# ----------------------------------------------------------------------------- step 1
def step1(W):
    rows = []
    for Y, s in W.items():
        ent, oe = s.attrs["entry"], s.attrs["opt_exp"]
        s = s[s.index <= oe]
        mo = s["T"] * 12
        for iv, rvl, rv20, nm in (("nv_hj", "rv_life_S", "rv20_S", "HJ"), ("iv_h", "rv_life_H", "rv20_H", "H"),
                                  ("iv_j", "rv_life_J", "rv20_J", "J")):
            d = pd.DataFrame({"Y": Y, "inst": nm, "date": s.index, "iv": s[iv].values, "rv_life": s[rvl].values,
                              "rv20": s[rv20].values, "months": mo.values, "season": (s.index >= ent)})
            rows.append(d)
    d = pd.concat(rows)
    d["bucket"] = pd.cut(d.months, [0, 1, 2, 4, 7, 11], labels=["<1m", "1-2m", "2-4m", "4-7m", "7-11m"], right=False)
    d.to_csv(D.CACHE / "step1_daily.csv", index=False)

    def summ(sub, rvcol):
        sub = sub.dropna(subset=["iv", rvcol])
        if sub.empty:
            return None
        pw = sub.groupby("Y").apply(lambda g: pd.Series({"iv": g.iv.mean(), "rv": g[rvcol].mean(),
                                                         "gap": (g.iv - g[rvcol]).mean(), "n": len(g),
                                                         "share_pos": (g.iv > g[rvcol]).mean()}))
        lo, hi = boot_mean(pw.gap.values)
        return {"winters": len(pw), "days": int(pw.n.sum()), "iv_mean": pw.iv.mean(), "rv_mean": pw.rv.mean(),
                "gap": pw.gap.mean(), "gap_lo": lo, "gap_hi": hi, "ratio": pw.iv.mean() / pw.rv.mean(),
                "winters_pos": int((pw.gap > 0).sum()), "share_days_pos": (sub.iv > sub[rvcol]).mean(),
                "gap_sd_across_winters": pw.gap.std()}

    res = []
    win = {"choose": D.CHOOSE, "check": D.CHECK, "all": D.YEARS, "all_ex2013_14": D.YEARS[2:]}
    for inst in ("HJ", "H", "J"):
        for rvcol in ("rv_life", "rv20"):
            for wn, ys in win.items():
                for per in ("season", "all_T"):
                    sub = d[(d.inst == inst) & d.Y.isin(ys)]
                    if per == "season":
                        sub = sub[sub.season]
                    r = summ(sub, rvcol)
                    if r:
                        res.append({"inst": inst, "rv": rvcol, "window": wn, "period": per, **r})
            for b in d.bucket.cat.categories:
                r = summ(d[(d.inst == inst) & (d.bucket == b)], rvcol)
                if r:
                    res.append({"inst": inst, "rv": rvcol, "window": "all", "period": f"T {b}", **r})
    res = pd.DataFrame(res)
    res.to_csv(D.CACHE / "step1_summary.csv", index=False)
    # per-winter season table
    q = d[d.season & d.iv.notna()].assign(gap20=lambda x: x.iv - x.rv20, gapL=lambda x: x.iv - x.rv_life)
    pw = q.groupby(["inst", "Y"]).agg(iv=("iv", "mean"), n=("iv", "size"), rv_life=("rv_life", "mean"),
                                      gap_life=("gapL", "mean"), rv20=("rv20", "mean"), gap20=("gap20", "mean"))
    pw.to_csv(D.CACHE / "step1_by_winter.csv")
    return d, res, pw


# ----------------------------------------------------------------------------- step 2
EPISODES = {"Jan 2014 polar vortex": (2014, "2014-01-02", "2014-01-31"),
            "Feb 2014 expiry squeeze": (2014, "2014-02-01", "2014-02-24"),
            "Nov 2018 squeeze": (2019, "2018-10-15", "2018-12-31"),
            "Sep-Nov 2021 (pre-entry)": (2022, "2021-09-01", "2021-11-30")}


def atm_smile_vol(Y, s, opts):
    """Own ATM normal vol from the smile, with a flag for F inside the quoted strike range."""
    sm = D.Smile(Y, opts, s)
    v, inside = {}, {}
    for d in s.index:
        if d in sm.by_date:
            ks = sm.by_date[d][0]
            v[d] = float(np.interp(s.at[d, "S"], ks, sm.by_date[d][1]))
            inside[d] = ks.min() <= s.at[d, "S"] <= ks.max()
    return pd.Series(v), pd.Series(inside), sm


def step2_episodes(W, opts):
    rows, paths = [], []
    for name, (Y, a, b) in EPISODES.items():
        s = W[Y]
        nv_own, inside, _ = atm_smile_vol(Y, s, opts)
        s = s.assign(nv_own=nv_own.reindex(s.index), inside=inside.reindex(s.index))
        w = s[(s.index >= a) & (s.index <= b)]
        rise5 = w.S.shift(-5) - w.S
        t0 = rise5.idxmax()
        i0 = s.index.get_loc(t0)
        g = lambda k, c: s[c].iloc[i0 + k] if 0 <= i0 + k < len(s) else np.nan
        # typical 20-day |change| of nv over all G4X seasons (for scale)
        r = {"episode": name, "t0": t0.date(), "rise_5d_$": rise5.max(),
             "rise_5d_$per_spread": rise5.max() * D.MULT}
        s["nv_best"] = s.nv_hj.fillna(s.nv_own)
        for c in ("nv_best", "iv_h", "rvtr20_S"):
            for k in (-20, -5, -1, 5):
                r[f"{c}@{k}"] = g(k, c)
        r["dS_pre(-20..-1)"] = g(-1, "S") - g(-20, "S")
        r["dS_post(-1..+5)"] = g(5, "S") - g(-1, "S")
        r["nv_pre_chg"] = g(-1, "nv_best") - g(-20, "nv_best")
        r["nv_post_chg"] = g(5, "nv_best") - g(-1, "nv_best")
        r["ivh_pre_chg"] = g(-1, "iv_h") - g(-20, "iv_h")
        r["ivh_post_chg"] = g(5, "iv_h") - g(-1, "iv_h")
        r["inside_share_window"] = float(w.inside.fillna(False).mean())
        r["nv_quoted_days_window"] = int(w.nv_hj.notna().sum())
        r["days_window"] = len(w)
        rows.append(r)
        seg = s.iloc[max(0, i0 - 30): i0 + 21].copy()
        seg["k"] = np.arange(len(seg)) - min(30, i0)
        seg["episode"] = name
        paths.append(seg[["k", "episode", "S", "nv_hj", "nv_own", "nv_best", "inside", "iv_h", "rvtr20_S"]])
    ep = pd.DataFrame(rows)
    ep.to_csv(D.CACHE / "step2_episodes.csv", index=False)
    pa = pd.concat(paths)
    pa.to_pickle(D.CACHE / "step2_paths.pkl")
    # scale: distribution of 20-day and 5-day changes in nv_hj over all season days (all winters)
    ch = []
    for Y, s in W.items():
        x = s.nv_hj[s.index >= s.attrs["entry"]]
        ch.append(pd.DataFrame({"d20": x.diff(20), "d5": x.diff(5), "lvl": x}))
    ch = pd.concat(ch)
    scale = {"sd_d20": ch.d20.std(), "sd_d5": ch.d5.std(), "p90_lvl": ch.lvl.quantile(.9), "med_lvl": ch.lvl.median()}
    # lead/lag: corr(d nv_t, dS_{t+k}) and corr(d nv_t, |dS_{t+k}|), season days, all winters
    ll = []
    for Y, s in W.items():
        x = s[s.index >= s.attrs["entry"]].copy()
        x["dnv"] = x.nv_hj.diff()
        for k in range(-5, 6):
            ll.append(pd.DataFrame({"k": k, "Y": Y, "dnv": x.dnv, "dS": x.dS.shift(-k), "adS": x.dS.shift(-k).abs()}))
    ll = pd.concat(ll).dropna()
    lead = ll.groupby("k").apply(lambda g: pd.Series({"corr_dS": g.dnv.corr(g.dS), "corr_absdS": g.dnv.corr(g.adS),
                                                      "n": len(g)}))
    lead.to_csv(D.CACHE / "step2_leadlag.csv")
    return ep, pa, scale, lead


def step2_predict(W, opts):
    """Weekly (Friday) points from entry to 20 trading days before option expiry; walk-forward OLS."""
    pts = []
    for Y, s in W.items():
        oe = s.attrs["opt_exp"]
        sm = D.Smile(Y, opts, s)
        live = s[(s.index >= s.attrs["entry"]) & (s.index <= oe)]
        idx = live.index
        for i, d in enumerate(idx):
            if d.weekday() != 4 or i + 20 >= len(idx):
                continue
            fut = live.S.iloc[i + 1: i + 21]
            F = live.S.iloc[i]
            r = {"Y": Y, "date": d, "S": F, "nv": live.nv_hj.iloc[i], "rvtr": live.rvtr20_S.iloc[i],
                 "mae20": fut.max() - F, "dS20": fut.iloc[-1] - F, "rv20": live.rv20_S.iloc[i]}
            r["gap"] = r["nv"] - r["rvtr"]
            if d in sm.by_date:
                ks = sm.by_date[d][0]
                c0 = sm.call(d, F, count=False)
                for dl in (0.25, 0.50):
                    r[f"skew{int(dl*100)}"] = (sm.call(d, F + dl, count=False) / c0
                                               if ks.min() <= F + dl <= ks.max() and ks.min() <= F and c0 > 0 else np.nan)
            pts.append(r)
    P = pd.DataFrame(pts)
    P.to_csv(D.CACHE / "step2_weekly.csv", index=False)

    def wf(target, xs, bench):
        """Expanding-window OLS by winter. Returns per-row predictions for x-model, benchmark, mean."""
        out = []
        for Y in D.YEARS:
            tr = P[(P.Y < Y)].dropna(subset=[target] + xs + bench)
            te = P[(P.Y == Y)].dropna(subset=[target] + xs + bench)
            if tr.Y.nunique() < 2 or te.empty or len(tr) < 15:
                continue

            def fitpred(cols):
                X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols])
                b = np.linalg.lstsq(X, tr[target].values, rcond=None)[0]
                Xt = np.column_stack([np.ones(len(te))] + [te[c] for c in cols])
                return Xt @ b
            out.append(pd.DataFrame({"Y": Y, "y": te[target].values, "mean": tr[target].mean(),
                                     "bench": fitpred(bench) if bench else tr[target].mean(),
                                     "model": fitpred(bench + xs)}))
        return pd.concat(out) if out else pd.DataFrame()

    res = []
    for target in ("mae20", "dS20", "rv20"):
        for xs in (["nv"], ["gap"], ["skew25"], ["skew50"], ["S"], ["rvtr"]):
            for bench in ([], ["S", "rvtr"]):
                if xs[0] in bench:
                    continue
                o = wf(target, xs, bench)
                if o.empty:
                    continue
                for wn, ys in (("choose", D.CHOOSE), ("check", D.CHECK)):
                    q = o[o.Y.isin(ys)]
                    if q.empty:
                        continue
                    sse = ((q.y - q.model) ** 2).sum()
                    ref = ((q.y - q.bench) ** 2).sum()
                    # winter-block bootstrap of the R2 difference
                    ws = q.Y.unique()
                    bs = []
                    for _ in range(1000):
                        pick = RNG.choice(ws, len(ws))
                        qq = pd.concat([q[q.Y == w] for w in pick])
                        bs.append(1 - ((qq.y - qq.model) ** 2).sum() / ((qq.y - qq.bench) ** 2).sum())
                    res.append({"target": target, "x": "+".join(xs), "bench": "+".join(bench) or "mean",
                                "window": wn, "n": len(q), "winters": len(ws), "oos_r2_vs_bench": 1 - sse / ref,
                                "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                                "corr": q.y.corr(q.model)})
    res = pd.DataFrame(res)
    res.to_csv(D.CACHE / "step2_predict.csv", index=False)
    return P, res


# ----------------------------------------------------------------------------- step 3
VARIANTS = {"naked": None, "half": None, "call+0.25": (0.25, None), "call+0.50": (0.50, None),
            "cs+0.25/+1.00": (0.25, 1.00)}


def kround(x):
    return round(round(x / 0.05) * 0.05, 3)


def step3(W, opts, hs=OPT_HS, fut_side=FUT_SIDE):
    rows, paths, marks = [], {}, {"exact": 0, "interp": 0, "carry": 0}
    for Y, s in W.items():
        ent, fe, oe = s.attrs["entry"], s.attrs["fut_exp"], s.attrs["opt_exp"]
        live = s[(s.index >= ent) & (s.index <= min(fe, D.CUTOFF))]
        sm = D.Smile(Y, opts, s)
        F0 = live.S.iloc[0]
        fut = -(live.S - F0) * D.MULT                  # short 1 spread, MTM before costs
        fut = fut - fut_side                            # entry cost
        complete = fe <= D.CUTOFF
        if complete:
            fut.iloc[-1] -= fut_side                    # exit at expiry settle
        P = {"naked": fut, "half": 0.5 * fut}
        info = {"Y": Y, "entry": ent.date(), "F0": F0, "complete": complete}
        for vn, spec in VARIANTS.items():
            if spec is None:
                continue
            legs = [(kround(F0 + spec[0]), +1)] + ([(kround(F0 + spec[1]), -1)] if spec[1] else [])
            val = pd.Series(0.0, index=live.index)
            for K, sgn in legs:
                c = []
                for d in live.index:
                    if d >= oe:
                        c.append(max(live.S.loc[oe] - K, 0.0) if oe in live.index else np.nan)
                    else:
                        c.append(sm.call(d, K))
                c = pd.Series(c, index=live.index).ffill()
                p0 = c.iloc[0]
                cost = max(hs * p0, OPT_HS_MIN)
                val += sgn * (c - p0) * D.MULT - cost * D.MULT
                info[f"{vn}_K{'+' if sgn > 0 else '-'}"] = K
                info[f"{vn}_prem{'+' if sgn > 0 else '-'}"] = p0
            P[vn] = fut + val
        marks["exact"] += sm.n_exact
        marks["interp"] += sm.n_interp
        marks["carry"] += sm.n_carry
        for vn, x in P.items():
            dd = (x - x.cummax().clip(lower=0)).min()     # peak-to-trough, peak floored at 0 (entry)
            info[f"{vn}_pnl"] = x.iloc[-1]
            info[f"{vn}_dd"] = min(dd, x.min(), 0.0)
        rows.append(info)
        paths[Y] = pd.DataFrame(P)
    t = pd.DataFrame(rows).set_index("Y")
    return t, paths, marks


def step3_summary(t):
    out = []
    for wn, ys in (("choose", D.CHOOSE), ("check", D.CHECK), ("all", D.YEARS), ("all_ex2014", [y for y in D.YEARS if y != 2014])):
        q = t.loc[[y for y in ys if y in t.index]]
        qc = q[q.complete]
        base_m, base_dd = qc.naked_pnl.mean(), q.naked_dd.min()
        for vn in VARIANTS:
            m = qc[f"{vn}_pnl"].mean()
            lo, hi = boot_mean(qc[f"{vn}_pnl"].values)
            dlo, dhi = boot_mean((qc[f"{vn}_pnl"] - qc.naked_pnl).values)
            dd = q[f"{vn}_dd"].min()
            kept, cut = m / base_m, 1 - dd / base_dd
            out.append({"window": wn, "variant": vn, "winters_pnl": len(qc), "winters_dd": len(q), "mean": m,
                        "lo": lo, "hi": hi, "diff_vs_naked_lo": dlo, "diff_vs_naked_hi": dhi,
                        "worst_dd": dd, "worst_winter": qc[f"{vn}_pnl"].min(),
                        "kept": kept, "cut": cut, "kept_plus_cut": kept + cut,
                        "calmar": m / abs(dd) if dd < 0 else np.nan})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- step 4
def step4(W, opts):
    rows = []
    for Y, s in W.items():
        ent, oe = s.attrs["entry"], s.attrs["opt_exp"]
        sm = D.Smile(Y, opts, s)
        idx = s.index[s.index >= ent][:5]
        ST = s.S.loc[oe] if oe <= D.CUTOFF and oe in s.index else np.nan
        path = s.S[(s.index > ent) & (s.index <= oe)]
        F0 = s.S.loc[ent]
        r = {"Y": Y, "F0": F0, "ST": ST, "nv0": s.nv_hj.loc[idx].mean(), "path_max": path.max()}
        h = 0.05
        for lab, kfun in (("abs0.50", lambda F: 0.50), ("abs1.00", lambda F: 1.00),
                          ("rel+0.25", lambda F: F + 0.25), ("rel+0.50", lambda F: F + 0.50),
                          ("rel+1.00", lambda F: F + 1.00)):
            pr, inside = [], []
            for d in idx:
                if d not in sm.by_date:
                    continue
                F = s.at[d, "S"]
                K = kfun(F)
                ks = sm.by_date[d][0]
                pr.append((sm.call(d, K - h, False) - sm.call(d, K + h, False)) / (2 * h))
                inside.append(ks.min() <= K - h and K + h <= ks.max())
            r[f"q_{lab}"] = np.mean(pr) if pr else np.nan
            r[f"inside_{lab}"] = np.mean(inside) if inside else np.nan
            K0 = kfun(F0)
            r[f"p_{lab}"] = float(ST > K0) if np.isfinite(ST) else np.nan
            r[f"hit_{lab}"] = float(path.max() > K0) if len(path) else np.nan
        # call buyer's payoff / price at entry, by strike
        for dl in (0.0, 0.25, 0.50, 1.00):
            K = kround(F0 + dl)
            c0 = sm.call(ent, K, False) if ent in sm.by_date or sm.dates else np.nan
            r[f"c0_{dl}"] = c0
            r[f"pay_{dl}"] = max(ST - K, 0) if np.isfinite(ST) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Y")
    t.to_csv(D.CACHE / "step4_by_winter.csv")
    # physical frequencies over ALL dev winters with expiries (2011-2021): futures only
    sp = D.spread_frame()
    phys = []
    for Y in range(2011, 2022):
        s = sp[Y]
        F0, ST = s.S.loc[s.attrs["entry"]], s.S.loc[s.attrs["opt_exp"]]
        mx = s.S[(s.index > s.attrs["entry"]) & (s.index <= s.attrs["opt_exp"])].max()
        phys.append({"Y": Y, "F0": F0, "ST": ST, "max": mx})
    phys = pd.DataFrame(phys).set_index("Y")
    phys.to_csv(D.CACHE / "step4_physical_2011_2021.csv")
    return t, phys


# ----------------------------------------------------------------------------- step 5
def step5(W):
    f = D.ROOT / "data/processed/research_v4_news_channel/decomposed_dev.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date"])
    d = d[(d.date <= D.CUTOFF) & (d.instrument == "d_hj")].dropna(subset=["news_part", "premium_part"])
    rows = []
    for Y, s in W.items():
        x = s[["nv_hj", "iv_h"]].copy()
        x["dnv"] = x.nv_hj.diff()
        x["divh"] = x.iv_h.diff()
        x["Y"] = Y
        rows.append(x)
    v = pd.concat(rows).dropna(subset=["dnv"])
    m = d.merge(v.reset_index(), on="date")
    m = m[m.winter == m.Y]                       # same target spread
    m["core"] = m.in_core.astype(bool)
    out = {}
    for lab, q in (("all", m), ("core Nov-Feb", m[m.core]), ("choose", m[m.Y.isin(D.CHOOSE)]),
                   ("check", m[m.Y.isin(D.CHECK)])):
        if len(q) < 20:
            continue

        def stats(qq):
            X = np.column_stack([np.ones(len(qq)), qq.news_part, qq.premium_part])
            b = np.linalg.lstsq(X, qq.dnv.values, rcond=None)[0]
            return {"b_news": b[1], "b_prem": b[2], "corr_news": qq.dnv.corr(qq.news_part),
                    "corr_prem": qq.dnv.corr(qq.premium_part),
                    "corr_abs_news": qq.dnv.abs().corr(qq.news_part.abs()),
                    "corr_abs_prem": qq.dnv.abs().corr(qq.premium_part.abs()),
                    "sd_news": qq.news_part.std(), "sd_prem": qq.premium_part.std()}
        st = stats(q)
        ws = q.Y.unique()
        bs = []
        for _ in range(2000):
            qq = pd.concat([q[q.Y == w] for w in RNG.choice(ws, len(ws))])
            r = stats(qq)
            bs.append((r["b_prem"] - r["b_news"], r["corr_prem"] - r["corr_news"]))
        bs = np.array(bs)
        st.update({"n": len(q), "winters": len(ws),
                   "bdiff_lo": np.nanpercentile(bs[:, 0], 2.5), "bdiff_hi": np.nanpercentile(bs[:, 0], 97.5),
                   "cdiff_lo": np.nanpercentile(bs[:, 1], 2.5), "cdiff_hi": np.nanpercentile(bs[:, 1], 97.5)})
        out[lab] = st
    # jump days: top 5% |dnv| days -- share of |d_hj| explained by news vs premium
    j = m[m.dnv.abs() >= m.dnv.abs().quantile(0.95)]
    out["jump_days"] = {"n": len(j), "mean_abs_news": j.news_part.abs().mean(),
                        "mean_abs_prem": j.premium_part.abs().mean(),
                        "share_same_sign_prem": float((np.sign(j.premium_part) == np.sign(j.dnv)).mean()),
                        "share_same_sign_news": float((np.sign(j.news_part) == np.sign(j.dnv)).mean())}
    out["other_days"] = {"mean_abs_news": m.news_part.abs().mean(), "mean_abs_prem": m.premium_part.abs().mean()}
    res = pd.DataFrame(out).T
    res.to_csv(D.CACHE / "step5_news_vs_iv.csv")
    return res
