"""Score every candidate option-market signal (parts 1-3) in tradable terms, walk-forward.

For each (signal, target, horizon):
  z_t = walk-forward z of the signal (expanding, monthly refit)
  direction = sign of the choose-window (2010-2016) slope of the forward target on z   [in-sample choice]
  position = direction * sign(z_t), overlapping h-day hold; P&L net of costs (see common.py)
  'incremental' regression: fwd target on z + r_front(t) + r_front(t-4..t) (+ atm1, past rv for
  option targets), HAC lags = max(5, h), run separately on choose and check windows.
Benchmarks: zero; and the 'futures-only' rule (same machinery with the past 1-day / 5-day front
return as the signal; direction also chosen in-sample).
Multiple testing: Holm over all option-signal variants on the choose-window P&L t-stat (two-sided),
then the CHECK window with the frozen direction (one-sided p < 0.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.v5_greeks_signals import common as C
from src.research.v5_greeks_signals.features import build

SIGNALS = {
    "greek": ["vshock1", "vshock123", "vshock123_5", "vega_pnl123", "resid_rel123"],
    "surface": ["d_level", "d_term", "d_skew1", "d_skew2", "d_curv1", "d_level_res", "d_term_res",
                "d_skew1_res", "d_skew2_res", "d_curv1_res", "skew1", "skew2", "term", "curv1",
                "skew1_ch5", "pc1", "pc2", "pc3"],
    "flow": ["pc_oi", "d_pc_oi5", "pc_vol", "call_spike", "put_spike", "spike_diff"],
    "bench": ["r_front", "r_front_5", "gt_rel123"],
}
TARGETS = [("front", 1), ("front", 5), ("second", 1), ("second", 5), ("hj", 1), ("hj", 5),
           ("straddle", 1), ("straddle", 5)]


def target_series(f: pd.DataFrame, tgt: str, h: int) -> pd.Series:
    if tgt == "straddle":   # row t = P&L t -> t+1
        return f.st_pnl[::-1].rolling(h, min_periods=h).sum()[::-1]
    col = {"front": "dF_front", "second": "dF_second", "hj": "d_hj"}[tgt]
    return C.fwd_sum(f[col].fillna(0), h)


def pnl_for(f: pd.DataFrame, pos: pd.Series, tgt: str) -> pd.Series:
    if tgt == "straddle":
        st = f[["st_pnl", "st_P0", "st_gamma0", "st_dF", "st_switch"]].rename(
            columns={"st_pnl": "pnl", "st_P0": "P0", "st_gamma0": "gamma0", "st_dF": "dF"}).fillna(0)
        p = C.straddle_pnl(pos, st)
        # rolling a held straddle to the next contract = close + open
        p -= pos.shift(1).abs().fillna(0) * f.st_switch.fillna(0) * 2 * C.OPT_HALF * f.st_P0.fillna(0) * C.UNIT
        return p.shift(1).fillna(0)   # book at t+1 like futures
    col, roll = {"front": ("dF_front", "roll_front"), "second": ("dF_second", "roll_second")}[tgt]
    return C.futures_pnl(pos, f[col], f[roll])


def hj_pnl(pos, f):
    pos = pos.reindex(f.index).fillna(0)
    gross = pos.shift(1) * f.d_hj.fillna(0) * C.UNIT
    cost = pos.diff().abs().fillna(pos.abs()) * 2 * C.FUT_COST * C.UNIT + \
        pos.shift(1).abs() * f.roll_hj * 4 * C.FUT_COST * C.UNIT
    return (gross - cost).fillna(0)


def in_window(x, w):
    return x[(x.index >= w[0]) & (x.index <= w[1])]


def run():
    f = build()
    f["r_front_5"] = f.r_front.rolling(5).sum()
    f["rv_past5"] = (f.r_front ** 2).rolling(5).sum()
    zs = {}
    for fam, sigs in SIGNALS.items():
        for s in sigs:
            zs[s] = C.wf_z(f[s])
    rows, pnls = [], {}
    for fam, sigs in SIGNALS.items():
        for s in sigs:
            z = zs[s]
            for tgt, h in TARGETS:
                y = target_series(f, tgt, h)
                lags = max(5, h)
                b, t, n = C.hac_ols(in_window(y, C.CHOOSE), in_window(z, C.CHOOSE).to_frame("z"), lags)
                direction = np.sign(b[1]) if np.isfinite(b[1]) else 1.0
                for rule in ("sign", "thr1"):
                    raw = direction * np.sign(z).fillna(0)
                    if rule == "thr1":
                        raw = raw.where(z.abs() > 1, 0.0)
                    pos = C.overlap_pos(raw, h)
                    pnl = hj_pnl(pos, f) if tgt == "hj" else pnl_for(f, pos, tgt)
                    with C.no_costs():
                        gross = hj_pnl(pos, f) if tgt == "hj" else pnl_for(f, pos, tgt)
                    rec = {"family": fam, "signal": s, "target": tgt, "h": h, "rule": rule,
                           "dir": direction, "ic_choose_t": t[1]}
                    rec.update(C.summarise(pnl, lags))
                    g = C.summarise(gross, lags)
                    rec.update({f"gross_{k}": v for k, v in g.items() if k.endswith(("_ann", "_t"))})
                    rec["turnover"] = pos.diff().abs().mean()
                    rows.append(rec)
                    pnls[(s, tgt, h, rule)] = pnl
                rec = rows[-1]
                # incremental information over the futures path
                ctrl = pd.DataFrame({"z": z, "r1": zs["r_front"], "r5": zs["r_front_5"]})
                if tgt == "straddle":
                    ctrl["atm1"] = f.atm1
                    ctrl["rvp"] = f.rv_past5
                for wname, w in (("choose", C.CHOOSE), ("check", C.CHECK)):
                    bb, tt, nn = C.hac_ols(in_window(y, w), in_window(ctrl, w), lags)
                    rec[f"inc_{wname}_b"] = bb[1]
                    rows[-1][f"inc_{wname}_t"] = tt[1]
                    rows[-2][f"inc_{wname}_t"] = tt[1]
                    rows[-1][f"inc_{wname}_b"] = bb[1]
                    rows[-2][f"inc_{wname}_b"] = bb[1]
    res = pd.DataFrame(rows)
    # benchmarks for comparison: always-short straddle / always-long futures
    always = {}
    one = pd.Series(1.0, index=f.index)
    always["short_straddle"] = pnl_for(f, -one, "straddle")
    always["long_front"] = pnl_for(f, one, "front")
    for k, v in always.items():
        rr = {"family": "always", "signal": k, "target": k.split("_")[1], "h": 0, "dir": 1}
        rr.update(C.summarise(v, 5))
        with C.no_costs():
            gv = pnl_for(f, -one if k == "short_straddle" else one, k.split("_")[1])
        rr.update({f"gross_{kk}": vv for kk, vv in C.summarise(gv, 5).items() if kk.endswith(("_ann", "_t"))})
        rows.append(rr)
        pnls[(k, "always", 0)] = v
    res = pd.DataFrame(rows)
    opt = res.family.isin(["greek", "surface", "flow"])
    # tradable test: one-sided (net P&L > 0) on the choose window, Holm over all option variants
    p1 = norm.sf(res.loc[opt, "choose_t"])
    res.loc[opt, "p_choose"] = p1
    res.loc[opt, "holm_choose"] = C.holm(np.asarray(p1))
    # information test: incremental slope (two-sided), Holm over signal x target x h (rule-free)
    info = res[opt & (res.rule == "sign")]
    pi = 2 * norm.sf(info.inc_choose_t.abs())
    res.loc[info.index, "p_info"] = pi
    res.loc[info.index, "holm_info"] = C.holm(np.asarray(pi))
    res["p_check_1s"] = norm.sf(res.check_t)
    res.to_csv(C.CACHE / "scores.csv", index=False)
    pd.to_pickle(pnls, C.CACHE / "pnls.pkl")
    return res


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250)
    cols = ["family", "signal", "target", "h", "rule", "dir", "gross_choose_ann", "gross_check_ann", "choose_ann", "choose_sr", "choose_t", "check_ann",
            "check_sr", "check_t", "inc_choose_t", "inc_check_t", "holm_choose"]
    print(r.sort_values("choose_t", ascending=False)[cols].head(30).round(2).to_string())
    print(r[r.family.isin(["bench", "always"])][cols].round(2).to_string())
    opt = r.family.isin(["greek", "surface", "flow"])
    print("tradable variants", opt.sum(), "holm<0.05:", (r.holm_choose < 0.05).sum(),
          "raw p<0.05 choose:", (r.p_choose < 0.05).sum(),
          "of which check p1s<0.05:", ((r.p_choose < 0.05) & (r.p_check_1s < 0.05)).sum())
    i = r[r.p_info.notna()]
    print("info variants", len(i), "holm<0.05:", (i.holm_info < 0.05).sum(), "raw<0.05:", (i.p_info < 0.05).sum(),
          "raw & check same sign |t|>1.645:", ((i.p_info < 0.05) & (np.sign(i.inc_check_t) == np.sign(i.inc_choose_t))
                                               & (i.inc_check_t.abs() > 1.645)).sum())
