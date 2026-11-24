"""Q3/Q4: tradable test.  Long (or short) delta-hedged front ATM straddle switched by a weekly signal.

Instrument.  'Front' = nearest ON contract with >= 10 trading days to option expiry at the open.
Engine (fixed strike, no daily re-striking):
  * Weekly decision days: first trading day of each week inside the Oct-Apr weather_unc sample, from
    Oct 2011 (winter 2010-11 is burn-in for walk-forward thresholds).  Signal uses data known at the
    decision-day settle; trade at that settle.
  * Target position in {-1, 0, +1} unit vega.  If the target equals the current position and the held
    contract still has >= 10 trading days, the SAME straddle is kept (no cost).  Otherwise close
    (exit half-spread) and, if the target is non-zero, open a new ATM straddle (K = F) on the front.
  * The last week of each winter is closed at most 5 trading days after the last decision day.
  * Marks: Black-76 at the day's surface smile interpolated in ln(K/F) (vol_rv's validated pricer);
    delta-hedged daily at the settle.  P&L in vol pts per unit vega at entry (x $vega/vol-pt = $).
  * Costs: option half-spread c * premium per side (c = 1.5% base; 1% / 2% sensitivity), minimum
    $0.002 per option; futures 2 ticks ($0.002/MMBtu) x |change in hedge| incl. initial hedge and unwind.
Signals (walk-forward percentiles = expanding over all earlier sample days, so no look-ahead):
  see RULES below.  Benchmarks: always long, always short, vol_rv storage-sign (long if sdev<0 else
  short), storage tight-only (long if sdev < -150).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from src.research.v5_vol_rv.trade import black_straddle, smile_iv

START = pd.Timestamp("2011-10-01")
TMIN = 10


def wf_pct(x: pd.Series, q):
    """Walk-forward expanding quantile using strictly earlier observations."""
    return x.expanding(min_periods=100).quantile(q).shift(1)


def signals():
    w = D.weather_unc()
    P = pd.read_csv(D.CACHE / "q1_preds.csv", parse_dates=["date"])
    P = P[(P.h == 20) & (P.leg == 1)].set_index("date")
    b = D.base()
    b = b[(b.h == 20) & (b.leg == 1)].set_index("date")
    S = pd.DataFrame(index=w.index)
    for c in ("unc_n_w2", "unc_n_w3", "unc_n_w23"):
        x = w[c + "_m3"]
        S[c] = x
        S[c + "_p67"] = wf_pct(x, 0.67)
        S[c + "_p50"] = wf_pct(x, 0.50)
    S["rise"] = w["d5l_unc_n_w23"]
    S["sdev"] = b.sdev.reindex(S.index)
    S["gap"] = (np.log(b.atm_iv) - P["p_HAR"]).reindex(S.index)          # IV vs HAR forecast (20d)
    S["gapU"] = (np.log(b.atm_iv) - P["p_HAR+UL"]).reindex(S.index)      # IV vs HAR+disagreement forecast
    for g in ("gap", "gapU"):
        S[g + "_p33"] = wf_pct(S[g].dropna(), 0.33).reindex(S.index)
        S[g + "_p50"] = wf_pct(S[g].dropna(), 0.50).reindex(S.index)
    return S


def _f(x):
    return x.fillna(False).astype(bool)


RULES = {
    # benchmarks
    "always_long": lambda S: pd.Series(1, index=S.index),
    "always_short": lambda S: pd.Series(-1, index=S.index),
    "storage_sign": lambda S: np.sign(-S.sdev).fillna(0),
    "storage_tight": lambda S: (S.sdev < -150).astype(int),
    # disagreement rules (long when on, flat otherwise)
    "U2_high": lambda S: _f(S.unc_n_w2 > S.unc_n_w2_p67).astype(int),
    "U3_high": lambda S: _f(S.unc_n_w3 > S.unc_n_w3_p67).astype(int),
    "U23_high": lambda S: _f(S.unc_n_w23 > S.unc_n_w23_p67).astype(int),
    "U23_rising": lambda S: _f(S.rise > 0).astype(int),
    "U23_high_ls": lambda S: np.where(_f(S.unc_n_w23 > S.unc_n_w23_p67), 1, -1),
    "U23_high_rising": lambda S: _f((S.unc_n_w23 > S.unc_n_w23_p50) & (S.rise > 0)).astype(int),
    # IV - forecast-RV gap
    "gap_low": lambda S: _f(S.gap < S.gap_p33).astype(int),
    "gapU_low": lambda S: _f(S.gapU < S.gapU_p33).astype(int),
    "gap_low_x_U": lambda S: _f((S.gap < S.gap_p50) & (S.unc_n_w23 > S.unc_n_w23_p50)).astype(int),
    # Q4 interaction
    "tight_x_U": lambda S: _f((S.sdev < -150) & (S.unc_n_w23 > S.unc_n_w23_p50)).astype(int),
    "tight_x_notU": lambda S: _f((S.sdev < -150) & ~(S.unc_n_w23 > S.unc_n_w23_p50)).astype(int),
}
BENCH = ["always_long", "always_short", "storage_sign", "storage_tight"]


class Market:
    def __init__(self):
        s = D.surface()
        px, days, oexp = D.futures()
        self.days = days[days <= D.CUT]
        self.px = px
        self.oexp = oexp
        self.sx = s.set_index(["date", "cid"]).sort_index()
        self.td = s.pivot_table(index="date", columns="cid", values="tdays")
        self.memo = {}

    def front(self, d):
        t = self.td.loc[d].dropna()
        t = t[t >= TMIN]
        return None if t.empty else int(t.index.min())

    def segment(self, cid, K, iv0, t0, t1):
        """Long 1 straddle at strike K held t0 -> t1 (or option expiry), daily hedged.
        Returns dates (t0+1..end), daily pnl $/MMBtu, P0, Pend, expired, deltas De (per path day), end date."""
        key = (cid, K, t0, t1)
        if key in self.memo:
            return self.memo[key]
        e = self.oexp[cid]
        end = min(t1, e)
        path = self.days[(self.days >= t0) & (self.days <= end)]
        F = self.px.loc[path, cid].values
        V, De = np.empty(len(path)), np.empty(len(path))
        iv = iv0
        for j, t in enumerate(path):
            ntd = self.days.searchsorted(e) - self.days.searchsorted(t)
            if ntd <= 0:
                V[j], De[j] = abs(F[j] - K), np.sign(F[j] - K)
                continue
            if (t, cid) in self.sx.index:
                v = smile_iv(self.sx.loc[(t, cid)], np.log(K / F[j]))
                iv = v if np.isfinite(v) else iv
            V[j], De[j], _ = black_straddle(F[j], K, ntd / 252.0, iv)
        daily = np.diff(V) - De[:-1] * np.diff(F)
        out = dict(dates=path[1:], daily=daily, P0=V[0], Pend=V[-1], expired=bool(path[-1] == e),
                   De=De.copy(), end=path[-1])
        self.memo[key] = out
        return out


def decision_days(mk, S):
    d = S.index[(S.index >= START) & (S.index <= D.CUT)]
    wk = d.to_period("W")
    first = pd.Series(d, index=d).groupby(wk).first().values
    return pd.DatetimeIndex(first)


def simulate(mk, S, pos_series, c=0.015, fut=0.002):
    """Return daily DataFrame: date, gross, cost (vol pts per unit vega, sign included), pos, and trade log."""
    dd = decision_days(mk, S)
    pos_t = pos_series.reindex(dd).fillna(0).astype(int).values
    days = mk.days
    recs = []        # (date, gross, cost, pos)
    trades = []
    held = None      # dict(cid, K, iv0, vega0, sign, t_open)

    for i, r in enumerate(dd):
        nxt = dd[i + 1] if i + 1 < len(dd) else None
        ir = days.get_loc(r)
        hard_end = days[min(ir + 5, len(days) - 1)]
        t1 = nxt if (nxt is not None and (nxt - r).days <= 9) else hard_end
        tgt = pos_t[i]
        f = mk.front(r)
        keep = (held is not None and tgt == held["sign"] and
                mk.td.loc[r].get(held["cid"], 0) >= TMIN and not held.get("closed", False))
        if not keep:
            if held is not None and not held.get("closed", False):
                cst = max(c * held["Pcur"], 0.004) if not held["expired"] else 0.0
                cst += fut * abs(held["dcur"])            # unwind hedge
                recs.append((r, 0.0, -cst * 100.0 / held["vega0"], 0))
            held = None
            if tgt != 0 and f is not None and (r, f) in mk.sx.index:
                row = mk.sx.loc[(r, f)]
                K = float(row.F)
                P0, d0, _ = black_straddle(K, K, (days.searchsorted(mk.oexp[f]) - ir) / 252.0, row.atm_iv)
                held = dict(cid=f, K=K, iv0=float(row.atm_iv), vega0=float(row.vega), sign=int(tgt), t_mark=r,
                            t_open=r, expired=False)
                cst = max(c * P0, 0.004) + fut * abs(d0)
                recs.append((r, 0.0, -cst * 100.0 / held["vega0"], 0))
                trades.append((r, f, tgt))
        if held is not None:
            seg = mk.segment(held["cid"], held["K"], held["iv0"], held["t_mark"], t1)
            scale = held["sign"] * 100.0 / held["vega0"]
            for dt, p in zip(seg["dates"], seg["daily"]):
                recs.append((dt, p * scale, 0.0, held["sign"]))
            # hedge re-balances inside the segment, incl. the re-hedge at its last settle (initial hedge
            # charged at open, unwind at close)
            hc = fut * np.abs(np.diff(seg["De"])).sum()
            if hc:
                recs.append((seg["end"], 0.0, -hc * 100.0 / held["vega0"], held["sign"]))
            # state for continuation
            held["t_mark"] = seg["end"]
            held["Pcur"] = seg["Pend"]
            held["expired"] = seg["expired"]
            held["dcur"] = seg["De"][-1]
            if seg["expired"]:
                held["closed"] = True
                held = None
            elif t1 == hard_end and (nxt is None or (nxt - r).days > 9):
                cst = max(c * held["Pcur"], 0.004) + fut * abs(held["dcur"])
                recs.append((seg["end"], 0.0, -cst * 100.0 / held["vega0"], 0))
                held = None
    df = pd.DataFrame(recs, columns=["date", "gross", "cost", "pos"])
    df = df.groupby("date").agg(gross=("gross", "sum"), cost=("cost", "sum"), pos=("pos", "last")).reset_index()
    df["net"] = df.gross + df.cost
    return df, pd.DataFrame(trades, columns=["date", "cid", "pos"])


def weekly_outcomes(mk, S):
    """Fresh long front straddle opened at each decision day, held to the next (<= 5 tdays):
    gross vol pts per unit vega and round-trip cost; for the on/off selection test (no path dependence)."""
    dd = decision_days(mk, S)
    rows = []
    for i, r in enumerate(dd):
        ir = mk.days.get_loc(r)
        nxt = dd[i + 1] if i + 1 < len(dd) else None
        t1 = nxt if (nxt is not None and (nxt - r).days <= 9) else mk.days[min(ir + 5, len(mk.days) - 1)]
        f = mk.front(r)
        if f is None or (r, f) not in mk.sx.index:
            continue
        row = mk.sx.loc[(r, f)]
        seg = mk.segment(f, float(row.F), float(row.atm_iv), r, t1)
        g = seg["daily"].sum() * 100 / row.vega
        # realised vs implied over the week
        rv = np.sqrt(252 * np.mean(np.diff(np.log(mk.px.loc[mk.days[(mk.days >= r) & (mk.days <= t1)], f].values)) ** 2))
        rows.append({"date": r, "cid": f, "gross": g, "iv": row.atm_iv * 100, "rv": rv * 100,
                     "P0": seg["P0"], "vega": row.vega})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mk = Market()
    S = signals()
    df, tr = simulate(mk, S, RULES["always_long"](S))
    print(df.describe())
    print(df.groupby(D.window(df.date))[["gross", "cost", "net"]].sum())
