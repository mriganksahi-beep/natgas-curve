"""Position engine and overlays for a short 1 x March-April (H-J) spread.

Conventions
-----------
pos[t]  = spreads short held after the close of day t (0..1). P&L of day t:
          pos[t-1] * -(S[t]-S[t-1]) * 10,000  -  |pos[t]-pos[t-1]| * $80   (4 ticks x 2 contracts x $10)
The position is forced flat at the close of the last day (expiry, or 2021-12-31 for 2021-22), paying
the exit cost. Trigger-based exits (stops, vol, physical, level) are observed at the close of t and
executed at the close of t+1 (one-day lag, conservative). Scheduled changes (entry, calendar,
stagger, timed re-entry) execute at the scheduled close.
Hedge legs: long h units of the hedge instrument from entry to end, costs $80 per unit per side for a
spread hedge (2 contracts) and $40 for an outright.
"""
import numpy as np
import pandas as pd

from src.research.premium_risk_management.data import MULT, SIDE_COST


def _idx(w):
    live = w.live.values
    return int(np.argmax(live)), len(w) - 1


# ------------------------------------------------------------------ P&L
def pnl(w, pos, hedge=None, hedge_col=None, hedge_side_cost=SIDE_COST):
    pos = np.asarray(pos, float).copy()
    pos[-1] = 0.0
    S = w.S.values
    dS = np.r_[0.0, np.diff(S)]
    prev = np.r_[0.0, pos[:-1]]
    daily = prev * -dS * MULT - np.abs(pos - prev) * SIDE_COST
    if hedge is not None:
        hp = np.asarray(hedge, float).copy()
        hp[-1] = 0.0
        X = w[hedge_col].ffill().values
        dX = np.r_[0.0, np.diff(X)]
        hprev = np.r_[0.0, hp[:-1]]
        daily = daily + hprev * dX * MULT - np.abs(hp - hprev) * hedge_side_cost
    return pd.Series(daily, index=w.index)


def base(w):
    i0, _ = _idx(w)
    pos = np.zeros(len(w))
    pos[i0:] = 1.0
    return pos


# ------------------------------------------------------------------ (a)/(b) stops
def stop(w, X, reentry_weeks=None, trailing=False, lag=1):
    """Exit when the mark-to-market since the last entry is <= -X (fixed) or when it is X below its
    running peak since entry (trailing). Optional re-entry after reentry_weeks (new reference).
    lag=1: exit at the next settle (default); lag=0: exit at the trigger day's settle (sensitivity,
    an optimistic stand-in for an intraday stop order without gap slippage)."""
    S, dates = w.S.values, w.index
    i0, iend = _idx(w)
    pos = np.zeros(len(w))
    inpos, pending, re_date = False, False, None
    ref = peak = 0.0
    for t in range(i0, iend + 1):
        if t == i0:
            inpos, ref, peak = True, S[t], 0.0
        elif pending:
            inpos, pending = False, False
            re_date = dates[t] + pd.Timedelta(days=7 * reentry_weeks) if reentry_weeks else None
        elif (not inpos) and re_date is not None and dates[t] >= re_date:
            inpos, ref, peak, re_date = True, S[t], 0.0, None
        pos[t] = 1.0 if inpos else 0.0
        if inpos:
            m = -(S[t] - ref) * MULT
            peak = max(peak, m)
            if (trailing and m - peak <= -X) or ((not trailing) and m <= -X):
                if lag == 0 and t < iend:
                    pos[t], inpos = 0.0, False
                    re_date = dates[t] + pd.Timedelta(days=7 * reentry_weeks) if reentry_weeks else None
                else:
                    pending = True
    return pos


# ------------------------------------------------------------------ signal-driven exits
def _from_signal(w, off, reenter):
    """off[t] True = 'be flat' observed at close t -> applied at close t+1."""
    i0, iend = _idx(w)
    off = np.asarray(off, bool)
    if not reenter:
        o = off.copy()
        o[:i0] = False
        off = np.maximum.accumulate(o)
    pos = np.zeros(len(w))
    for t in range(i0, iend + 1):
        pos[t] = 0.0 if (t > i0 and off[t - 1]) else 1.0
    return pos


def realised_vol(w, n=10):
    return w.S.diff().rolling(n, min_periods=n).std()


def vol_trigger(w, k, reenter):
    """Flat when trailing 10-day (2-week) realised vol of daily dS > k x its level on the entry day
    (the 2 weeks ending at the November entry)."""
    i0, _ = _idx(w)
    rv = realised_vol(w).values
    ref = rv[i0]
    return _from_signal(w, rv > k * ref, reenter)


def phys_trigger(w, alarm, m, reenter):
    """Flat when the PHYS-M2 ES alarm ratio (ES / median earlier ES), forecast on Friday d, exceeds m.
    The Friday value is held until the next forecast. NaN alarm = no signal."""
    a = alarm.reindex(w.index, method="ffill", limit=7) if len(alarm) else pd.Series(np.nan, index=w.index)
    a = a.where(w.index >= alarm.index.min()) if len(alarm) else a
    return _from_signal(w, (a > m).fillna(False).values, reenter)


def level_trigger(w, L, reenter):
    """Price-only twin of the physical trigger: flat when the spread itself exceeds $L/MMBtu."""
    return _from_signal(w, (w.S > L).values, reenter)


# ------------------------------------------------------------------ (e) staggered entry
def stagger(w, n, offset_weeks):
    """n equal tranches (1/n spread), one per week, the first offset_weeks after the base entry."""
    i0, iend = _idx(w)
    e = w.index[i0]
    pos = np.zeros(len(w))
    for i in range(n):
        d = e + pd.Timedelta(days=7 * (offset_weeks + i))
        k = w.index.searchsorted(d)
        if k <= iend:
            pos[k:iend + 1] += 1.0 / n
    return pos


# ------------------------------------------------------------------ (g) calendar
def calendar(w, Y, start_md, weeks):
    """Flat in [start, start + weeks) where start is (month, day) in winter Y (Nov/Dec -> Y-1)."""
    m, d = start_md
    st = pd.Timestamp(Y - 1 if m >= 11 else Y, m, d)
    en = st + pd.Timedelta(days=7 * weeks)
    pos = base(w)
    pos[(w.index >= st) & (w.index < en)] = 0.0
    return pos


# ------------------------------------------------------------------ (f) hedge
def hedge_leg(w, units):
    return base(w) * units
