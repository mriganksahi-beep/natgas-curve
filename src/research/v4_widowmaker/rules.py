"""Position rules for the short March-April spread (v4_widowmaker).

pos[t] = spreads short held after the settle of day t (t = 0 is the entry settle). Daily P&L:
    pos[t-1] * -(S[t] - S[t-1]) * 10,000 - |pos[t] - pos[t-1]| * side_cost
Flat at the last settle (expiry, or 2021-12-31 for the truncated winter), paying the exit cost.

A rule produces a TARGET from information known at settle t. Forecast-only targets (the GEFS 00Z
run of day t is public ~06:00 ET) are executed at the settle of t (lag 0); lag 1 (next settle) is
the sensitivity. Targets that use the settle price of t (the add-on rule) are always lag >= 1.
NaN signals (fewer than 20 earlier season days) are treated as 'not cold'.
"""
import numpy as np

from src.research.v4_widowmaker.data import MULT


def pnl(w, pos, side):
    pos = np.asarray(pos, float).copy()
    pos[-1] = 0.0
    S = w.S.values
    dS = np.r_[0.0, np.diff(S)]
    prev = np.r_[0.0, pos[:-1]]
    return prev * -dS * MULT - np.abs(pos - prev) * side


def _lag(target, lag):
    target = np.asarray(target, float)
    if lag == 0:
        return target
    return np.r_[target[:1].repeat(lag), target[:-lag]]


def _sig(w, col):
    return w[col].values.astype(float)


def scale(w, f=1.0, lag=0):
    return np.full(len(w), f)


def cold_cut(w, sig="L23", k=5, th=1.0, f=0.0, lag=0):
    """Hold 1 spread; hold f while the k-day mean of the standardised forecast level > th."""
    z = _sig(w, f"z{sig}_{k}")
    cold = np.nan_to_num(z, nan=-9) > th
    return _lag(np.where(cold, f, 1.0), lag)


def news_cut(w, k=5, th=1.0, f=0.0, lag=0):
    """Hold f while the standardised k-day sum of news_tot (forecast revisions) > th."""
    z = _sig(w, f"zN_{k}")
    cold = np.nan_to_num(z, nan=-9) > th
    return _lag(np.where(cold, f, 1.0), lag)


def size3(w, k=5, th=1.0, lag=0):
    """1 if the forecast is warm/normal (z <= 0), 0.5 if 0 < z <= th, 0 if z > th (L23)."""
    z = np.nan_to_num(_sig(w, f"zL23_{k}"), nan=-9)
    return _lag(np.where(z <= 0, 1.0, np.where(z <= th, 0.5, 0.0)), lag)


def sizec(w, k=5, th=2.0, lag=0):
    """Continuous: 1 - max(z,0)/th, clipped to [0,1], rounded to quarter spreads (L23)."""
    z = np.nan_to_num(_sig(w, f"zL23_{k}"), nan=-9)
    t = np.clip(1 - np.maximum(z, 0) / th, 0, 1)
    return _lag(np.round(t * 4) / 4, lag)


def addon(w, X=0.15, add=0.5, hold=None, zmax=0.5, nmax=1.0, lag=1):
    """Sell the squeeze: base 1 spread; when S rose >= X over 2 settles and the forecast is not cold
    (zL23_5 <= zmax) and the 2-day forecast revision is not a large cold one (zN_2 <= nmax), add
    `add` spreads at the next settle; hold to expiry (hold=None) or for `hold` trading days.
    One add-on per winter."""
    S = w.S.values
    z = np.nan_to_num(_sig(w, "zL23_5"), nan=-9)
    n = np.nan_to_num(_sig(w, "zN_2"), nan=0)
    pos = np.ones(len(w))
    for t in range(2, len(w) - 1):
        if S[t] - S[t - 2] >= X and z[t] <= zmax and n[t] <= nmax:
            a = t + lag
            b = len(w) if hold is None else min(len(w), a + hold)
            pos[a:b] += add
            break
    return pos


def delay(w, th=0.0, lag=0):
    """Enter (1 spread) on the first day from the nominal entry with zL23_5 <= th; hold to expiry."""
    z = np.nan_to_num(_sig(w, "zL23_5"), nan=-9)
    ok = np.where(z <= th)[0]
    t = np.zeros(len(w))
    if len(ok):
        t[ok[0]:] = 1.0
    return _lag(t, lag)


def tight_news_cut(w, freq="w", th=1.0, f=0.0, lag=0, sign=1):
    """POST-HOC family G. Hold f while tightness-scaled forecast news (zNS_w: last Friday-to-Friday
    week, in force for the following week; zNS_d5: rolling 5 days) is > th sd."""
    z = sign * _sig(w, "zNS_w" if freq == "w" else "zNS_d5")
    cold = np.nan_to_num(z, nan=-9) > th
    return _lag(np.where(cold, f, 1.0), lag)


def hyst(w, k=5, th=1.0, lag=0):
    """POST-HOC family F. Flat when zL23_k > th; re-enter only once zL23_k <= 0 (normal or warm)."""
    z = np.nan_to_num(_sig(w, f"zL23_{k}"), nan=-9)
    t, on = np.zeros(len(w)), True
    for i in range(len(w)):
        if on and z[i] > th:
            on = False
        elif (not on) and z[i] <= 0:
            on = True
        t[i] = 1.0 if on else 0.0
    return _lag(t, lag)


def grid():
    """(family, name, fn, kwargs) - every variant scored."""
    G = [("bench", "base", scale, {"f": 1.0}), ("bench", "half", scale, {"f": 0.5})]
    for sig in ("L23", "L123"):
        for k in (1, 5):
            for th in (0.5, 1.0, 1.5):
                for f in (0.0, 0.5):
                    G.append(("A_cold_cut", f"A {sig} k{k} th{th} f{f}", cold_cut,
                              dict(sig=sig, k=k, th=th, f=f)))
    for k in (5, 10):
        for th in (1.0, 1.5):
            for f in (0.0, 0.5):
                G.append(("B_news_cut", f"B news k{k} th{th} f{f}", news_cut, dict(k=k, th=th, f=f)))
    for k in (1, 5):
        for th in (1.0, 1.5):
            G.append(("C_size", f"C3 k{k} th{th}", size3, dict(k=k, th=th)))
        for th in (1.0, 2.0):
            G.append(("C_size", f"Cc k{k} th{th}", sizec, dict(k=k, th=th)))
    for X in (0.15, 0.30):
        for add in (0.5, 1.0):
            G.append(("D_addon", f"D X{X} add{add}", addon, dict(X=X, add=add)))
    for th in (0.0, 0.5, 1.0):
        G.append(("E_delay", f"E th{th}", delay, dict(th=th)))
    # ---- post-hoc families, added after the first scoring of A-E (disclosed in the report)
    for k in (1, 5):
        for th in (1.0, 1.5):
            G.append(("F_hyst", f"F k{k} th{th}", hyst, dict(k=k, th=th)))
    for freq in ("w", "d5"):
        for th in (0.5, 1.0, 1.5):
            for f in (0.0, 0.5):
                G.append(("G_tight_news", f"G {freq} th{th} f{f}", tight_news_cut,
                          dict(freq=freq, th=th, f=f)))
    # control (not eligible): same rule on WARM tightness-scaled news (sign flipped)
    for th in (0.5, 1.0, 1.5):
        G.append(("Z_control", f"Gwarm w th{th} f0.0", tight_news_cut, dict(freq="w", th=th, f=0.0, sign=-1)))
    return G
