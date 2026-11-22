"""Spread definitions, spread values through time, and the drift panel.

A spread is a set of legs (offset in months from the anchor contract, weight); the anchor is the
first leg to expire. Drift at date t = V(t) - V(T):
  * two-leg spreads: T = expiry of the near (anchor) leg; V(T) = near final settle - far settle that day.
  * strip spreads: V(T) = weighted average of each leg's own final settle (the hold-to-expiry payoff);
    lead time is still measured to the anchor (first leg) expiry.
Positive drift = the spread falls into expiry = a short position earns it.
"""
import numpy as np
import pandas as pd

from .data import MNAME

SPREADS = {}
for m in range(12):                                   # adjacent pairs, near month m (0=Jan)
    SPREADS[f"{MNAME[m]}-{MNAME[(m + 1) % 12]}"] = dict(anchor_month=m, legs=[(0, 1.0), (1, -1.0)], strip=False)
SPREADS["Oct-Jan"] = dict(anchor_month=9, legs=[(0, 1.0), (3, -1.0)], strip=False)
# winter strip (Nov-Mar) minus the FOLLOWING summer strip (Apr-Oct): the winter scarcity premium
SPREADS["Win-nextSum"] = dict(anchor_month=10, legs=[(i, 1 / 5) for i in range(5)] + [(i, -1 / 7) for i in range(5, 12)], strip=True)
# winter strip minus the PRECEDING summer strip (Apr-Oct of the same year): the storage (space) spread
SPREADS["Win-prevSum"] = dict(anchor_month=3, legs=[(i, -1 / 7) for i in range(7)] + [(i, 1 / 5) for i in range(7, 12)], strip=True)

PAIR_ORDER = list(SPREADS)


def spread_values(W, expiry, name, anchors):
    """Matrix (date x anchor k) of spread values; NaN where any leg is missing.
    Expired strip legs are held at their final settle."""
    spec = SPREADS[name]
    dates = W.index
    out = {}
    gross = {}
    for a in anchors:
        v = np.zeros(len(dates))
        g = np.zeros(len(dates))
        ok = np.ones(len(dates), bool)
        T = expiry.get(a)
        for off, w in spec["legs"]:
            k = a + off
            if k not in W.columns:
                ok[:] = False
                break
            p = W[k].values.copy()
            if spec["strip"] and k in expiry.index:
                fin = W.loc[expiry[k], k]
                p[dates > expiry[k]] = fin
            v += w * np.nan_to_num(p)
            g += abs(w) * np.nan_to_num(p)
            ok &= ~np.isnan(p)
        if T is not None:
            ok &= dates <= T
        v[~ok] = np.nan
        g[~ok] = np.nan
        out[a] = v
        gross[a] = g / sum(abs(w) for _, w in spec["legs"])
    return pd.DataFrame(out, index=dates), pd.DataFrame(gross, index=dates)


def terminal_value(W, expiry, name, a):
    spec = SPREADS[name]
    T = expiry.get(a)
    if T is None:
        return np.nan
    if not spec["strip"]:
        return sum(w * W.loc[T, a + off] for off, w in spec["legs"])
    tot = 0.0
    for off, w in spec["legs"]:
        k = a + off
        if k not in expiry.index:          # some leg has not expired by the cutoff
            return np.nan
        tot += w * W.loc[expiry[k], k]
    return tot


def gas_year(k):
    """Gas year (Apr-Mar) of a delivery month index: Apr Y..Mar Y+1 -> Y."""
    y, m = k // 12, k % 12
    return y if m >= 3 else y - 1


def weekly_panel(W, expiry, fridays, max_lead=12):
    """One row per (spread, anchor contract, Friday t) with lead bucket 1..max_lead:
    drift ($), drift in % of the average leg price at t, the trailing 13-week realised vol of
    weekly spread changes (known at t), and the price of the front April contract at t."""
    fridays = pd.DatetimeIndex(fridays)
    # front April contract price at each Friday (April not yet expired)
    aprs = [k for k in W.columns if k % 12 == 3]
    apr_px = pd.Series(np.nan, index=fridays)
    for t in fridays:
        live = [k for k in aprs if (k not in expiry.index or expiry[k] > t) and k * 1 >= 0]
        live = [k for k in live if not np.isnan(W.at[t, k])] if live else []
        if live:
            apr_px[t] = W.at[t, min(live)]
    rows = []
    for name in PAIR_ORDER:
        spec = SPREADS[name]
        anchors = [k for k in expiry.index if k % 12 == spec["anchor_month"]]
        V, G = spread_values(W, expiry, name, anchors)
        Vf, Gf = V.loc[V.index.isin(fridays)], G.loc[G.index.isin(fridays)]
        rv = Vf.diff().rolling(13, min_periods=8).std()          # uses Fridays <= t only
        for a in anchors:
            VT = terminal_value(W, expiry, name, a)
            if np.isnan(VT):
                continue
            T = expiry[a]
            v = Vf[a].dropna()
            lead = (T - v.index).days / 30.4375
            L = np.round(lead).astype(int)
            df = pd.DataFrame({"t": v.index, "L": L, "V": v.values, "drift": v.values - VT,
                               "price": Gf.loc[v.index, a].values, "rv13": rv.loc[v.index, a].values,
                               "apr_px": apr_px.reindex(v.index).values})
            df["pct"] = 100 * df.drift / df.price
            df = df[(df.L >= 1) & (df.L <= max_lead)]
            df["pair"], df["anchor"], df["expiry"], df["VT"], df["gy"] = name, a, T, VT, gas_year(a)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def drift_panel(wk, extra=()):
    """Collapse the weekly panel to one row per (spread, anchor, L): means over the bucket's Fridays."""
    cols = ["drift", "pct", "V", "price", "rv13", "apr_px", *extra]
    agg = wk.groupby(["pair", "anchor", "L"]).agg(
        **{c: (c, "mean") for c in cols}, t=("t", "first"), n=("drift", "size"),
        expiry=("expiry", "first"), gy=("gy", "first"), VT=("VT", "first")).reset_index()
    return agg
