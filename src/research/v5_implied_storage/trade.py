"""Step 4: is selling the implied storage tail tradable?

Instruments (G4X March options, one contract = 10,000 MMBtu, held to the option expiry, cash value =
intrinsic on the H-J futures settle on the option-expiry day):
  CS  short call spread K1 = round_0.05(F + 0.25), K2 = round_0.05(F + 0.75)   (max loss $0.50 - credit)
  SC  short call K1 = round_0.05(F + 0.25)
Entry price: the day's settle at that strike (call, or put + parity) if quoted, else the fitted RND price
(flagged "model"). Cost: c x premium per option per side (c = 4%; 3% and 5% as sensitivities), at least one
tick ($0.001); hold to expiry, no exit cost.
Marks (for drawdowns): same rule each day (settle if quoted, else that day's fitted RND).
Futures benchmarks (v4): short 1 H-J at the first decision date of the winter to the futures LTD settle,
cost 2 ticks per leg per side ($80 round trip per spread); half size.
"""
import numpy as np
import pandas as pd

from src.research.v5_implied_storage import data as D, rnd as R

GRID = 0.05


def rnd5(x):
    return np.round(np.round(x / GRID) * GRID, 2)


def price_at(row, K):
    """Call price at strike K on a day (row of rnd_daily). Returns (price, quoted?)."""
    Kq, px, isc = row.quotes
    C = np.where(isc, px, px + (row.F - Kq))
    j = np.flatnonzero(np.isclose(Kq, K))
    if len(j):
        return float(C[j[0]]), True
    return float(R.mix_call(row.theta, [K])[0]), False


def option_trades(panel, rnd, cost=0.04):
    ex = D.expiries()
    rd = rnd.set_index(["date", "Y"])
    rows = []
    for _, r in panel.iterrows():
        q = rd.loc[(r.obs, r.Y)]
        K1, K2 = rnd5(q.F + 0.25), rnd5(q.F + 0.75)
        c1, q1 = price_at(q, K1)
        c2, q2 = price_at(q, K2)
        ST = ex.S_oexp[r.Y]
        pay1, pay2 = max(ST - K1, 0), max(ST - K2, 0)
        tc1, tc2 = max(cost * c1, 0.001), max(cost * c2, 0.001)
        # mark-to-market path (short positions): value of liability each day until expiry
        days = rnd[(rnd.Y == r.Y) & (rnd.date > r.obs)].sort_values("date")
        m1 = np.array([price_at(d, K1)[0] for _, d in days.iterrows()] + [pay1])
        m2 = np.array([price_at(d, K2)[0] for _, d in days.iterrows()] + [pay2])
        cs_path = (c1 - c2) - (m1 - m2) - (tc1 + tc2)
        sc_path = c1 - m1 - tc1
        rows.append({"obs": r.obs, "Y": r.Y, "window": r.window, "F": q.F, "K1": K1, "K2": K2,
                     "c1": c1, "c2": c2, "quoted1": q1, "quoted2": q2, "S_T": ST,
                     "pnl_cs": D.MULT * ((c1 - c2) - (tc1 + tc2) - (pay1 - pay2)),
                     "pnl_sc": D.MULT * (c1 - tc1 - pay1),
                     "dd_cs": D.MULT * min(0.0, cs_path.min()), "dd_sc": D.MULT * min(0.0, sc_path.min()),
                     "pnl_cs_gross": D.MULT * ((c1 - c2) - (pay1 - pay2)),
                     "pnl_sc_gross": D.MULT * (c1 - pay1),
                     "tc_cs": D.MULT * (tc1 + tc2), "tc_sc": D.MULT * tc1})
    return pd.DataFrame(rows)


def futures_trades(panel):
    fut = D.load_futures()
    rows = []
    for _, r in panel.iterrows():
        d = fut[r.Y]
        path = d.S.loc[r.obs:]
        pnl_path = (path.iloc[0] - path) * D.MULT - 80
        rows.append({"obs": r.obs, "Y": r.Y, "pnl_fut": float(pnl_path.iloc[-1]),
                     "dd_fut": float(min(0.0, pnl_path.min()))})
    return pd.DataFrame(rows)


def winter_boot(v, g, n=4000, seed=0):
    """Winter-block bootstrap CI of the pooled mean of v grouped by g."""
    v, g = np.asarray(v, float), np.asarray(g)
    u = np.unique(g)
    if len(u) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = {k: np.flatnonzero(g == k) for k in u}
    ms = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        ii = np.concatenate([idx[k] for k in pick])
        ms.append(v[ii].mean())
    return tuple(np.percentile(ms, [2.5, 97.5]))
