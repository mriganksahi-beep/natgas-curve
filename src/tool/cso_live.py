"""Live spread-option richness reading for the monitor (added 25 Nov 2026, after the gate run; the frozen
monitor.py is untouched). This is the measure that passed its locked test (study v5, test P):
ln(implied normal vol of the March-April calendar-spread option / normal vol the spread then realised).

Live, the realised half is in the future, so the reading shows:
  * today's at-the-money implied normal vol of the H-J spread option (Bachelier, from the option settles);
  * the realised normal vol of the spread over the last 20 trading days (what it has been doing);
  * the same mid-November comparison in past winters: implied vs vol realised from then to expiry.
Note: the v5 panel (src/v5/options_panel.py) sets expiry to the last settle date; for contracts not yet
expired this makes T far too short, so live T is taken from rnd_direct.trading_T instead. Expired contracts
(all that the v5 locked test used) are unaffected.
"""
import os

import numpy as np
import pandas as pd

from src.tool import rnd_direct as RA


def atm_normal_vol(date, Y):
    d = pd.Timestamp(date)
    day, _ = RA._day_rows(d, Y, d)
    if not len(day):
        return None
    F, T = RA.parity_F(day), RA.trading_T(d, Y)
    c = day[day.cp == "C"].groupby("K").settle.first().sort_index()
    p = day[day.cp == "P"].groupby("K").settle.first().sort_index()
    C, P = np.interp(F, c.index, c.values), np.interp(F, p.index, p.values)
    return {"date": d.date(), "F": float(F), "T": float(T), "nv": float((C + P) / 2 * np.sqrt(2 * np.pi) / np.sqrt(T))}


def trailing_realised(date, Y, n=20):
    fut = RA.load_futures()
    S = fut.loc[Y].S
    S = S[S.index <= pd.Timestamp(date)].tail(n + 1)
    dS = np.diff(S.values)
    return float(np.sqrt(np.mean(dS ** 2) * 252))


if __name__ == "__main__":
    os.environ.setdefault("RND_DATA_CUTOFF", "2026-12-31")
    d = os.environ.get("AS_OF", "2026-11-17")
    print(atm_normal_vol(d, 2027), "trailing 20d realised", round(trailing_realised(d, 2027), 3))
