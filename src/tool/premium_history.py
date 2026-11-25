"""March-April futures premium by winter: short H-J from the first trading day of December to the March option
expiry, $ per spread (10,000 MMBtu), no costs. EIA c3-c4 on the entry day for winters <= 2010, CME curve after.
S_T from src/tool/_map_data (same convention as the monitor). Writes data/processed/tool_live/premium_history.csv."""
import os
os.environ.setdefault("RND_DATA_CUTOFF", "2026-12-31")
import numpy as np
import pandas as pd
from src.tool import _map_data as MD

t = MD.winter_table()
e = MD.eia_futures()
c = MD.cme_futures()
rows = []
for Y in range(2001, 2027):
    d0 = pd.Timestamp(Y - 1, 12, 1)
    if Y >= 2011 and (Y, 3) in c.columns:
        s = (c[(Y, 3)] - c[(Y, 4)]).dropna()
        s = s[s.index >= d0]
        F, ent = float(s.iloc[0]), s.index[0]
    else:
        x = e[e.index >= d0].iloc[0]
        F, ent = float(x.c3 - x.c4), e[e.index >= d0].index[0]
    rows.append({"Y": Y, "entry": ent.date(), "F": F, "S_T": float(t.at[Y, "S"]), "x": float(t.at[Y, "x"])})
h = pd.DataFrame(rows)
h["pnl"] = (h.F - h.S_T) * 10_000
if __name__ == "__main__":
    os.makedirs("data/processed/tool_live", exist_ok=True)
    h.to_csv("data/processed/tool_live/premium_history.csv", index=False)
    print(h.round(3).to_string())
    print("positive", int((h.pnl > 0).sum()), "of", len(h), " mean $", round(h.pnl.mean()), " median $", round(h.pnl.median()))
