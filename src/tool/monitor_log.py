"""Write the public weekly log of the squeeze-premium monitor (reports/monitor/README.md).
Only derived readings are published; the licensed option and futures prices stay out of the repository.
Run after new option data have been collected:  RND_DATA_CUTOFF=2026-12-31 python -m src.tool.monitor_log"""
import os
os.environ.setdefault("RND_DATA_CUTOFF", "2026-12-31")
from pathlib import Path
import pandas as pd
from src.tool import cso_live as C

r = pd.read_csv("data/processed/tool_live/readings.csv")
h = pd.read_csv("data/processed/tool_live/cso_midnov_history.csv")
rows = []
for x in r.itertuples():
    v = C.atm_normal_vol(x.as_of, 2027)
    rows.append(f"| {x.as_of} | ${v['nv']:.2f} | ${C.trailing_realised(x.as_of, 2027):.2f} | {x.Q_def400:.0%} | {x.P_def400_raw:.0%} | {x.Q_squeeze:.0%} |")
past = (h.nv > h.rv).sum()
text = f"""# Squeeze-premium monitor: weekly readings, winter 2026–27

This page records the monitor described in section 5 of the [paper](../../README.md). Each row is one reading, taken from that day's settlement prices of the March–April 2027 spread options.

| Date | Movement implied by options (per year) | Movement in the past 20 trading days (per year) | Options: chance of a 400+ Bcf shortfall | Physical model: same chance | Options: chance March settles 25¢+ above April |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

**How to read the table.**
- The second and third columns compare the price of protection with the movement it protects against. In mid-November of past winters, the implied movement exceeded the movement that followed in {past} of {len(h)} cases.
- The fourth and fifth columns compare the market's odds of a large shortfall at the end of March with the odds given by a model of storage and weather. A shortfall means end-March storage falling 400 billion cubic feet or more below its five-year average.
- The options' odds failed their pre-registered test on 2022–2026 ([result](../tool/GATE_RESULT.md)). They show what the market is pricing, not a forecast of storage.

**Updates.** The option prices are licensed and are collected in a separate step, so new readings are added when fresh data are collected, rather than automatically. The weekly storage forecast is fully automated and is recorded in [`reports/live_forecasts.csv`](../live_forecasts.csv).
"""
Path("reports/monitor").mkdir(parents=True, exist_ok=True)
Path("reports/monitor/README.md").write_text(text)
print(text)
