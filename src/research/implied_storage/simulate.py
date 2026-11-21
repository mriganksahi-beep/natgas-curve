"""Step 1: Model 2's full end-March distribution for every winter origin week (Nov-Feb),
Nov 2016 - Dec 2021. Saves percentiles 1..99 of the simulated levels.

Holdout rule: only origins <= 2021-12-31. Simulator.simulate(t) uses only storage and
weather dated <= t. The `actual` outcome is kept only when the target week is <= 2021-12-31
(so winter 2021-22 has no outcome here).

Run: PYTHONPATH=. python -m src.research.implied_storage.simulate   (~1-2 min)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.model2 import Simulator, season_end

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/processed/research_implied_storage"
CUTOFF = pd.Timestamp("2021-12-31")
PCTS = np.arange(1, 100)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "model2_percentiles.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["origin", "target"])
    sim = Simulator()
    weeks = sim.df.index
    origins = [t for t in weeks if pd.Timestamp("2016-11-01") <= t <= CUTOFF and t.month in (11, 12, 1, 2)]
    rows = []
    for t in origins:
        r = sim.simulate(t)
        if r is None:
            continue
        q = np.percentile(r["levels"], PCTS)
        row = {"origin": t, "target": r["target"], "weeks_to_go": r["weeks_to_go"],
               "n_scenarios": r["n_scenarios"], "mean": float(r["levels"].mean()),
               "benchmark": r["benchmark"],
               "actual": r["actual"] if r["target"] <= CUTOFF else np.nan}
        row.update({f"p{p}": v for p, v in zip(PCTS, q)})
        rows.append(row)
        print(t.date(), r["target"].date(), round(q[49]), flush=True)
    df = pd.DataFrame(rows)
    assert df["origin"].max() <= CUTOFF
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = run()
    print(len(d), "origins")
