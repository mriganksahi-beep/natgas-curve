"""Step 1: Model 2's end-October storage distribution for every summer origin week
(April-October), 2017-2021. Saves percentiles 1..99 of the simulated levels.

Holdout rule: origins <= 2021-12-31 only (summers 2017-2021 end in Oct 2021).
Simulator.simulate(t) uses only storage and weather dated <= t.

Run: PYTHONPATH=. python -m src.research.storage_space_option.simulate   (~3-4 min, 1 CPU)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.model2 import Simulator

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/processed/research_storage_space_option"
CUTOFF = pd.Timestamp("2021-12-31")
PCTS = np.arange(1, 100)


def run(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "model2_summer_percentiles_dev.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["origin", "target"])
    sim = Simulator(n_resid_paths=200)
    weeks = sim.df.index
    origins = [t for t in weeks if pd.Timestamp("2017-04-01") <= t <= CUTOFF and 4 <= t.month <= 10]
    rows = []
    for t in origins:
        r = sim.simulate(t)
        if r is None:
            continue
        q = np.percentile(r["levels"], PCTS)
        row = {"origin": t, "target": r["target"], "weeks_to_go": r["weeks_to_go"],
               "n_scenarios": r["n_scenarios"], "mean": float(r["levels"].mean()),
               "sd": float(r["levels"].std()), "benchmark": r["benchmark"],
               "actual": r["actual"] if r["target"] <= CUTOFF else np.nan}
        row.update({f"p{p}": v for p, v in zip(PCTS, q)})
        rows.append(row)
        print(t.date(), r["target"].date(), r["weeks_to_go"], round(q[49]), round(r["benchmark"]), flush=True)
    df = pd.DataFrame(rows)
    assert df["origin"].max() <= CUTOFF
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = run()
    print(len(d), "origins")
