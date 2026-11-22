"""Reviewer v3 helpers. Every loader drops rows dated >= 2022-01-01 immediately after reading."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
MULT, RT = 10_000, 2 * 2 * 4 * 10.0  # $ per $1 of spread; $160 round trip per spread


def load_hj():
    """Daily H-J per delivery year Y (2011..2022), dates <= 2021-12-31."""
    parts = []
    for ch in pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"], chunksize=500_000,
                          usecols=["date", "contract_year", "contract_month", "settle"]):
        ch = ch[(ch.date <= CUT) & ch.contract_month.isin([3, 4])]
        parts.append(ch)
    c = pd.concat(parts).dropna(subset=["settle"])
    assert c.date.max() <= CUT
    px = c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle")
    out = {}
    for Y in range(2011, 2023):
        s = (px[(Y, 3)] - px[(Y, 4)]).dropna()
        out[Y] = s[s.index >= pd.Timestamp(Y - 1, 4, 1)]
    return out, px


def wboot(vals_by_group, B=4000, seed=0, stat=np.mean):
    """Bootstrap over groups; vals_by_group: dict g -> array. Returns (lo, hi) of the pooled mean."""
    rng = np.random.default_rng(seed)
    gs = list(vals_by_group)
    out = []
    for _ in range(B):
        pick = rng.choice(len(gs), len(gs))
        out.append(stat(np.concatenate([np.atleast_1d(vals_by_group[gs[i]]) for i in pick])))
    return np.quantile(out, [0.025, 0.975])
