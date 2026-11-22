"""Data for premium_risk_management. Every loader cuts at 2021-12-31 (hard rule: no holdout).

Winter Y = the March (H) / April (J) contracts of delivery year Y; Nov Y-1 .. Feb Y.
S   = H_Y - J_Y           ($/MMBtu), the traded spread.
N   = H_{Y+1} - J_{Y+1}   next year's March-April spread (hedge candidate).
HO  = H_Y outright        (hedge candidate).
Entry = settle of the first trading day on/after the first Friday >= 1 Nov of Y-1.
End   = last H_Y settle (expiry) for Y <= 2021; for Y = 2022 the last settle <= 2021-12-31
        (winter 2021-22 is truncated; the 2022 expiry is never read).
PHYS-M2 ES: the scenario_risk h=1 forecasts (src/research/scenario_risk/models.run_all, cached),
        ES97.5 = mean of the loss quantile function above 0.975, made on Friday d with the
        settle of d and Model 2 origin d-7.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_premium_risk_management"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports/agents/figures"
MULT = 10_000.0          # $ per $1/MMBtu per spread
SIDE_COST = 2 * 4 * 10.0  # 2 contracts x 4 ticks x $10 = $80 per spread per side
YEARS = list(range(2011, 2023))
TRAIN = list(range(2011, 2018))   # winters 2010-11 .. 2016-17
TEST = list(range(2018, 2023))    # winters 2017-18 .. 2021-22 (truncated)
PRE_DAYS = 30                     # trading days kept before entry (vol reference)


def wlabel(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


def load_px():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & c.contract_month.isin([3, 4]) & (c.contract_year <= 2023)]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    assert px.index.max() <= CUTOFF
    return px


def winters():
    """dict Y -> daily DataFrame (index date) with S, N, HO, flag 'live' (entry..end)."""
    f = CACHE / "winters.pkl"
    if f.exists():
        return pd.read_pickle(f)
    px = load_px()
    out = {}
    for Y in YEARS:
        h, j = px[(Y, 3)], px[(Y, 4)]
        n = px[(Y + 1, 3)] - px[(Y + 1, 4)]
        d = pd.DataFrame({"S": h - j, "N": n, "HO": h}).dropna(subset=["S"])
        d = d[d.index >= pd.Timestamp(Y - 1, 8, 1)]
        nov1 = pd.Timestamp(Y - 1, 11, 1)
        fri = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
        ent = d.index[d.index.searchsorted(fri)]
        end = d.index.max()
        if Y <= 2021:
            assert end.month == 2 and end.year == Y, (Y, end)
        i0 = d.index.get_loc(ent)
        d = d.iloc[max(0, i0 - PRE_DAYS):].copy()
        d["live"] = d.index >= ent
        d.attrs.update(entry=ent, end=end)
        out[Y] = d
    pd.to_pickle(out, f)
    return out


def phys_m2_alarm():
    """Weekly PHYS-M2 ES (h=1, $ per spread) and alarm = ES / median of all earlier PHYS-M2 ES
    (>= 10 earlier forecasts, else NaN), indexed by forecast Friday."""
    from src.research.scenario_risk import models as SRM
    fc = SRM.run_all()
    p = fc[(fc.model == "PHYS-M2") & (fc.h == 1)].sort_values("date").copy()
    assert p.date.max() <= CUTOFF
    p["ES"] = p.q.apply(lambda q: float(np.mean(q[975:]))) * MULT
    med = [np.median(p.ES.values[:i]) if i >= 10 else np.nan for i in range(len(p))]
    p["alarm"] = p.ES.values / np.array(med)
    return p.set_index("date")[["ES", "alarm", "S", "winter"]]
