"""Study v4 shared building blocks: gas-weighted HDD forecasts, anomalies and news.

Inputs (free, committed by .github/workflows/collect_history.yml):
  data/forecasts/gefs_reforecast_ens5.csv.gz    2000-2019, 00Z runs
  data/forecasts/gefs_operational_ens5.csv.gz   2020-  (v11 to 2020-09-22, v12 after)
  data/forecasts/cpc_outlooks.csv.gz            2012-09-
  data/weather/daily_temps_2000.csv             realised, 2000-

Conventions
  * HDD is gas-weighted over the 17 Model-1 cities (heating weights), base 65F, per ET day.
  * A run with init date d (00Z) is public by ~06:00 ET on d, i.e. known at the d settle.
  * Forecast anomaly = forecast HDD - model climatology(lead, day of year), the climatology
    fitted on DEVELOPMENT years only (valid dates 2010-10-01..2021-12-31) with 3 annual
    harmonics per lead. Realised anomaly uses the same harmonic form fitted to realised HDD
    over the same years, so forecast and realised anomalies share a baseline up to model bias.
  * News between two runs (old, new) over a set of valid dates = sum over those dates of
    (new anomaly - old anomaly), where the old anomaly is 0 (climatology) for dates beyond
    the old run's last lead.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch import BASE_F, CITIES

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "data/forecasts"
DEV = (pd.Timestamp("2010-10-01"), pd.Timestamp("2021-12-31"))
V12_OPERATIONAL = pd.Timestamp("2020-09-23")
W = pd.Series({c: v[2] for c, v in CITIES.items()})


def _gw_hdd(df: pd.DataFrame, tcol: str) -> pd.Series:
    h = (BASE_F - df[tcol]).clip(lower=0) * df["city"].map(W)
    return h.groupby([df[k] for k in df.columns if k in ("init_date", "valid_et_date", "lead_days", "date")]).sum() / W.sum()


@lru_cache(maxsize=1)
def gefs_hdd() -> pd.DataFrame:
    """Long table: init_date, valid_date, lead, hdd_c00, hdd_ens, system."""
    parts = []
    for f, system in (("gefs_reforecast_ens5.csv.gz", "reforecast_v12"),
                      ("gefs_operational_ens5.csv.gz", "operational")):
        p = FC / f
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["init_date", "valid_et_date"])
        n = d.groupby(["init_date", "valid_et_date"]).city.transform("size")
        d = d[n == len(CITIES)]
        out = pd.DataFrame({"hdd_c00": _gw_hdd(d, "temp_c00"), "hdd_ens": _gw_hdd(d, "temp_ens")}).reset_index()
        out["system"] = system
        parts.append(out)
    g = pd.concat(parts).rename(columns={"valid_et_date": "valid_date", "lead_days": "lead"})
    g.loc[(g.system == "operational") & (g.init_date < V12_OPERATIONAL), "system"] = "operational_v11"
    g.loc[g.system == "operational", "system"] = "operational_v12"
    # where both reforecast and operational exist for a date, keep the reforecast (consistent model)
    pri = g.system.map({"reforecast_v12": 0, "operational_v12": 1, "operational_v11": 2})
    g = g.assign(_p=pri).sort_values("_p").drop_duplicates(["init_date", "valid_date", "lead"]).drop(columns="_p")
    return g.sort_values(["init_date", "lead"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def realised_hdd() -> pd.Series:
    d = pd.read_csv(ROOT / "data/weather/daily_temps_2000.csv", parse_dates=["date"])
    n = d.groupby("date").city.transform("size")
    d = d[(n == len(CITIES)) & d.temp_f.notna()]
    return _gw_hdd(d, "temp_f").rename("hdd")


def _harmonics(dates: pd.DatetimeIndex, k=3) -> np.ndarray:
    t = 2 * np.pi * dates.dayofyear.values / 365.25
    cols = [np.ones(len(t))]
    for j in range(1, k + 1):
        cols += [np.cos(j * t), np.sin(j * t)]
    return np.column_stack(cols)


@lru_cache(maxsize=1)
def climatology_coefs():
    """Per-lead harmonic climatology of forecast HDD (by system family) and of realised HDD,
    fitted on development valid dates only."""
    g = gefs_hdd()
    dev = g[(g.valid_date >= DEV[0]) & (g.valid_date <= DEV[1])]
    coefs = {}
    for col in ("hdd_c00", "hdd_ens"):
        for lead, grp in dev.groupby("lead"):
            X = _harmonics(pd.DatetimeIndex(grp.valid_date))
            coefs[(col, lead)] = np.linalg.lstsq(X, grp[col].values, rcond=None)[0]
    r = realised_hdd()
    r = r[(r.index >= DEV[0]) & (r.index <= DEV[1])]
    coefs[("realised", None)] = np.linalg.lstsq(_harmonics(pd.DatetimeIndex(r.index)), r.values, rcond=None)[0]
    return coefs


@lru_cache(maxsize=1)
def gefs_anomalies() -> pd.DataFrame:
    g = gefs_hdd().copy()
    c = climatology_coefs()
    for col in ("hdd_c00", "hdd_ens"):
        clim = np.full(len(g), np.nan)
        for lead, idx in g.groupby("lead").groups.items():
            if (col, lead) in c:
                clim[g.index.get_indexer(idx)] = _harmonics(pd.DatetimeIndex(g.loc[idx, "valid_date"])) @ c[(col, lead)]
        g[col.replace("hdd", "anom")] = g[col] - clim
    return g


@lru_cache(maxsize=1)
def realised_anomaly() -> pd.Series:
    r = realised_hdd()
    return (r - _harmonics(pd.DatetimeIndex(r.index)) @ climatology_coefs()[("realised", None)]).rename("anom")


def run_table(col="anom_ens") -> pd.DataFrame:
    """Wide: index init_date, columns valid_date -> anomaly (for fast news computations)."""
    a = gefs_anomalies()
    return a.pivot_table(index="init_date", columns="valid_date", values=col)


def news(table: pd.DataFrame, old_init, new_init, valid_dates) -> float:
    """Sum over valid_dates of (new - old) anomaly; old beyond its horizon = 0 (climatology)."""
    vd = pd.DatetimeIndex(valid_dates)
    new = table.loc[new_init].reindex(vd)
    if new.isna().any():
        return np.nan
    old = table.loc[old_init].reindex(vd).fillna(0.0) if old_init in table.index else pd.Series(0.0, index=vd)
    return float((new - old).sum())


def window_anomaly(table: pd.DataFrame, init, a: int, b: int) -> float:
    """Sum of the run's anomalies over valid dates init+a .. init+b (calendar days)."""
    vd = pd.date_range(init + pd.Timedelta(days=a), init + pd.Timedelta(days=b))
    v = table.loc[init].reindex(vd)
    return float(v.sum()) if v.notna().all() else np.nan


# ---------------------------------------------------------------- CPC (issued after the settle)
@lru_cache(maxsize=1)
def cpc_scores() -> pd.DataFrame:
    """Gas-weighted signed score (p_above - p_below, in points) per issue date and product.
    Positive = warm = FEWER HDD. Known from the settle of the NEXT trading day."""
    c = pd.read_csv(FC / "cpc_outlooks.csv.gz", parse_dates=["issue_date"])
    c["w"] = c.city.map(W)
    s = (c.signed * c.w).groupby([c.issue_date, c["product"].astype(str)]).sum() / W.sum()
    return s.unstack("product").rename(columns={"610": "cpc610", "814": "cpc814"})
