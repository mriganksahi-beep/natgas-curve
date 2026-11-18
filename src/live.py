"""Weekly live forecast of the EIA storage change, published before the release.

EIA publishes the storage change for the week ending last Friday on Thursday at 10:30
New York time (21:00 IST in winter, 20:00 IST in summer). Run this before then:

    python -m src.live forecast            # forecast the next unreported week
    python -m src.live score               # after release: fill in actuals

Weather for the recent days comes from Open-Meteo's forecast API (`past_days`), because
its archive lags by about five days. That source is model analysis rather than the
reanalysis used in training; the difference is logged per forecast.
Storage: refresh data/raw/storage.csv each week (EIA bulk file via src.fetch.from_bulk),
or add the latest level by hand with `python -m src.live add-storage 2026-11-13 3900`.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.fetch import BASE_F, CITIES
from src.model1 import FEATURES_V2, build_features, load_raw, ridge

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports/live_forecasts.csv"
ALPHA = 0.1  # chosen on 2018-2021 (protocol.md); frozen for live use
COLS = ["forecast_made_utc", "week_ending", "forecast_bcf", "lo80_bcf", "hi80_bcf",
        "five_year_avg_bcf", "hdd", "cdd", "weather_source", "actual_bcf", "miss_bcf"]


def recent_weather(days=16) -> pd.DataFrame:
    """Daily mean temperature for the last `days` days for each city (forecast API)."""
    frames = []
    for city, (lat, lon, _, _) in CITIES.items():
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon,
                                 "daily": "temperature_2m_mean", "temperature_unit": "fahrenheit",
                                 "timezone": "America/New_York", "past_days": days, "forecast_days": 1},
                         timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]
        frames.append(pd.DataFrame({"date": pd.to_datetime(d["time"]), "city": city,
                                    "temp_f": d["temperature_2m_mean"]}))
    return pd.concat(frames)


def week_degree_days(daily: pd.DataFrame, week_end: pd.Timestamp) -> dict:
    w = pd.DataFrame(CITIES, index=["lat", "lon", "hw", "cw"]).T
    d = daily[(daily.date > week_end - pd.Timedelta(days=7)) & (daily.date <= week_end)].join(w, on="city")
    if d.groupby("city").size().min() < 7 or d["temp_f"].isna().any():
        raise ValueError(f"Incomplete weather for week ending {week_end.date()}")
    hdd = ((BASE_F - d.temp_f).clip(lower=0) * d.hw).sum() / w.hw.sum()
    cdd = ((d.temp_f - BASE_F).clip(lower=0) * d.cw).sum() / w.cw.sum()
    return {"hdd": round(hdd, 1), "cdd": round(cdd, 1)}


def forecast(daily=None, now=None):
    s, w = load_raw()
    target = s.index.max() + pd.Timedelta(days=7)  # next week EIA has not reported
    now = now or datetime.now(timezone.utc)
    if pd.Timestamp(now).tz_convert(None) < target + pd.Timedelta(hours=24):
        raise ValueError(f"Week ending {target.date()} is not over yet")

    if target in w.index:
        dd, source = w.loc[target, ["hdd", "cdd"]].to_dict(), "archive"
    else:
        dd, source = week_degree_days(daily if daily is not None else recent_weather(), target), "forecast-api past_days"
    w2 = pd.concat([w, pd.DataFrame([{**dd, "forecast_days": 0}], index=[target])])
    w2.index.name = "week_ending"
    s2 = pd.concat([s, pd.DataFrame({"storage_bcf": [np.nan]}, index=[target])])
    s2.index.name = "week_ending"

    df = build_features(s2, w2, keep_last_unknown=True)
    assert df.index.max() == target and np.isnan(df.loc[target, "chg"])
    train = df[df.index < target]
    model = ridge(ALPHA)().fit(train[FEATURES_V2], train["chg"])
    point = float(model.predict(df.loc[[target], FEATURES_V2])[0])

    # 80% range from the walk-forward errors of the same model (2018 onward)
    wf = pd.read_csv(ROOT / "data/processed/model1_predictions_v2_own.csv", index_col=0, parse_dates=True)
    col = [c for c in wf if c.startswith("Ridge")][0]
    err = (wf["actual"] - wf[col]).dropna()
    lo, hi = point + err.quantile(0.1), point + err.quantile(0.9)

    row = {"forecast_made_utc": pd.Timestamp(now).strftime("%Y-%m-%d %H:%M"),
           "week_ending": target.date().isoformat(), "forecast_bcf": round(point),
           "lo80_bcf": round(lo), "hi80_bcf": round(hi),
           "five_year_avg_bcf": round(float(df.loc[target, "chg_5y"])),
           "hdd": dd["hdd"], "cdd": dd["cdd"], "weather_source": source,
           "actual_bcf": np.nan, "miss_bcf": np.nan}
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    if (log["week_ending"] == row["week_ending"]).any():
        raise ValueError(f"A forecast for {row['week_ending']} is already logged; not overwriting")
    pd.concat([log, pd.DataFrame([row])], ignore_index=True)[COLS].to_csv(LOG, index=False)
    print(f"Week ending {row['week_ending']}: forecast {row['forecast_bcf']:+d} Bcf "
          f"(80% range {row['lo80_bcf']:+d} to {row['hi80_bcf']:+d}); "
          f"5-year average {row['five_year_avg_bcf']:+d}; HDD {dd['hdd']}, CDD {dd['cdd']} [{source}]")
    return row


def add_storage(week_ending, level):
    p = ROOT / "data/raw/storage.csv"
    s = pd.read_csv(p, parse_dates=["week_ending"])
    t = pd.Timestamp(week_ending)
    if t in set(s.week_ending):
        raise ValueError("Week already present")
    if t - s.week_ending.max() != pd.Timedelta(days=7):
        raise ValueError("Must be the week right after the latest one")
    pd.concat([s, pd.DataFrame({"week_ending": [t], "storage_bcf": [int(level)]})]).to_csv(p, index=False)


def score():
    s, _ = load_raw()
    chg = s["storage_bcf"].diff()
    log = pd.read_csv(LOG)
    for i, r in log.iterrows():
        t = pd.Timestamp(r.week_ending)
        if pd.isna(r.actual_bcf) and t in chg.index:
            log.loc[i, "actual_bcf"] = chg[t]
            log.loc[i, "miss_bcf"] = chg[t] - r.forecast_bcf
    log.to_csv(LOG, index=False)
    done = log.dropna(subset=["actual_bcf"])
    if len(done):
        print(done[["week_ending", "forecast_bcf", "lo80_bcf", "hi80_bcf", "actual_bcf", "miss_bcf"]].to_string(index=False))
        inside = ((done.actual_bcf >= done.lo80_bcf) & (done.actual_bcf <= done.hi80_bcf)).mean()
        print(f"Live record: {len(done)} weeks, MAE {done.miss_bcf.abs().mean():.1f} Bcf, "
              f"inside 80% range {100 * inside:.0f}%")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    if cmd == "forecast":
        forecast()
    elif cmd == "score":
        score()
    elif cmd == "add-storage":
        add_storage(sys.argv[2], sys.argv[3])
