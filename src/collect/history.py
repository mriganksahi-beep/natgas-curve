"""Free history extensions (run on a GitHub runner; the dev workspace cannot reach these hosts).

  weather  : Open-Meteo archive daily mean temperature, 17 cities, 2000-01-01 to now
             -> data/weather/daily_temps_2000.csv   (Open-Meteo, CC BY 4.0)
  eia      : EIA weekly storage history workbook (pre-2010 estimates included) and the EIA bulk
             natural gas file -> data/raw/eia/ngshistory.xls, storage_history_*.csv
             (EIA data are public domain)

Run:  python -m src.collect.history weather | eia
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.fetch import CITIES

ROOT = Path(__file__).resolve().parents[2]


def weather(start="2000-01-01"):
    out = ROOT / "data/weather"
    out.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    frames = []
    for city, (lat, lon, _, _) in CITIES.items():
        for attempt in range(8):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params={"latitude": lat, "longitude": lon, "start_date": start,
                                     "end_date": end, "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
                                     "temperature_unit": "fahrenheit", "timezone": "America/New_York"},
                             timeout=180)
            if r.status_code == 429:
                print(f"{city}: rate limited, sleeping 75s", flush=True)
                time.sleep(75)
                continue
            r.raise_for_status()
            break
        d = r.json()["daily"]
        frames.append(pd.DataFrame({"date": d["time"], "city": city, "temp_f": d["temperature_2m_mean"],
                                    "tmax_f": d["temperature_2m_max"], "tmin_f": d["temperature_2m_min"]}))
        print(f"{city}: {len(d['time'])} days", flush=True)
        time.sleep(20)
    pd.concat(frames).to_csv(out / "daily_temps_2000.csv", index=False)


def eia():
    out = ROOT / "data/raw/eia"
    out.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://ir.eia.gov/ngs/ngshistory.xls", timeout=120)
    r.raise_for_status()
    (out / "ngshistory.xls").write_bytes(r.content)
    print("ngshistory.xls", len(r.content))
    # All weekly storage series in the bulk file (regions, totals), any start date.
    z = requests.get("https://www.eia.gov/opendata/bulk/NG.zip", timeout=300)
    z.raise_for_status()
    rows = []
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for line in io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8"):
            if '"f":"W"' not in line or "SWO" not in line:
                continue
            j = json.loads(line)
            for d, v in j.get("data", []):
                rows.append((j["series_id"], j.get("name"), d, v))
    df = pd.DataFrame(rows, columns=["series_id", "name", "date", "value"])
    df.to_csv(out / "storage_weekly_series_bulk.csv", index=False)
    print(df.groupby("series_id").date.agg(["min", "max", "count"]))


def eia_monthly():
    """All monthly natural-gas STORAGE series from the EIA bulk file (e.g. NG.N5020US2.M total working gas)."""
    out = ROOT / "data/raw/eia"
    out.mkdir(parents=True, exist_ok=True)
    z = requests.get("https://www.eia.gov/opendata/bulk/NG.zip", timeout=300)
    z.raise_for_status()
    rows = []
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for line in io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8"):
            if '"f":"M"' not in line:
                continue
            j = json.loads(line)
            name = (j.get("name") or "").lower()
            if "storage" not in name and "working gas" not in name:
                continue
            for d, v in j.get("data", []):
                rows.append((j["series_id"], j.get("name"), j.get("units"), d, v))
    df = pd.DataFrame(rows, columns=["series_id", "name", "units", "month", "value"])
    df.to_csv(out / "storage_monthly_series_bulk.csv", index=False)
    print(df.groupby(["series_id", "name"]).month.agg(["min", "max", "count"]).to_string()[:6000])


if __name__ == "__main__":
    {"weather": weather, "eia": eia, "eia_monthly": eia_monthly}[sys.argv[1]]()
