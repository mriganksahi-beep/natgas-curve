"""Download raw data from the original free sources.

Run this on your own machine or Colab (the sources are public; EIA needs a free API key
from https://www.eia.gov/opendata/register.php, set as the EIA_API_KEY env variable).

    python -m src.fetch storage
    python -m src.fetch weather
    python -m src.fetch futures

NOTE: written against the documented endpoints but not yet run end-to-end
(the build machine could not reach eia.gov / open-meteo.com). Check the first pull by eye.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# --- 1. EIA weekly working gas in storage, Lower 48 (Bcf) -----------------------------
EIA_STORAGE_SERIES = "NW2_EPG0_SWO_R48_BCF"


def fetch_storage(start="2010-01-01") -> pd.DataFrame:
    key = os.environ["EIA_API_KEY"]
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": EIA_STORAGE_SERIES,
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])
    df = df.rename(columns={"period": "week_ending", "value": "storage_bcf"})
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df["storage_bcf"] = pd.to_numeric(df["storage_bcf"])
    df = df[["week_ending", "storage_bcf"]].sort_values("week_ending")
    df.to_csv(RAW / "storage.csv", index=False)
    return df


# --- 2. Daily temperatures -> gas-weighted degree days ----------------------------------
# Cities stand in for regional gas demand. Weights are rough shares of residential +
# commercial gas use (heating) and of gas-fired power burn (cooling). They are a modelling
# choice and are tested as such (see reports/protocol.md).
CITIES = {
    #             lat     lon     heat_w  cool_w
    "New York":  (40.71, -74.01, 0.14, 0.06),
    "Chicago":   (41.88, -87.63, 0.14, 0.05),
    "Detroit":   (42.33, -83.05, 0.08, 0.03),
    "Boston":    (42.36, -71.06, 0.06, 0.02),
    "Philadelphia": (39.95, -75.17, 0.07, 0.04),
    "Pittsburgh": (40.44, -79.99, 0.05, 0.02),
    "Columbus":  (39.96, -83.00, 0.05, 0.03),
    "Minneapolis": (44.98, -93.27, 0.05, 0.02),
    "Atlanta":   (33.75, -84.39, 0.04, 0.08),
    "Houston":   (29.76, -95.37, 0.04, 0.18),
    "Dallas":    (32.78, -96.80, 0.04, 0.14),
    "Los Angeles": (34.05, -118.24, 0.07, 0.10),
    "Denver":    (39.74, -104.99, 0.04, 0.02),
    "Kansas City": (39.10, -94.58, 0.04, 0.04),
    "Miami":     (25.76, -80.19, 0.00, 0.14),
    "Seattle":   (47.61, -122.33, 0.05, 0.01),
    "Washington": (38.91, -77.04, 0.04, 0.02),
}
BASE_F = 65.0


WEATHER_CACHE = RAW / "weather_cache"


class RateLimited(RuntimeError):
    pass


def fetch_weather(start="2004-12-01", end=None) -> pd.DataFrame:
    """Daily mean temperature per city from the Open-Meteo archive.

    Open-Meteo's free tier counts a multi-year request as many calls, so a full pull
    can hit the hourly (and daily) limit part-way through. Each city is therefore saved
    as soon as it arrives, and a rerun skips cities already saved. If the limit is hit,
    wait about an hour and rerun; it picks up where it stopped.

    start: 2004-12-01 is enough. Storage starts in 2010 and the 5-year weather norms
    need the five years before that.
    """
    end = end or (pd.Timestamp.today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    for city, (lat, lon, _, _) in CITIES.items():
        path = WEATHER_CACHE / f"{city.replace(' ', '_')}.csv"
        if path.exists():
            continue
        for attempt in range(3):
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                        "daily": "temperature_2m_mean", "temperature_unit": "fahrenheit",
                        "timezone": "America/New_York"},
                timeout=120,
            )
            if r.status_code != 429:
                break
            if attempt < 2:
                print(f"{city}: rate limited, waiting 70s (attempt {attempt + 1} of 3)")
                time.sleep(70)
        if r.status_code == 429:
            done = len(list(WEATHER_CACHE.glob("*.csv")))
            raise RateLimited(
                f"Open-Meteo limit reached with {done} of {len(CITIES)} cities saved. "
                "Wait about an hour and rerun the same cell; saved cities are skipped. "
                "If it still fails, the daily limit is used up: rerun tomorrow.")
        r.raise_for_status()
        d = r.json()["daily"]
        pd.DataFrame({"date": d["time"], "city": city, "temp_f": d["temperature_2m_mean"]}
                     ).to_csv(path, index=False)
        print(f"saved {city}")
        time.sleep(2)
    daily = pd.concat(pd.read_csv(p, parse_dates=["date"]) for p in sorted(WEATHER_CACHE.glob("*.csv")))
    daily.to_csv(RAW / "daily_temps.csv", index=False)
    return daily


def weekly_degree_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Gas-weighted HDD/CDD summed over EIA storage weeks (Saturday to Friday)."""
    w = pd.DataFrame(CITIES, index=["lat", "lon", "hw", "cw"]).T
    d = daily.join(w, on="city")
    d["hdd"] = (BASE_F - d["temp_f"]).clip(lower=0) * d["hw"]
    d["cdd"] = (d["temp_f"] - BASE_F).clip(lower=0) * d["cw"]
    per_day = d.groupby("date")[["hdd", "cdd"]].sum() / pd.Series({"hdd": w.hw.sum(), "cdd": w.cw.sum()})
    wk = per_day.resample("W-FRI").agg(["sum", "count"])
    out = pd.DataFrame({"hdd": wk[("hdd", "sum")], "cdd": wk[("cdd", "sum")],
                        "days": wk[("hdd", "count")]})
    out = out[out["days"] == 7].drop(columns="days").round(1)
    out.index.name = "week_ending"
    out.reset_index().to_csv(RAW / "weekly_degree_days_own.csv", index=False)
    return out.reset_index()


# --- 3. NYMEX Henry Hub futures, contracts 1-4 (EIA; stops 2024-04-05) ------------------
def fetch_eia_futures() -> pd.DataFrame:
    frames = []
    for c in range(1, 5):
        url = f"https://www.eia.gov/dnav/ng/hist_xls/RNGC{c}d.xls"
        x = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
        x.columns = ["date", f"c{c}"]
        frames.append(x.set_index("date"))
    fut = pd.concat(frames, axis=1).dropna(how="all")
    fut.index = pd.to_datetime(fut.index)
    fut.to_csv(RAW / "eia_futures_c1_c4.csv")
    return fut


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("storage", "all"):
        print(fetch_storage().tail())
    if what in ("weather", "all"):
        print(weekly_degree_days(fetch_weather()).tail())
    if what in ("futures", "all"):
        print(fetch_eia_futures().tail())




# --- 5. Alternative to 1 and 3: EIA bulk file (no API key) ------------------------------
# Download NG.zip from https://www.eia.gov/opendata/bulkfiles.php, unzip to NG.txt.
BULK_SERIES = {
    "NG.NW2_EPG0_SWO_R48_BCF.W": "storage_bcf",
    "NG.RNGC1.D": "c1", "NG.RNGC2.D": "c2", "NG.RNGC3.D": "c3", "NG.RNGC4.D": "c4",
    "NG.RNGWHHD.D": "spot",
}


def from_bulk(path):
    import json
    found = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("series_id") in BULK_SERIES:
                s = pd.DataFrame(d["data"], columns=["date", "v"])
                s["date"] = pd.to_datetime(s["date"])
                found[BULK_SERIES[d["series_id"]]] = s.set_index("date")["v"].sort_index()
    st = found["storage_bcf"].rename("storage_bcf").rename_axis("week_ending").reset_index()
    st.to_csv(RAW / "storage.csv", index=False)
    fut = pd.concat([found[c].rename(c) for c in ("c1", "c2", "c3", "c4")], axis=1)
    fut.rename_axis("date").to_csv(RAW / "eia_futures_c1_c4.csv")
    found["spot"].rename("spot").rename_axis("date").to_csv(RAW / "henry_hub_spot.csv")
    return st, fut
