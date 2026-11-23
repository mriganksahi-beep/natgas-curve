"""Live tests for winter 2026-27 (frozen in reports/live_2026_27/PROTOCOL.md before 1 Dec 2026).

L1  March-April 2027 premium path: short 1 H27-J27 spread from two frozen entry dates, marked
    to market daily to the March 2027 expiry.
L2  The curve as a 2-week weather forecaster: frozen coefficients from
    data/processed/research_curve_weather/frozen_holdout_coefs.json (copied into the protocol),
    scored weekly on Nov-Feb Fridays.

Prices: Yahoo Finance daily closes of individual NYMEX contracts (free; no licensed CME data
is stored in this repo). Weather: Open-Meteo, same 17 cities and gas weights as Model 1.

Run:  python -m src.live2627            (updates reports/live_2026_27/*)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.fetch import BASE_F, CITIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/live_2026_27"
RAW = ROOT / "data/raw"
MONTH_CODE = "FGHJKMNQUVXZ"

# ---- frozen parameters (do not edit after 2026-11-30; see PROTOCOL.md) -------------------
NEAR, FAR = (2027, 3), (2027, 4)            # March and April 2027
ENTRIES = {"may": pd.Timestamp("2026-05-22"),  # last trading day on/before Mar-27 expiry - 9 months
           "nov": pd.Timestamp("2026-11-06")}  # first trading day on/after first Friday on/after 1 Nov
COST = 160.0                                 # $ per spread round trip (4 ticks x 2 legs x 2 sides)
SIZE = 10_000                                # MMBtu per leg
COEF = {"front":      {"const": -5.308149484407565, "r_front": 3.108801158297475},
        "front+pers": {"const": -3.3781931726610157, "r_front": 2.6731921065618662,
                       "a0": 0.6852289128217662},
        "pers":       {"const": -6.615909939758121, "a0": 0.7513212685014502}}
SEASON = (pd.Timestamp("2026-11-06"), pd.Timestamp("2027-02-26"))
# ------------------------------------------------------------------------------------------


def ticker(y: int, m: int) -> str:
    return f"NG{MONTH_CODE[m - 1]}{str(y)[2:]}.NYM"


def expiry(y: int, m: int) -> pd.Timestamp:
    """NG last trading day: 3 business days before the first calendar day of delivery."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays("2025-01-01", "2028-12-31")
    return pd.Timestamp(y, m, 1) - pd.offsets.CustomBusinessDay(3, holidays=hol)


def yahoo_closes(contracts, start="2026-04-01") -> pd.DataFrame:
    import yfinance as yf
    out = {}
    for y, m in contracts:
        for attempt in range(4):
            try:
                d = yf.download(ticker(y, m), start=start, progress=False, auto_adjust=False,
                                timeout=30, threads=False)
                print(f"{ticker(y, m)}: {len(d)} rows", flush=True)
                break
            except Exception as e:  # network hiccup
                if attempt == 3:
                    raise
                print(f"{ticker(y, m)}: {e}; retry")
                time.sleep(20)
        if len(d):
            s = d["Close"].squeeze("columns") if isinstance(d["Close"], pd.DataFrame) else d["Close"]
            out[(y, m)] = s.rename(ticker(y, m))
    px = pd.DataFrame(out)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px.sort_index()


# ---------------------------------------------------------------------------------- L1
def widowmaker_path(px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = (px[NEAR] - px[FAR]).dropna().rename("spread")
    s = s[s.index <= expiry(*NEAR)]
    df = s.to_frame()
    summary = {}
    for name, d in ENTRIES.items():
        if d not in s.index:
            raise ValueError(f"entry {d.date()} missing from price data")
        pnl = SIZE * (s.loc[d] - s) - COST
        pnl[s.index < d] = np.nan
        dd = pnl - pnl.cummax()
        df[f"pnl_{name}"], df[f"drawdown_{name}"] = pnl.round(0), dd.round(0)
        live = pnl.dropna()
        summary[name] = {"entry": str(d.date()), "spread_at_entry": round(float(s.loc[d]), 3),
                         "last_date": str(live.index[-1].date()),
                         "spread_last": round(float(s.iloc[-1]), 3),
                         "pnl_last": round(float(live.iloc[-1])),
                         "worst_pnl": round(float(live.min())),
                         "worst_drawdown": round(float(dd.min())),
                         "expired": bool(s.index[-1] >= expiry(*NEAR))}
    return df, summary


# ---------------------------------------------------------------------------------- L2
def daily_temps(start: str) -> pd.DataFrame:
    """Realised daily mean temps (F): Open-Meteo archive, topped up with the forecast API's
    past days for the most recent dates the archive does not have yet."""
    end = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for city, (lat, lon, _, _) in CITIES.items():
        got = []
        for url, extra in (("https://archive-api.open-meteo.com/v1/archive",
                            {"start_date": start, "end_date": end}),
                           ("https://api.open-meteo.com/v1/forecast",
                            {"past_days": 20, "forecast_days": 1})):
            for attempt in range(4):
                try:
                    r = requests.get(url, params={"latitude": lat, "longitude": lon,
                                                  "daily": "temperature_2m_mean",
                                                  "temperature_unit": "fahrenheit",
                                                  "timezone": "America/New_York", **extra}, timeout=90)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 3:
                        raise
                    print(f"{city}: {e}; retry", flush=True)
                    time.sleep(30 * (attempt + 1))
            d = r.json()["daily"]
            got.append(pd.Series(d["temperature_2m_mean"], index=pd.to_datetime(d["time"]), dtype=float))
        s = got[0].combine_first(got[1])  # archive first
        s = s[s.index <= end]
        frames.append(pd.DataFrame({"date": s.index, "city": city, "temp_f": s.values}))
        print(f"weather {city}: {s.notna().sum()} days", flush=True)
        time.sleep(1)
    return pd.concat(frames)


def weekly_hdd(daily: pd.DataFrame) -> pd.Series:
    w = pd.DataFrame(CITIES, index=["lat", "lon", "hw", "cw"]).T
    d = daily.dropna().join(w, on="city")
    d["hdd"] = (BASE_F - d.temp_f).clip(lower=0) * d.hw
    per_day = d.groupby("date").agg(hdd=("hdd", "sum"), n=("city", "size"))
    per_day = per_day[per_day.n == len(CITIES)]["hdd"] / w.hw.sum()
    wk = per_day.resample("W-FRI").agg(["sum", "count"])
    return wk.loc[wk["count"] == 7, "sum"].round(1).rename("hdd")


def hdd_anomaly(recent: pd.Series) -> pd.Series:
    """Anomaly vs the mean of the same calendar week in the prior 10 years (>= 5), exactly as
    in the development panel, on our archived weekly series extended with recent weeks."""
    from src.research.curve_weather.data import same_week
    hist = pd.read_csv(RAW / "weekly_degree_days_own.csv", parse_dates=["week_ending"]
                       ).set_index("week_ending")["hdd"]
    s = hist.combine_first(recent).sort_index()  # archived values win where both exist
    return (s - same_week(s, 10, 5)).rename("anom")


def weather_skill(px: pd.DataFrame, anom: pd.Series) -> tuple[pd.DataFrame, dict]:
    days = px.dropna(how="all").index
    rows = []
    for F in pd.date_range(*SEASON, freq="W-FRI"):
        p0, pp = days.searchsorted(F), days.searchsorted(F - pd.Timedelta(days=7))
        if p0 >= len(days):
            break
        d0, dp = days[p0], days[pp]
        live = sorted(c for c in px.columns if (expiry(*c) - d0).days >= 0
                      and len(pd.bdate_range(d0, expiry(*c))) - 1 >= 2)
        front = live[0]
        a, b = px.loc[dp, front], px.loc[d0, front]
        r = 100 * (np.log(b) - np.log(a)) if np.isfinite(a) and np.isfinite(b) else np.nan
        a0 = anom.get(F, np.nan)
        wx2 = anom.get(F + pd.Timedelta(days=7), np.nan) + anom.get(F + pd.Timedelta(days=14), np.nan)
        x = {"r_front": r, "a0": a0}
        row = {"friday": F.date(), "settle_day": d0.date(), "front": ticker(*front),
               "r_front_pct": round(r, 3), "a0_hdd": a0, "wx2_realised_hdd": wx2}
        for k, c in COEF.items():
            row[f"fc_{k}"] = round(c["const"] + sum(c[f] * x[f] for f in c if f != "const"), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    done = df.dropna(subset=["wx2_realised_hdd", "fc_front", "fc_pers", "fc_front+pers"])
    summ = {"weeks_scored": int(len(done))}
    if len(done) >= 2:
        y = done.wx2_realised_hdd
        for k in COEF:
            summ[f"SS_{k}"] = round(float(1 - ((y - done[f"fc_{k}"]) ** 2).sum() / (y ** 2).sum()), 3)
            summ[f"corr_{k}"] = round(float(np.corrcoef(y, done[f"fc_{k}"])[0, 1]), 3)
        summ["dSS_front_vs_pers"] = round(summ["SS_front"] - summ["SS_pers"], 3)
        summ["dSS_frontpers_vs_pers"] = round(summ["SS_front+pers"] - summ["SS_pers"], 3)
    return df, summ


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contracts = sorted({NEAR, FAR, (2026, 12), (2027, 1), (2027, 2), (2027, 5)})
    px = yahoo_closes(contracts)
    print("Yahoo closes (for the settle check):")
    print(px.tail(15).round(3).to_string())
    path, s1 = widowmaker_path(px)
    path.to_csv(OUT / "L1_widowmaker_path.csv", index_label="date")
    daily = daily_temps("2026-09-01")
    anom = hdd_anomaly(weekly_hdd(daily))
    ws, s2 = weather_skill(px, anom)
    ws.to_csv(OUT / "L2_weather_skill.csv", index=False)
    stamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    (OUT / "status.json").write_text(json.dumps({"updated": stamp, "L1": s1, "L2": s2}, indent=1))
    print(json.dumps({"L1": s1, "L2": s2}, indent=1))


if __name__ == "__main__":
    main()
