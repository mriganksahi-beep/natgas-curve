"""Model 3, step 1: the weekly dataset.

One row per EIA storage report in winter (November to February):
  obs_date   the first trading day on or after the Friday after the storage week ends.
             EIA normally reports on Thursday (sometimes Wednesday in holiday weeks), so
             this settlement always comes after the report. That holds even in weeks
             where the report slips to Friday: it is out at 10:30 ET, before the 14:30 ET
             settlement.
  spread     March minus April settlement for the delivery year whose end-March storage
             Model 2 forecasts ($/MMBtu)
  inputs     Model 2's end-March view from storage through the reported week:
             expected surplus = median - 5-year average end-March level
             downside = 10th percentile - 5-year average end-March level
             plus the April price and weeks until the March contract stops trading
  targets    spread 1 and 4 weeks later (first trading day on or after obs_date + 7h days)

All inputs in a row were public before that row's settlement. The checks are in
tests/test_model3_data.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
MIN_DAYS_BEFORE_EXPIRY = 25  # protocol: last observation at least 25 days before March expires


def load_curve():
    c = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"])
    assert not c["date"].dt.dayofweek.ge(5).any(), "weekend dates: run src.prices.fix_first_pull()"
    return c.dropna(subset=["settle"])


def end_march_levels():
    """Storage level in the last EIA week of March (25-31 March), by year."""
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"]).set_index("week_ending")["storage_bcf"]
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    return m.groupby(m.index.year).last()


def build(m2_file="model2_backtest_v2b_from2016.csv"):
    curve = load_curve()
    px = curve.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle")
    days = px.index

    m2 = pd.read_csv(PROC / m2_file, parse_dates=["origin", "target"])
    m2 = m2[m2["season"] == "winter"].copy()
    em = end_march_levels()

    rows = []
    for _, r in m2.iterrows():
        Y = r.target.year
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        h, j = px[(Y, 3)].dropna(), px[(Y, 4)]
        h_last = h.index.max()  # last settlement of the March contract (its expiry)
        report_friday = r.origin + pd.Timedelta(days=7)
        pos = days.searchsorted(report_friday)
        if pos >= len(days):
            continue
        obs = days[pos]
        if obs > h_last - pd.Timedelta(days=MIN_DAYS_BEFORE_EXPIRY) or r.origin.month not in (11, 12, 1, 2):
            continue
        if pd.isna(h.get(obs)) or pd.isna(j.get(obs)):
            continue
        prior = [em.get(Y - k) for k in range(1, 6)]
        if any(v is None or np.isnan(v) for v in prior):
            continue
        avg5 = float(np.mean(prior))
        row = {"obs_date": obs, "origin_week": r.origin, "delivery_year": Y,
               "winter": f"{Y - 1}-{str(Y)[2:]}",
               "spread": h[obs] - j[obs], "march": h[obs], "april": j[obs],
               "weeks_to_expiry": (h_last - obs).days / 7,
               "m2_median": r["median"], "m2_p10": r["p10"], "m2_p90": r["p90"],
               "end_march_5y_avg": avg5,
               "exp_surplus": r["median"] - avg5, "downside": r["p10"] - avg5,
               "actual_end_march": r["actual"]}
        for hz in (1, 4):
            p2 = days.searchsorted(obs + pd.Timedelta(days=7 * hz))
            t2 = days[p2] if p2 < len(days) else None
            ok = t2 is not None and t2 <= h_last and not pd.isna(h.get(t2)) and not pd.isna(j.get(t2))
            row[f"date_{hz}w"] = t2 if ok else pd.NaT
            row[f"spread_{hz}w"] = h[t2] - j[t2] if ok else np.nan
            row[f"dS_{hz}w"] = row[f"spread_{hz}w"] - row["spread"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("obs_date").reset_index(drop=True)
    df["period"] = np.where(df.obs_date >= pd.Timestamp("2022-01-01"), "holdout", "validation")
    df.to_csv(PROC / "model3_weekly.csv", index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby(["period", "winter"]).agg(rows=("spread", "size"), first=("obs_date", "min"),
                                             last=("obs_date", "max")).to_string())
