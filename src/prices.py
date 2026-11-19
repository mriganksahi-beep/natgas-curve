"""Henry Hub futures settlement prices from Databento (CME Globex, GLBX.MDP3).

Model 3 needs the official daily settlement of each named contract (e.g. March 2025 =
NGH5, April 2025 = NGJ5). Daily OHLCV bars are cut at UTC midnight and use last trades,
so settlements come from the `statistics` schema instead.

Use (Colab):
    pip install databento
    export DATABENTO_API_KEY=...
    python -m src.prices cost        # prints the price of the pull; downloads nothing
    python -m src.prices pull        # downloads, keeps outright contracts, saves CSV

NOT YET RUN against Databento (the build machine cannot reach it). Check the first pull.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec
OUTRIGHT = re.compile(r"^NG([FGHJKMNQUVXZ])(\d{1,2})$")
QUERY = dict(dataset="GLBX.MDP3", symbols=["NG.FUT"], stype_in="parent")


def cost(start="2010-06-06", end=None):
    import databento as db
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    for schema in ("statistics", "ohlcv-1d", "definition"):
        kw = dict(QUERY, schema=schema, start=start, end=end)
        c = client.metadata.get_cost(**kw)
        size = client.metadata.get_billable_size(**kw)
        print(f"{schema:12s} cost ${c:,.2f}   size {size / 1e6:,.1f} MB")


def contract_from_symbol(sym: str, trade_date: pd.Timestamp):
    """'NGH5' traded in 2024 -> (2025, 3). The year digit is resolved to the first year
    on or after the trade year with that last digit (contracts list ~12 years ahead at most)."""
    m = OUTRIGHT.match(sym)
    if not m:
        return None
    month = MONTH_CODES.index(m.group(1)) + 1
    digits = m.group(2)
    if len(digits) == 2:
        year = 2000 + int(digits)
    else:
        year = trade_date.year - trade_date.year % 10 + int(digits)
        if year < trade_date.year:
            year += 10
    return year, month


def pull(start="2010-06-06", end=None):
    import databento as db
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    data = client.timeseries.get_range(**QUERY, schema="statistics", start=start, end=end)
    df = data.to_df(map_symbols=True)
    settle = df[df["stat_type"] == db.StatType.SETTLEMENT_PRICE].copy()
    # ts_ref is the trading session the settlement belongs to
    settle["date"] = pd.to_datetime(settle["ts_ref"]).dt.tz_convert("America/Chicago").dt.normalize().dt.tz_localize(None)
    settle = settle[settle["symbol"].str.match(OUTRIGHT)]  # drop exchange-listed spreads
    parsed = [contract_from_symbol(s, d) for s, d in zip(settle["symbol"], settle["date"])]
    settle["contract_year"] = [p[0] for p in parsed]
    settle["contract_month"] = [p[1] for p in parsed]
    out = (settle.sort_index()
           .groupby(["date", "symbol"], as_index=False).last()  # final settlement per session
           [["date", "symbol", "contract_year", "contract_month", "price"]]
           .rename(columns={"price": "settle"}))
    out.to_csv(RAW / "ng_settlements.csv", index=False)
    print(out.groupby(out.date.dt.year).size().rename("rows per year").to_string())
    return out


def spread_panel(settles: pd.DataFrame, near_month=3, far_month=4) -> pd.DataFrame:
    """Daily near-minus-far spread for each year, e.g. March minus April (H-J)."""
    s = settles.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle")
    rows = []
    for (y, m) in s.columns:
        if m != near_month or (y, far_month) not in s.columns:
            continue
        sp = (s[(y, near_month)] - s[(y, far_month)]).dropna()
        rows.append(pd.DataFrame({"date": sp.index, "year": y, "spread": sp.values,
                                  "near": s.loc[sp.index, (y, near_month)].values,
                                  "far": s.loc[sp.index, (y, far_month)].values}))
    return pd.concat(rows, ignore_index=True)


def cross_check_with_eia(settles: pd.DataFrame) -> pd.DataFrame:
    """Compare against EIA's contract 1-4 series (to April 2024): front-month settles should match."""
    eia = pd.read_csv(RAW / "eia_futures_c1_c4.csv", parse_dates=["date"]).set_index("date")
    s = settles.copy()
    s["ym"] = s.contract_year * 12 + s.contract_month
    first = s.sort_values("ym").groupby("date").first()["settle"]
    j = eia[["c1"]].join(first.rename("databento_front"), how="inner")
    j["diff"] = j.databento_front - j.c1
    print(f"{len(j)} common days; median |diff| {j['diff'].abs().median():.4f}; "
          f"days with |diff| > 0.01: {(j['diff'].abs() > 0.01).sum()}")
    return j


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    if cmd == "cost":
        cost()
    elif cmd == "pull":
        out = pull()
        cross_check_with_eia(out)
