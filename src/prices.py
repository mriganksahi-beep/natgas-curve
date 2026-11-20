"""Henry Hub futures settlement prices from Databento (CME Globex, GLBX.MDP3).

Model 3 needs the official daily settlement of each named contract (e.g. March 2025 =
NGH5, April 2025 = NGJ5). Daily OHLCV bars are cut at UTC midnight and use last trades,
so settlements come from the `statistics` schema instead.

Use (Colab):
    pip install databento
    export DATABENTO_API_KEY=...
    python -m src.prices cost        # price of the targeted pull; downloads nothing
    python -m src.prices pull        # downloads only the contracts Model 3 needs, saves CSV
    python -m src.prices cost-full-feed / pull-full
                                     # whole NG curve: settlement, volume, open interest for every
                                     # outright contract, via a Databento batch job (~9 GB raw, compact output)

Licence: CME/Databento data may not be redistributed. The downloaded files are git-ignored;
only derived results (model scores, charts) go in the public repo.

Why targeted: the whole NG parent feed (`NG.FUT`) in the statistics schema is ~9 GB
(thousands of exchange-listed spreads and intraday statistic updates). Model 3 needs only
four contracts per year, each over the months it is used, which is a few MB.

NOT YET RUN against Databento (the build machine cannot reach it). Check the first pull.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def historical_end():
    """Latest end date the free historical licence allows. Databento treats roughly the
    last day as live data, which needs a subscription, so stop two days back."""
    return (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec
OUTRIGHT = re.compile(r"^NG([FGHJKMNQUVXZ])(\d{1,2})$")
QUERY = dict(dataset="GLBX.MDP3", symbols=["NG.FUT"], stype_in="parent")


def cost(start="2010-06-06", end=None):
    import databento as db
    end = end or historical_end()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    for schema in ("statistics", "ohlcv-1d", "definition"):
        kw = dict(QUERY, schema=schema, start=start, end=end)
        c = client.metadata.get_cost(**kw)
        size = client.metadata.get_billable_size(**kw)
        print(f"{schema:12s} cost ${c:,.2f}   size {size / 1e6:,.1f} MB")


def contract_windows(first_year=2011, last_year=None):
    """(raw_symbol, contract_year, contract_month, start, end) for the contracts Model 3 uses.

    Primary:   March (H) and April (J) of year Y, from 1 Aug of Y-1 to 1 Mar of Y.
    Secondary: October (V) of Y and January (F) of Y+1, from 1 Mar to 1 Nov of Y.
    The single-digit CME year code is unambiguous inside each window.
    """
    last_year = last_year or pd.Timestamp.today().year + 1
    today = pd.Timestamp(historical_end())
    out = []
    for y in range(first_year, last_year + 1):
        d = y % 10
        for code, month, start, end in (("H", 3, f"{y - 1}-08-01", f"{y}-03-01"),
                                        ("J", 4, f"{y - 1}-08-01", f"{y}-03-01")):
            out.append((f"NG{code}{d}", y, month, start, end))
        out.append((f"NGV{d}", y, 10, f"{y}-03-01", f"{y}-11-01"))
        out.append((f"NGF{(y + 1) % 10}", y + 1, 1, f"{y}-03-01", f"{y}-11-01"))
    out = [(s, cy, cm, a, min(pd.Timestamp(b), today).strftime("%Y-%m-%d"))
           for s, cy, cm, a, b in out if pd.Timestamp(a) < today]
    return out


def cost_contracts(first_year=2011):
    import databento as db
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    total_cost = total_size = 0.0
    for sym, cy, cm, a, b in contract_windows(first_year):
        kw = dict(dataset="GLBX.MDP3", symbols=[sym], stype_in="raw_symbol",
                  schema="statistics", start=a, end=b)
        total_cost += client.metadata.get_cost(**kw)
        total_size += client.metadata.get_billable_size(**kw)
    print(f"Targeted pull: {len(contract_windows(first_year))} contract windows, "
          f"cost ${total_cost:,.2f}, size {total_size / 1e6:,.1f} MB")


def pull_contracts(first_year=2011):
    import databento as db
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    frames = []
    for sym, cy, cm, a, b in contract_windows(first_year):
        data = client.timeseries.get_range(dataset="GLBX.MDP3", symbols=[sym], stype_in="raw_symbol",
                                           schema="statistics", start=a, end=b)
        df = data.to_df()
        if df.empty:
            print(f"{sym} {a}..{b}: no data")
            continue
        st = df[df["stat_type"] == db.StatType.SETTLEMENT_PRICE].copy()
        st["date"] = (pd.to_datetime(st["ts_ref"]).dt.tz_convert("America/Chicago")
                      .dt.normalize().dt.tz_localize(None))
        st = st.sort_index().groupby("date", as_index=False).last()[["date", "price"]]
        st["symbol"], st["contract_year"], st["contract_month"] = sym, cy, cm
        frames.append(st.rename(columns={"price": "settle"}))
        print(f"{sym} ({cy}-{cm:02d}): {len(st)} settlements {st.date.min().date()}..{st.date.max().date()}")
    out = (pd.concat(frames).drop_duplicates(["date", "contract_year", "contract_month"], keep="last")
           .sort_values(["contract_year", "contract_month", "date"]))
    out[["date", "symbol", "contract_year", "contract_month", "settle"]].to_csv(RAW / "ng_settlements.csv", index=False)
    return out


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


BATCH_DIR = RAW / "ng_statistics_batch"


def _batch_files(client, start, end, out_dir: Path):
    """Get the full NG statistics feed as a Databento batch job (recommended over
    streaming above 5 GB). Resumable: the job id is saved, finished files are reused."""
    import time
    out_dir = Path(os.path.realpath(out_dir))  # follow a link to Google Drive if there is one
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(out_dir.rglob("*.dbn.zst"))
    if files:
        print(f"Using {len(files)} files already downloaded in {out_dir}")
        return files
    id_file = out_dir / "job_id.txt"
    if id_file.exists():
        job_id = id_file.read_text().strip()
        print("Resuming batch job", job_id)
    else:
        job = client.batch.submit_job(**QUERY, schema="statistics", start=start, end=end,
                                      encoding="dbn", compression="zstd", split_duration="month")
        job_id = job["id"]
        id_file.write_text(job_id)
        print("Submitted batch job", job_id, "- Databento prepares the files; this can take a while")
    while True:
        state = next((j["state"] for j in client.batch.list_jobs(since="2026-01-01") if j["id"] == job_id), "unknown")
        print(f"  job state: {state}")
        if state == "done":
            break
        if state in ("expired", "unknown"):
            raise RuntimeError(f"Batch job {job_id} is {state}; delete {id_file} to resubmit")
        time.sleep(60)
    client.batch.download(job_id=job_id, output_dir=out_dir)
    return sorted(out_dir.rglob("*.dbn.zst"))


def pull_full(start="2010-06-06", end=None, chunk=2_000_000):
    """Full NG curve: for every OUTRIGHT contract and trading day, the settlement price,
    cleared volume and open interest. Exchange-listed spreads and intraday updates are
    discarded. Files are read chunk by chunk, so memory use stays small.
    Output: data/raw/ng_curve_daily.csv (one row per contract per day).
    """
    import databento as db
    end = end or historical_end()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    files = _batch_files(client, start, end, BATCH_DIR)
    keep = {db.StatType.SETTLEMENT_PRICE: "settle", db.StatType.CLEARED_VOLUME: "volume",
            db.StatType.OPEN_INTEREST: "open_interest"}
    parts, n = [], 0
    for f in files:
        store = db.DBNStore.from_file(str(f))
        for df in store.to_df(count=chunk, map_symbols=True):
            n += len(df)
            df = df[df["stat_type"].isin(list(keep)) & df["symbol"].astype(str).str.match(OUTRIGHT)]
            if df.empty:
                continue
            df = df.assign(
                field=df["stat_type"].map(keep),
                value=df["price"].where(df["stat_type"] == db.StatType.SETTLEMENT_PRICE, df["quantity"]),
                date=pd.to_datetime(df["ts_ref"]).dt.tz_convert("America/Chicago").dt.normalize().dt.tz_localize(None),
            )
            parts.append(df.reset_index()[["ts_recv", "date", "instrument_id", "symbol", "field", "value"]])
        print(f"  {f.name}: processed {n:,} records so far")
    long = pd.concat(parts).sort_values("ts_recv")
    # last published value of each field per contract per session (settlements can be revised)
    last = long.groupby(["date", "instrument_id", "symbol", "field"], as_index=False)["value"].last()
    wide = last.pivot_table(index=["date", "instrument_id", "symbol"], columns="field",
                            values="value", aggfunc="last").reset_index()
    wide = assign_contract_years(wide)
    cols = ["date", "symbol", "contract_year", "contract_month", "settle", "volume", "open_interest"]
    wide = wide[[c for c in cols if c in wide.columns]].sort_values(["contract_year", "contract_month", "date"])
    wide.to_csv(RAW / "ng_curve_daily.csv", index=False)
    print(f"Saved {len(wide):,} contract-days, {wide.groupby(['contract_year', 'contract_month']).ngroups} "
          f"contracts, {wide.date.min().date()}..{wide.date.max().date()}")
    s = wide.dropna(subset=["settle"])  # the Model 3 subset, same format as pull_contracts
    s[["date", "symbol", "contract_year", "contract_month", "settle"]].to_csv(RAW / "ng_settlements.csv", index=False)
    return wide


def assign_contract_years(wide: pd.DataFrame) -> pd.DataFrame:
    """Resolve CME's one-digit year code per instrument.

    A contract for month m of year Y stops trading in month m-1 (December of Y-1 for
    January). For an instrument that expired inside the data, Y follows from its last
    trading day. For one still trading, use the first year on or after its last day whose
    final digit matches the code.
    """
    g = wide.groupby("instrument_id").agg(symbol=("symbol", "first"), last=("date", "max"))
    data_end = wide["date"].max()
    years = {}
    for iid, r in g.iterrows():
        m = OUTRIGHT.match(r.symbol)
        month = MONTH_CODES.index(m.group(1)) + 1
        digits = m.group(2)
        if len(digits) == 2:
            years[iid] = 2000 + int(digits)
            continue
        if r["last"] < data_end - pd.Timedelta(days=5):  # expired inside the data
            y = r["last"].year + (1 if month == 1 else 0)
        else:
            y = r["last"].year - r["last"].year % 10 + int(digits)
            if (y, month) < (r["last"].year, r["last"].month):
                y += 10
        if y % 10 != int(digits):
            raise ValueError(f"year code mismatch for {r.symbol} (instrument {iid})")
        years[iid] = y
    wide["contract_year"] = wide["instrument_id"].map(years)
    wide["contract_month"] = wide["symbol"].map(lambda s: MONTH_CODES.index(OUTRIGHT.match(s).group(1)) + 1)
    return wide


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
    """Each February (1st-20th) March is EIA's contract 1 and April its contract 2.
    Our Databento settlements should match those EIA series (available to April 2024)."""
    eia = pd.read_csv(RAW / "eia_futures_c1_c4.csv", parse_dates=["date"]).set_index("date")
    s = settles.copy()
    s["date"] = pd.to_datetime(s["date"])
    feb = s[(s.date.dt.month == 2) & (s.date.dt.day <= 20) & (s.contract_year == s.date.dt.year)]
    h = feb[feb.contract_month == 3].set_index("date")["settle"].rename("march")
    j = feb[feb.contract_month == 4].set_index("date")["settle"].rename("april")
    c = eia[["c1", "c2"]].join(h, how="inner").join(j, how="inner")
    c["diff_c1"], c["diff_c2"] = c.march - c.c1, c.april - c.c2
    print(f"Cross-check vs EIA (February days, {c.index.year.min()}-{c.index.year.max()}): {len(c)} days; "
          f"max |March - c1| {c.diff_c1.abs().max():.4f}, max |April - c2| {c.diff_c2.abs().max():.4f}; "
          f"days off by > $0.005: {((c.diff_c1.abs() > 0.005) | (c.diff_c2.abs() > 0.005)).sum()}")
    return c


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    if cmd == "cost":
        cost_contracts()
    elif cmd == "cost-full-feed":
        cost()
    elif cmd == "pull":
        out = pull_contracts()
        cross_check_with_eia(out)
    elif cmd == "pull-full":
        wide = pull_full()
        cross_check_with_eia(wide.dropna(subset=["settle"]))
