"""GEFS control-member 2 m temperature forecasts at our 17 gas cities (free NOAA data on AWS).

Sources (public, no key):
  reforecast  s3://noaa-gefs-retrospective/GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/
              Days:1-10 (0.25 deg, 3-hourly) and Days:10-16 (0.5 deg, 6-hourly); 2000-2019
  operational s3://noaa-gefs-pds/gefs.YYYYMMDD/00/...  (GEFSv12 from 2020-09-23; v11 before)

Only the TMP:2 m messages at 6-hourly steps are fetched, via HTTP range requests built from the
.idx files, so one run costs ~35 MB instead of ~1 GB.

Output, one row per (init date, valid ET date, city): daily mean temperature (F) from the four
6-hourly values falling on that US-Eastern day (valid 06, 12, 18 UTC that day and 00 UTC the
next; ET = UTC-5 in winter). Lead = valid date - init date in days. Only complete days are kept.

Run:  python -m src.collect.gefs reforecast 2005      (all winter days of calendar year 2005)
      python -m src.collect.gefs operational 2026-11-20 2026-11-22
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.fetch import CITIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/forecasts/gefs"
RF = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OP = "https://noaa-gefs-pds.s3.amazonaws.com"
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=64))
import threading
_DECODE_LOCK = threading.Lock()
WINTER_MONTHS = (10, 11, 12, 1, 2, 3, 4)
MAX_LEAD_H = 384


def get(url, rng=None, tries=5):
    for k in range(tries):
        try:
            r = SESSION.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else None, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def parse_idx(text: str):
    """-> list of (start, end_or_None, fcst_hour, description) for TMP:2 m messages."""
    lines = [l.split(":") for l in text.strip().splitlines()]
    out = []
    for i, p in enumerate(lines):
        if p[3] != "TMP" or p[4] != "2 m above ground":
            continue
        desc = p[5]
        if "anl" in desc:
            h = 0
        elif "hour fcst" in desc:
            h = int(desc.split()[0])
        else:
            continue
        start = int(p[1])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        out.append((start, end, h, desc))
    return out


def decode(blobs: list[bytes]):
    """Decode GRIB2 messages -> (list of value arrays, lats, lons) using pygrib."""
    import pygrib
    with tempfile.NamedTemporaryFile(suffix=".grb2") as f:
        f.write(b"".join(blobs))
        f.flush()
        g = pygrib.open(f.name)
        msgs = list(g)
        lats, lons = msgs[0].latlons()
        vals = [m.values for m in msgs]
        g.close()
    return vals, lats, lons


_IDX_CACHE: dict = {}


def city_index(lats, lons):
    key = (lats.shape, float(lats[0, 0]), float(lons[0, 0]))
    if key not in _IDX_CACHE:
        idx = []
        for _, (la, lo, _, _) in CITIES.items():
            lo360 = lo % 360
            d = (lats - la) ** 2 + ((lons % 360) - lo360) ** 2
            idx.append(np.unravel_index(np.argmin(d), d.shape))
        _IDX_CACHE[key] = idx
    return _IDX_CACHE[key]


def fetch_messages(file_url: str, want_hours: set[int]):
    """Range-fetch TMP:2 m messages for wanted forecast hours from one GRIB2 file."""
    idx = get(file_url + ".idx")
    if idx is None:
        return {}
    got = {}
    for start, end, h, _ in parse_idx(idx.decode()):
        if h in want_hours and h not in got:
            if end is None:
                end = start + 5_000_000
            got[h] = get(file_url, (start, end))
    return got


MEMBERS = ("c00", "p01", "p02", "p03", "p04")  # the reforecast has 5 daily members
POOL = cf.ThreadPoolExecutor(24)  # shared pool for HTTP requests


def plan(init: pd.Timestamp, source: str, mem: str):
    """-> list of (grib_url, wanted forecast hours) for one member's 00Z run."""
    hours = set(range(6, MAX_LEAD_H + 1, 6))
    ymdh = init.strftime("%Y%m%d") + "00"
    if source == "reforecast":
        base = f"{RF}/{init.year}/{ymdh}/{mem}"
        return [(f"{base}/Days%3A1-10/tmp_2m_{ymdh}_{mem}.grib2", {h for h in hours if h <= 240}),
                (f"{base}/Days%3A10-16/tmp_2m_{ymdh}_{mem}.grib2", {h for h in hours if h > 240})]
    day = init.strftime("%Y%m%d")
    pre = "gec00" if mem == "c00" else "gep" + mem[1:]
    if init >= pd.Timestamp("2020-09-23"):
        return [((f"{OP}/gefs.{day}/00/atmos/pgrb2sp25/{pre}.t00z.pgrb2s.0p25.f{h:03d}" if h <= 240
                  else f"{OP}/gefs.{day}/00/atmos/pgrb2ap5/{pre}.t00z.pgrb2a.0p50.f{h:03d}"), {h})
                for h in sorted(hours)]
    return [(f"{OP}/gefs.{day}/00/pgrb2a/{pre}.t00z.pgrb2af{h:03d}" if h >= 100 else
             f"{OP}/gefs.{day}/00/pgrb2a/{pre}.t00z.pgrb2af{h:02d}", {h}) for h in sorted(hours)]


def fetch_run(init: pd.Timestamp, source: str) -> dict:
    """-> {member: DataFrame(hour, city, temp_k)} with every HTTP request issued in parallel."""
    tasks = [(m, url, want) for m in MEMBERS for url, want in plan(init, source, m)]
    idx = list(POOL.map(lambda t: get(t[1] + ".idx"), tasks))
    ranges = []
    for (m, url, want), text in zip(tasks, idx):
        if text is None:
            continue
        seen = set()
        for start, end, h, _ in parse_idx(text.decode()):
            if h in want and h not in seen:
                seen.add(h)
                ranges.append((m, h, url, start, end if end is not None else start + 5_000_000))
    blobs = list(POOL.map(lambda r: get(r[2], (r[3], r[4])), ranges))
    out = {}
    for (m, h, *_), b in zip(ranges, blobs):
        if b is None:
            continue
        with _DECODE_LOCK:
            vals, lats, lons = decode([b])
            ix = city_index(lats, lons)
        out.setdefault(m, []).extend((h, city, float(vals[0][i, j])) for (city, _), (i, j) in zip(CITIES.items(), ix))
    return {m: pd.DataFrame(v, columns=["hour", "city", "temp_k"]) for m, v in out.items()}


def run_values(init: pd.Timestamp, source: str, mem: str = "c00"):
    """Single member (kept for spot checks)."""
    return fetch_run(init, source).get(mem)


def daily_means(init: pd.Timestamp, v: pd.DataFrame) -> pd.DataFrame:
    v = v.copy()
    v["valid_utc"] = init + pd.to_timedelta(v.hour, unit="h")
    v["valid_et_date"] = (v.valid_utc - pd.Timedelta(hours=5) - pd.Timedelta(minutes=1)).dt.normalize()
    g = v.groupby(["valid_et_date", "city"]).agg(temp_k=("temp_k", "mean"), n=("temp_k", "size")).reset_index()
    g = g[g.n == 4]
    g["temp_f"] = ((g.temp_k - 273.15) * 9 / 5 + 32).round(2)
    g["init_date"] = init.normalize()
    g["lead_days"] = (g.valid_et_date - g.init_date).dt.days
    return g[["init_date", "valid_et_date", "lead_days", "city", "temp_f"]]


def _one(args):
    d, source = args
    per = {}
    for m, v in fetch_run(d, source).items():
        if len(v):
            per[m] = daily_means(d, v).set_index(["init_date", "valid_et_date", "lead_days", "city"]).temp_f
    if "c00" not in per:
        return d, None
    out = per["c00"].rename("temp_c00").to_frame()
    ens = pd.concat(per.values(), axis=1)
    out["temp_ens"] = ens.mean(axis=1).round(2)
    out["n_mem"] = ens.notna().sum(axis=1)
    return d, out.reset_index()


def collect(dates, source: str, out_file: Path, workers=4):
    """Fetch runs in `workers` processes (GRIB decoding is CPU-bound); resumable."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file, usecols=["init_date"]).init_date.unique())
    todo = [d for d in dates if d.strftime("%Y-%m-%d") not in done]
    print(f"{source}: {len(todo)} runs to fetch -> {out_file.name}", flush=True)
    t0 = time.time()
    frames, missing = [], []
    with cf.ProcessPoolExecutor(workers) as ex:
        for k, (d, df) in enumerate(ex.map(_one, [(d, source) for d in todo])):
            if df is None or df.empty:
                missing.append(str(d.date()))
            else:
                frames.append(df)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(todo)} runs, {time.time() - t0:.0f}s", flush=True)
                _flush(frames, out_file)
                frames = []
    _flush(frames, out_file)
    if missing:
        print(f"missing runs ({len(missing)}): {missing[:20]}...")


def _flush(frames, out_file):
    if not frames:
        return
    df = pd.concat(frames)
    df["init_date"] = df.init_date.dt.strftime("%Y-%m-%d")
    df["valid_et_date"] = df.valid_et_date.dt.strftime("%Y-%m-%d")
    df.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)


def winter_days(year: int, start=None, end=None):
    d = pd.date_range(start or f"{year}-01-01", end or f"{year}-12-31", freq="D")
    return [x for x in d if x.month in WINTER_MONTHS]


if __name__ == "__main__":
    src = sys.argv[1]
    if src == "reforecast":
        y = int(sys.argv[2])
        half = sys.argv[3] if len(sys.argv) > 3 else ""
        days = winter_days(y)
        if half == "a":
            days = [d for d in days if d.month <= 4]
        elif half == "b":
            days = [d for d in days if d.month >= 10]
        collect(days, "reforecast", OUT / f"reforecast_ens5_{y}{half}.csv")
    else:
        a, b = sys.argv[2], sys.argv[3]
        days = [x for x in pd.date_range(a, b, freq="D")]
        tag = sys.argv[4] if len(sys.argv) > 4 else f"{a[:4]}"
        collect(days, "operational", OUT / f"operational_ens5_{tag}.csv")
