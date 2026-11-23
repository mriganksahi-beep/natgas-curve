"""NOAA CPC 6-10 and 8-14 day temperature outlooks at our 17 gas cities (free, 2012-09 onward).

Each daily shapefile holds polygons with the favoured tercile category ('Above', 'Below',
'Normal') and its probability (%). A city outside every polygon is 'EC' (equal chances,
33/33/33). We record, per issue date, product and city: the category and probability, plus a
signed score  p_above - p_below  under the usual tercile convention (favoured category gets
`prob`, the opposite category gets 33 - (prob - 33)/2, normal gets the rest).

Run:  python -m src.collect.cpc 2012-09-11 2026-11-22  [out_tag]
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.fetch import CITIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/forecasts/cpc"
BASE = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst"


def get(url, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def signed(cat: str, p: float) -> float:
    """p_above - p_below. Favoured category gets p; the opposite tercile gets
    max(0, 33.3 - (p - 33.3)/2) (CPC convention: normal takes about half the excess)."""
    if cat in (None, "EC") or pd.isna(p):
        return 0.0
    if p <= 33.4:  # weakest contour: barely tilted
        return 0.0
    opp = max(0.0, 33.3 - (p - 33.3) / 2)
    if cat == "Above":
        return p - opp
    if cat == "Below":
        return opp - p
    return 0.0  # 'Normal' favoured: symmetric


def one(date: pd.Timestamp, product: str):
    import shapefile
    from shapely.geometry import Point, shape
    zf = None
    for attempt in range(4):
        blob = get(f"{BASE}/{product}temp_{date:%Y%m%d}.zip")
        if blob is None:
            return []
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
            break
        except zipfile.BadZipFile:
            time.sleep(10 * (attempt + 1))
    if zf is None:
        print(f"bad zip after retries: {product} {date:%Y-%m-%d}", flush=True)
        return []
    base = [n[:-4] for n in zf.namelist() if n.lower().endswith(".shp")][0]
    r = shapefile.Reader(shp=io.BytesIO(zf.read(base + ".shp")), dbf=io.BytesIO(zf.read(base + ".dbf")))
    fields = [f[0] for f in r.fields[1:]]
    polys = []
    for sr in r.shapeRecords():
        rec = dict(zip(fields, sr.record))
        try:
            g = shape(sr.shape.__geo_interface__)
        except Exception:
            continue
        polys.append((g, rec))
    rows = []
    for city, (lat, lon, _, _) in CITIES.items():
        pt = Point(lon, lat)
        hits = [rec for g, rec in polys if g.covers(pt)]
        # nested polygons: the highest probability is the innermost contour
        best = max(hits, key=lambda x: float(x.get("Prob") or 0)) if hits else None
        cat = best.get("Cat") if best else "EC"
        p = float(best.get("Prob")) if best else 33.3
        rec0 = polys[0][1] if polys else {}
        rows.append({"issue_date": date.date(), "product": product,
                     "start": rec0.get("Start_Date"), "end": rec0.get("End_Date"),
                     "city": city, "cat": cat, "prob": p, "signed": round(signed(cat, p), 2)})
    return rows


def collect(start, end, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"cpc_outlooks_{tag}.csv"
    jobs = [(d, p) for d in pd.date_range(start, end) for p in ("610", "814")]
    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        for k, res in enumerate(ex.map(lambda a: one(*a), jobs)):
            rows.extend(res)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["issue_date", "product", "city"])
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows, {df.issue_date.nunique()} dates -> {out}")


if __name__ == "__main__":
    collect(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all")
