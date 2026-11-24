"""Data for the spread->storage map (agent storage_map, tool build, 24 Nov 2026).

Hard cuts (SPEC rules): futures and storage rows dated <= 2021-12-31 only, filtered on load.
Pre-2010 data is no longer locked (v4 spent it) and IS used here: EIA front-4 futures (1994-) and
EIA monthly US working gas (NG.N5020US2.M).

One row per winter Y (March delivery year), Y = 1994..2021:
  oexp     March option expiry = trading day before the H futures LTD (LTD = 3 business days before 1 Mar)
  S        H - J settle on oexp ($/MMBtu)  [EIA c1-c2 for Y <= 2010, CME curve file for Y >= 2011]
  H, J, K  March, April, May settles on oexp;  JK = J - K;  lnHJ = ln(H/J)
  em_m     end-March US working gas, monthly survey (Bcf)
  em_w     end-March Lower-48 working gas, weekly (last report dated Mar 25-31), Y >= 2010
  avg5_m   mean of em_m over Y-1..Y-5;  x = em_m - avg5_m (Bcf)  [primary driver]
  xw       weekly analogue (weekly em_w minus mean of prior weekly levels, monthly used where weekly absent)
  xpct     100 * x / avg5_m;  cap = working-gas design capacity (N5290 total minus base gas), Bcf
"""
from pathlib import Path

import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Integration (orchestrator, 24 Nov 2026): the cut is read from MAP_DATA_CUTOFF / RND_DATA_CUTOFF (default
# the dev cut 2021-12-31) so the locked/live run can add winters 2022+ WITHOUT code changes. Walk-forward is
# still enforced by storage_map.fit_map(last_winter): winter Y is scored with fit_map(Y - 1).
CUT = pd.Timestamp(os.environ.get("MAP_DATA_CUTOFF", os.environ.get("RND_DATA_CUTOFF", "2021-12-31")))
CACHE = ROOT / "data/processed/research_tool_map"
CACHE.mkdir(parents=True, exist_ok=True)
Y_FIRST = 1994
Y_LAST = CUT.year if CUT >= pd.Timestamp(CUT.year, 3, 1) else CUT.year - 1


def _ltd(Y):
    return pd.Timestamp(Y, 3, 1) - pd.offsets.BDay(3)


def eia_futures():
    e = pd.read_csv(ROOT / "data/raw/eia_futures_c1_c4.csv", parse_dates=["date"])
    e = e[e.date <= CUT]                                     # cut on load
    return e.set_index("date").sort_index()


def cme_futures():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.date <= CUT]                                     # cut on load
    c = c[c.contract_month.isin([3, 4, 5]) & c.contract_year.between(2011, Y_LAST)].dropna(subset=["settle"])
    return c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle").sort_index()


def monthly_storage():
    m = pd.read_csv(ROOT / "data/raw/eia/storage_monthly_series_bulk.csv")
    m = m[m.month <= CUT.year * 100 + CUT.month]                                 # cut on load
    out = {}
    for sid, name in (("NG.N5020US2.M", "wg"), ("NG.N5290US2.M", "tot_cap"), ("NG.N5010US2.M", "base")):
        s = m[m.series_id == sid].dropna(subset=["value"])
        out[name] = pd.Series(s.value.values / 1000.0, index=s.month.values).sort_index()   # Bcf
    return out


def weekly_storage():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUT]                              # cut on load
    s = s.set_index("week_ending")["storage_bcf"].sort_index()
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    return m.groupby(m.index.year).last()


def expiry_prices():
    e = eia_futures()
    c = cme_futures()
    rows = []
    for Y in range(Y_FIRST, Y_LAST + 1):
        ltd = _ltd(Y)
        d_e = e.index[e.index < ltd]
        oexp_e = d_e[-1]
        r = {"Y": Y, "oexp_eia": oexp_e, "H_eia": e.at[oexp_e, "c1"], "J_eia": e.at[oexp_e, "c2"],
             "K_eia": e.at[oexp_e, "c3"]}
        if (Y, 3) in c.columns and (Y, 4) in c.columns:
            d = c[[(Y, 3), (Y, 4), (Y, 5)]].dropna()
            d = d[(d.index >= pd.Timestamp(Y, 1, 15)) & (d.index < pd.Timestamp(Y, 3, 1))]
            cltd = d.index.max()
            oexp_c = d.index[-2]
            r.update({"oexp_cme": oexp_c, "ltd_cme": cltd, "H_cme": d.at[oexp_c, (Y, 3)],
                      "J_cme": d.at[oexp_c, (Y, 4)], "K_cme": d.at[oexp_c, (Y, 5)]})
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Y")
    use_cme = t.H_cme.notna() if "H_cme" in t else pd.Series(False, index=t.index)
    for k in "HJK":
        t[k] = np.where(use_cme, t.get(f"{k}_cme"), t[f"{k}_eia"])
    t["oexp"] = np.where(use_cme, t.get("oexp_cme"), t.oexp_eia)
    t["src"] = np.where(use_cme, "cme", "eia")
    t["S"] = t.H - t.J
    t["S_eia"] = t.H_eia - t.J_eia
    t["JK"] = t.J - t.K
    t["lnHJ"] = np.log(t.H / t.J)
    return t


def winter_table(refresh=False):
    p = CACHE / ("winter_table.csv" if CUT == pd.Timestamp("2021-12-31") else f"winter_table_{CUT:%Y%m%d}.csv")
    if p.exists() and not refresh:
        return pd.read_csv(p, index_col="Y", parse_dates=["oexp", "oexp_eia", "oexp_cme", "ltd_cme"])
    ms = monthly_storage()
    wg, cap = ms["wg"], ms["tot_cap"] - ms["base"]
    em_m = {Y: wg.get(Y * 100 + 3, np.nan) for Y in range(1970, Y_LAST + 1)}
    em_w = weekly_storage().to_dict()
    t = expiry_prices()
    t["em_m"] = [em_m[Y] for Y in t.index]
    t["avg5_m"] = [np.mean([em_m[Y - k] for k in range(1, 6)]) for Y in t.index]
    t["x"] = t.em_m - t.avg5_m
    t["xpct"] = 100 * t.x / t.avg5_m
    t["cap"] = [cap.get(Y * 100 + 3, np.nan) for Y in t.index]
    t["em_w"] = [em_w.get(Y, np.nan) for Y in t.index]
    # weekly analogue: weekly level minus mean of prior weekly levels; prior years before 2010 filled with the
    # monthly level shifted by the mean (weekly - monthly) offset on the overlap (documented in the report)
    off = np.nanmean([em_w[Y] - em_m[Y] for Y in em_w if Y in em_m])
    lev = {Y: em_w.get(Y, em_m[Y] + off) for Y in em_m}
    t["avg5_w"] = [np.mean([lev[Y - k] for k in range(1, 6)]) for Y in t.index]
    t["xw"] = t.em_w - t.avg5_w
    t["wm_offset"] = off
    t = t[t.index <= Y_LAST]
    assert t.oexp.max() <= CUT
    t.to_csv(p)
    return t
