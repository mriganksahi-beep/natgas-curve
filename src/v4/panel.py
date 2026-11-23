"""Study v4 shared daily panel (built once by the orchestrator; agents read it, never rebuild it).

Rows: trading days. Two files, because the 2000-2010 sample is the locked final exam:
  data/processed/v4/panel_dev.csv          trading days from 2010-06-07 (Databento curve)
  data/processed/v4_locked/panel_pre2010.csv  trading days 2000-01-03 .. 2010-06-04 (EIA c1-c4)
Both are git-ignored (derived from licensed data / locked).

Timing (no look-ahead), for the settle on trading day d (14:30 ET):
  * GEFS 00Z run initialised on d: public ~06:00 ET  -> known.
  * CPC outlook issued on d: 15:00-16:00 ET           -> first usable at the NEXT trading day.
  * Realised weather of days < d: known; day d itself: not complete -> use the d run's lead 0.
  * EIA storage report: Thursday 10:30 ET            -> known on its release day.

Forecast variables (HDD anomaly units, gas-weighted, vs dev-years model climatology; ensemble mean
unless suffixed _c00). Windows relative to d: w1 = d+1..d+5, w2 = d+6..d+10, w3 = d+11..d+15.
  lvl_w*   : sum of the d run's anomalies over the window (the forecast level)
  news_w*  : same valid dates, d run minus the run of the previous trading day dp (dp's anomaly
             beyond its lead 15 counts as climatology = 0)
  news_w0  : surprise on the days that passed, (dp, d]: realised (days < d) or the d run's lead 0
             (day d) minus dp's forecast for those days
  news_tot : news_w0 + news_w1 + news_w2 + news_w3
  real_w1, real_w2, real_w3 : realised anomaly sums over the same windows (OUTCOMES - not known at d)
Positive anomaly / news = colder than normal = more heating demand.

Prices (settles; changes on the same contracts between dp and d):
  front, second : nearest contract with >= 2 trading days to expiry, and the next one
  r_front, r_second : 100 * dlog
  hj      : March - April spread of the target winter (the nearest March whose contract is live
            and whose spread is quoted), $/MMBtu; d_hj its change (same contracts)
  strip   : mean of the target winter's live Dec-Mar contracts; r_strip 100*dlog   (dev only)
Options (dev only, from 2010 for ON, winter 2013+ for G4X; r = 0, no discounting):
  iv_front, iv_h, iv_j : Black-76 ATM implied vol (annualised) from ON settles, interpolated to
                         the money (straddle at the two strikes around F)
  nv_hj  : Bachelier (normal) implied vol of the target March-April spread from G4X March options,
           $/MMBtu per sqrt(year); T = years to the March option expiry
"""
from __future__ import annotations

from math import erf, exp, log, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from src.v4 import forecasts as F

ROOT = Path(__file__).resolve().parents[2]
OUT_DEV = ROOT / "data/processed/v4"
OUT_LOCK = ROOT / "data/processed/v4_locked"
DEV_START = pd.Timestamp("2010-06-07")


def addm(cid: int, k: int) -> int:
    y, m = divmod(cid, 100)
    m += k
    y += (m - 1) // 12
    return y * 100 + (m - 1) % 12 + 1


# ------------------------------------------------------------------------------- prices
def dev_curve():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"]).dropna(subset=["settle"])
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    days = px.index
    exp_ = {}
    for cid in px.columns:
        y, m = divmod(cid, 100)
        first = pd.Timestamp(y, m, 1)
        p = days.searchsorted(first)
        exp_[cid] = days[p - 3] if 3 <= p < len(days) else first - pd.offsets.BDay(3)
    return px, pd.Series(exp_)


def pre2010_curve():
    """EIA c1-c4 mapped to contracts (validated: 97% exact vs CME on 2010-24)."""
    e = pd.read_csv(ROOT / "data/raw/eia_futures_c1_c4.csv", parse_dates=["date"]).set_index("date")
    e = e[(e.index >= "1999-10-01") & (e.index < DEV_START)]
    rows = {}
    for d, r in e.iterrows():
        for k in range(1, 5):
            c1 = None
            for j in range(3):
                cid = addm(d.year * 100 + d.month, j)
                y, m = divmod(cid, 100)
                if pd.Timestamp(y, m, 1) - pd.offsets.BDay(3) >= d:
                    c1 = cid
                    break
            rows.setdefault(d, {})[addm(c1, k - 1)] = r[f"c{k}"]
    px = pd.DataFrame(rows).T.sort_index()
    px.index.name = "date"
    exp_ = pd.Series({cid: pd.Timestamp(cid // 100, cid % 100, 1) - pd.offsets.BDay(3) for cid in px.columns})
    return px, exp_


def price_block(px: pd.DataFrame, exp_: pd.Series, with_strip: bool) -> pd.DataFrame:
    days = px.index
    tpos = pd.Series(np.arange(len(days)), index=days)

    def tleft(cid, d):
        e = exp_[cid]
        if e > days[-1]:
            return 99
        if e < days[0]:
            return -1
        return tpos[days[days.searchsorted(e)] if e not in tpos.index else e] - tpos[d]

    out = []
    for i in range(1, len(days)):
        d, dp = days[i], days[i - 1]
        live = [c for c in px.columns if exp_[c] >= d and tleft(c, d) >= 2 and np.isfinite(px.at[d, c])]
        if len(live) < 2:
            continue
        live.sort()
        front, second = live[0], live[1]
        r = {"date": d, "prev_date": dp, "front": front, "second": second,
             "front_days_left": tleft(front, d)}
        for name, cid in (("front", front), ("second", second)):
            a, b = px.at[dp, cid], px.at[d, cid]
            r[f"px_{name}"] = b
            r[f"r_{name}"] = 100 * (log(b) - log(a)) if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 else np.nan
        # target March: nearest March contract still live on d with both legs quoted
        y = d.year if d.month <= 2 or (d.month == 3 and exp_.get(d.year * 100 + 3, d) >= d) else d.year + 1
        H, J = y * 100 + 3, y * 100 + 4
        if H in px.columns and J in px.columns and exp_[H] >= d:
            s1, s0 = px.at[d, H] - px.at[d, J], px.at[dp, H] - px.at[dp, J]
            r.update({"hj_year": y, "hj": s1, "d_hj": s1 - s0 if np.isfinite(s0) else np.nan,
                      "hj_days_left": tleft(H, d)})
        if with_strip:
            strip = [c for c in (addm(H, -3), addm(H, -2), addm(H, -1), H) if c in live]
            if strip:
                a, b = px.loc[dp, strip], px.loc[d, strip]
                if a.notna().all() and b.notna().all():
                    r["r_strip"] = 100 * (log(b.mean()) - log(a.mean()))
                    r["strip_n"] = len(strip)
        out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------- forecasts
def forecast_block(days: pd.Series, prev: pd.Series) -> pd.DataFrame:
    a = F.gefs_anomalies()
    ens = a.set_index(["init_date", "valid_date"])["anom_ens"]
    c00 = a.set_index(["init_date", "valid_date"])["anom_c00"]
    ens_d = {k: g.droplevel(0) for k, g in ens.groupby(level=0)}
    c00_d = {k: g.droplevel(0) for k, g in c00.groupby(level=0)}
    real = F.realised_anomaly()
    W = {"w1": (1, 5), "w2": (6, 10), "w3": (11, 15)}
    rows = []
    for d, dp in zip(days, prev):
        r = {"date": d}
        for tag, runs in (("", ens_d), ("_c00", c00_d)):
            new, old = runs.get(d), runs.get(dp)
            if new is None or (new.index.max() - d).days < 15:
                continue
            if old is not None and (old.index.max() - dp).days < 15:
                old = None  # incomplete previous run: news undefined rather than mis-measured
                r[f"old_run_incomplete{tag}"] = True
            tot = 0.0
            for w, (lo, hi) in W.items():
                vd = pd.date_range(d + pd.Timedelta(days=lo), d + pd.Timedelta(days=hi))
                nv = new.reindex(vd)
                if nv.isna().any():
                    break
                ov = old.reindex(vd).fillna(0.0) if old is not None else pd.Series(0.0, index=vd)
                r[f"lvl_{w}{tag}"] = nv.sum()
                r[f"news_{w}{tag}"] = (nv - ov).sum() if old is not None else np.nan
                tot += r[f"news_{w}{tag}"]
                if tag == "":
                    rv = real.reindex(vd)
                    r[f"real_{w}"] = rv.sum() if rv.notna().all() else np.nan
            # passed days (dp, d]: realised before d, the d run's lead 0 for d itself
            if old is not None:
                past = pd.date_range(dp + pd.Timedelta(days=1), d)
                x = pd.concat([real.reindex(past[:-1]), new.reindex(past[-1:])])
                o = old.reindex(past)
                if x.notna().all() and o.notna().all():
                    r[f"news_w0{tag}"] = (x - o).sum()
                    if all(f"news_{w}{tag}" in r for w in W):
                        r[f"news_tot{tag}"] = r[f"news_w0{tag}"] + sum(r[f"news_{w}{tag}"] for w in W)
        r["real_d"] = real.get(d, np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["gefs_system"] = out.date.map(a.drop_duplicates("init_date").set_index("init_date").system)
    return out


def cpc_block(days: pd.Series, prev: pd.Series) -> pd.DataFrame:
    """CPC score known at settle d = the latest issue made on or before dp (issued after its settle)."""
    c = F.cpc_scores().sort_index()
    idx = c.index
    rows = []
    for d, dp in zip(days, prev):
        p = idx.searchsorted(dp, side="right") - 1
        if p < 0:
            rows.append({"date": d})
            continue
        cur = c.iloc[p]
        r = {"date": d, "cpc_issue": idx[p], "cpc610": -cur.get("cpc610", np.nan), "cpc814": -cur.get("cpc814", np.nan)}
        rows.append(r)
    out = pd.DataFrame(rows)
    # sign flipped above: positive = colder (more HDD), matching the GEFS convention
    for k in ("cpc610", "cpc814"):
        out[f"d_{k}"] = out[k].diff().where(out.cpc_issue != out.cpc_issue.shift())
        out[f"d_{k}"] = out[f"d_{k}"].fillna(0.0)
    return out


# ------------------------------------------------------------------------------- options
def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _black_straddle(Fw, K, T, s):
    if s <= 0 or T <= 0:
        return abs(Fw - K)
    d1 = (log(Fw / K) + 0.5 * s * s * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    c = Fw * _ncdf(d1) - K * _ncdf(d2)
    return c + (c - (Fw - K))


def _bach_straddle(Fw, K, T, s):
    if s <= 0 or T <= 0:
        return abs(Fw - K)
    v = s * sqrt(T)
    d = (Fw - K) / v
    return (Fw - K) * (2 * _ncdf(d) - 1) + 2 * v * exp(-0.5 * d * d) / sqrt(2 * pi)


def _solve(fn, target, lo, hi):
    if not np.isfinite(target) or fn(lo) > target or fn(hi) < target:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def option_block(prices: pd.DataFrame, px: pd.DataFrame, exp_: pd.Series) -> pd.DataFrame:
    o = pd.read_csv(ROOT / "data/raw/options/ng_options_near_money.csv.gz", parse_dates=["date"])
    o = o[o.root.isin(["ON", "G4X"])]
    o["cid"] = o.cy * 100 + o.cm
    o["K"] = o.k_raw / 1000.0
    st = o.pivot_table(index=["root", "date", "cid", "K"], columns="cp", values="settle").dropna()
    st["straddle"] = st["C"] + st["P"]
    st = st["straddle"]
    groups = {k: g.droplevel([0, 1, 2]) for k, g in st.groupby(level=[0, 1, 2])}
    days = px.index

    def opt_expiry(cid):  # options expire the business day before the futures
        e = exp_[cid]
        p = days.searchsorted(e)
        return days[p - 1] if 1 <= p < len(days) and days[p] == e else e - pd.offsets.BDay(1)

    def atm_iv(root, d, cid, Fw, kind):
        g = groups.get((root, d, cid))
        if g is None or not np.isfinite(Fw):
            return np.nan
        T = (opt_expiry(cid) - d).days / 365.25
        if T <= 2 / 365:
            return np.nan
        ks = g.index.values
        lo, hi = ks[ks <= Fw], ks[ks >= Fw]
        if not len(lo) or not len(hi):
            return np.nan
        vals = []
        for K in {lo.max(), hi.min()}:
            if kind == "black":
                vals.append((K, _solve(lambda s: _black_straddle(Fw, K, T, s), g[K], 1e-4, 5.0)))
            else:
                vals.append((K, _solve(lambda s: _bach_straddle(Fw, K, T, s), g[K], 1e-5, 20.0)))
        vals = [(k, v) for k, v in vals if np.isfinite(v)]
        if not vals:
            return np.nan
        if len(vals) == 2 and vals[0][0] != vals[1][0]:
            (k0, v0), (k1, v1) = sorted(vals)
            return v0 + (v1 - v0) * (Fw - k0) / (k1 - k0)
        return vals[0][1]

    rows = []
    for _, r in prices.iterrows():
        d = r.date
        out = {"date": d}
        out["iv_front"] = atm_iv("ON", d, r.front, px.at[d, r.front], "black")
        if np.isfinite(r.get("hj_year", np.nan)):
            H, J = int(r.hj_year) * 100 + 3, int(r.hj_year) * 100 + 4
            out["iv_h"] = atm_iv("ON", d, H, px.at[d, H], "black")
            out["iv_j"] = atm_iv("ON", d, J, px.at[d, J], "black")
            out["nv_hj"] = atm_iv("G4X", d, H, r.hj, "bach")
            out["T_h"] = (opt_expiry(H) - d).days / 365.25
        rows.append(out)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- storage
def storage_block(days: pd.Series) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"]).set_index("week_ending")["storage_bcf"]
    ch = s.diff()
    rows = []
    for we, v in ch.dropna().items():
        prior = []
        for k in range(1, 6):
            t = we - pd.DateOffset(years=k)
            p = ch.index.searchsorted(t)
            cands = [q for q in (p - 1, p) if 0 <= q < len(ch)]
            if cands:
                q = min(cands, key=lambda q: abs(ch.index[q] - t))
                if abs(ch.index[q] - t) <= pd.Timedelta(days=3) and np.isfinite(ch.iloc[q]):
                    prior.append(ch.iloc[q])
        if len(prior) == 5:
            rows.append({"release": we + pd.Timedelta(days=6), "storage_week": we,
                         "storage_change": v, "storage_surprise_5y": v - np.mean(prior), "storage_level": s[we]})
    st = pd.DataFrame(rows)
    # attach to the first trading day on/after the (nominal Thursday) release
    dd = pd.DatetimeIndex(days)
    st["date"] = [dd[dd.searchsorted(x)] if dd.searchsorted(x) < len(dd) else pd.NaT for x in st.release]
    return st.dropna(subset=["date"]).drop_duplicates("date", keep="last")


# ------------------------------------------------------------------------------- build
def build():
    OUT_DEV.mkdir(parents=True, exist_ok=True)
    OUT_LOCK.mkdir(parents=True, exist_ok=True)
    # dev
    px, exp_ = dev_curve()
    pr = price_block(px, exp_, with_strip=True)
    fc = forecast_block(pr.date, pr.prev_date)
    cp = cpc_block(pr.date, pr.prev_date)
    op = option_block(pr, px, exp_)
    st = storage_block(pr.date)
    dev = (pr.merge(fc, on="date", how="left").merge(cp, on="date", how="left")
             .merge(op, on="date", how="left").merge(st[["date", "storage_change", "storage_surprise_5y", "storage_level"]],
                                                     on="date", how="left"))
    dev.to_csv(OUT_DEV / "panel_dev.csv", index=False)
    print("dev panel", dev.shape, dev.date.min().date(), dev.date.max().date())
    # locked pre-2010 (prices from EIA c1-c4; GEFS from 2000)
    pxl, expl = pre2010_curve()
    prl = price_block(pxl, expl, with_strip=False)
    prl = prl[prl.date >= "2000-01-03"]
    fcl = forecast_block(prl.date, prl.prev_date)
    lock = prl.merge(fcl, on="date", how="left")
    lock.to_csv(OUT_LOCK / "panel_pre2010.csv", index=False)
    print("locked panel", lock.shape, lock.date.min().date(), lock.date.max().date())
    return dev, lock


if __name__ == "__main__":
    build()
