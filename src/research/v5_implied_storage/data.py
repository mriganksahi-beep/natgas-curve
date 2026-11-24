"""Data layer for v5_implied_storage (study v5).

Hard cuts
---------
* Options: raw near-money file, root G4X, March contract (cm = 3), rows dated <= 2021-12-31 only.
* Futures: H and J settles dated <= 2021-12-31. ONE documented exception (brief rule 1): for positions
  opened on or before 31 Dec 2021 on the March-2022 contract, the H22/J22 settles from 3 Jan 2022 up to
  the March-2022 option expiry (Feb 2022) are read, to value those positions' outcome and path.
  No other 2022 row is read. Storage, weather and Model 2 rows are cut at 2021-12-31.

Definitions
-----------
Y: delivery year of the March/April pair; winter = Nov Y-1 .. Feb Y.
S = H - J ($/MMBtu); one G4X option = 10,000 MMBtu.
Option expiry = trading day before the H futures last trade (LTD = 3 business days before 1 March).
x = end-March storage - mean of prior years' end-March storage (<= 5 years), as in
    src/research/widowmaker_insurance (payout curve driver).
Decision dates follow Model 2 / Model 3 timing: storage week ending Friday F (Nov 1 .. Feb 14), decision
at the settle of the first trading day >= F + 7.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
CUT_2022_OUTCOME = pd.Timestamp("2022-03-31")
CACHE = ROOT / "data/processed/research_v5_implied_storage"
CACHE.mkdir(parents=True, exist_ok=True)
MULT = 10_000


def load_options():
    p = CACHE / "g4x_march_dev.pkl"
    if p.exists():
        return pd.read_pickle(p)
    o = pd.read_csv(ROOT / "data/raw/options/ng_options_near_money_v2.csv.gz", parse_dates=["date"])
    o = o[o.date <= CUT]                                  # cut on load
    o = o[(o.root == "G4X") & (o.cm == 3) & (o.settle > 0)].copy()
    o["Y"] = o.cy.astype(int)
    o["K"] = o.k_raw / 1000.0
    o = o[["date", "Y", "cp", "K", "settle", "volume", "oi"]].sort_values(["date", "Y", "cp", "K"])
    assert o.date.max() <= CUT
    o.to_pickle(p)
    return o


def load_futures():
    """Daily H, J, S per delivery year Y in 2013..2022 (2022: see module docstring exception)."""
    p = CACHE / "hj_futures.pkl"
    if p.exists():
        return pd.read_pickle(p)
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.contract_month.isin([3, 4]) & c.contract_year.between(2011, 2022)].dropna(subset=["settle"])
    dev = c[c.date <= CUT]
    # exception: March/April 2022 contracts only, Jan-Feb 2022 only (outcome of positions opened <= 2021)
    ext = c[(c.date > CUT) & (c.date <= CUT_2022_OUTCOME) & (c.contract_year == 2022) & (c.date.dt.month <= 2)]
    c = pd.concat([dev, ext])
    px = c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle").sort_index()
    out = {}
    for Y in range(2011, 2023):
        if (Y, 3) not in px.columns:
            continue
        d = pd.DataFrame({"H": px[(Y, 3)], "J": px[(Y, 4)]}).dropna()
        d = d[d.index >= pd.Timestamp(Y - 1, 8, 1)]
        d["S"] = d.H - d.J
        ltd = d.index.max()
        assert ltd.month == 2 and ltd < pd.Timestamp(Y, 3, 1), (Y, ltd)
        d = d.loc[:ltd]
        out[Y] = d
    pd.to_pickle(out, p)
    return out


def expiries(fut=None):
    fut = load_futures() if fut is None else fut
    rows = []
    for Y, d in fut.items():
        ltd = d.index[-1]
        oexp = d.index[-2]
        rows.append({"Y": Y, "ltd": ltd, "oexp": oexp, "S_oexp": d.S.iloc[-2], "S_ltd": d.S.iloc[-1],
                     "J_ltd": d.J.iloc[-1]})
    return pd.DataFrame(rows).set_index("Y")


def load_storage():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUT]
    return s.set_index("week_ending")["storage_bcf"].sort_index()


def avg5_table():
    s = load_storage()
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    em = m.groupby(m.index.year).last()
    rows = []
    for Y in range(2011, 2023):
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index]
        rows.append({"Y": Y, "avg5": float(np.mean(prior)), "n_prior": len(prior),
                     "end_march": float(em[Y]) if Y in em.index else np.nan})
    t = pd.DataFrame(rows).set_index("Y")
    t["x"] = t.end_march - t.avg5
    return t


def payout_table():
    """Payout-curve fitting table: x and the expiry-day spread (H LTD settle) for Y <= 2021."""
    t = avg5_table()
    ex = expiries()
    t = t.join(ex[["S_ltd", "S_oexp", "J_ltd"]])
    t.loc[t.index >= 2022, ["S_ltd", "S_oexp", "J_ltd"]] = np.nan  # never used for fitting
    return t


def decision_dates(first_Y=2013, last_Y=2022, min_tdays_to_oexp=5):
    fut = load_futures()
    ex = expiries(fut)
    s = load_storage()
    rows = []
    for F in s.index:
        if F.month not in (11, 12, 1, 2) or (F.month == 2 and F.day > 14):
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y < first_Y or Y > last_Y or Y not in fut:
            continue
        d = fut[Y]
        pos = d.index.searchsorted(F + pd.Timedelta(days=7))
        if pos >= len(d):
            continue
        obs = d.index[pos]
        if obs > CUT:
            continue
        oexp = ex.oexp[Y]
        tdays = d.index.searchsorted(oexp) - pos
        if tdays < min_tdays_to_oexp:
            continue
        rows.append({"origin": F, "obs": obs, "Y": Y, "storage": s[F], "S": d.S.iloc[pos], "H": d.H.iloc[pos],
                     "J": d.J.iloc[pos], "tdays": tdays, "T": tdays / 252.0, "oexp": oexp,
                     "wte": (oexp - obs).days / 7.0})
    df = pd.DataFrame(rows).sort_values("obs").reset_index(drop=True)
    assert df.obs.max() <= CUT and df.origin.max() <= CUT
    return df


def winter_label(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


def load_m2():
    m = pd.read_csv(ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv",
                    parse_dates=["origin", "target"])
    m = m[m.origin <= CUT].copy()
    m["Y"] = m.target.dt.year
    return m


PCOLS = [f"p{k}" for k in range(1, 100)]
