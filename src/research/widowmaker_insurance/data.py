"""Data layer for the widowmaker-as-insurance study (no data dated 2022-01-01 or later).

Definitions
-----------
Delivery year Y: the March (H) and April (J) contracts of year Y; winter = Nov Y-1 .. Feb Y.
Spread S = H - J ($/MMBtu). One spread = 1 H short + 1 J long (short) or reverse.
Expiry (LTD): last settlement of the H contract in the data (only used for Y <= 2021;
    the 2022 contract expires in the holdout and is never read).
avg5[Y]: mean end-March storage over Y-1..Y-5, using whatever prior years exist
    (storage starts Jan 2010, so Y=2011 has 1 prior year, 2012 has 2, 2013 has 3,
    2014 has 4, 2015+ has 5). n_prior is kept so variants can require >= 2.
x[Y] = end-March storage[Y] - avg5[Y]  (Bcf), "end-March surplus".

Weekly decision points (as src/model3_data.py): storage week ending Friday F is public
Thursday F+6; the decision uses the settle of the first trading day >= F+7.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_widowmaker_insurance"
CACHE.mkdir(parents=True, exist_ok=True)
TICK = 10.0           # $ per tick per contract
MULT = 10_000         # MMBtu per contract -> $ per $1 of spread
COST_TICKS = 4        # per contract per side
ROUND_TRIP = 2 * 2 * COST_TICKS * TICK   # 2 contracts x 2 sides = $160 per spread


def load_storage():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUTOFF]
    return s.set_index("week_ending")["storage_bcf"].sort_index()


def end_march(s=None):
    s = load_storage() if s is None else s
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    return m.groupby(m.index.year).last()


def avg5_table(min_prior=1):
    em = end_march()
    rows = []
    for Y in range(2011, 2023):
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index]
        if len(prior) < min_prior:
            continue
        rows.append({"Y": Y, "avg5": float(np.mean(prior)), "n_prior": len(prior),
                     "end_march": float(em[Y]) if Y in em.index else np.nan})
    t = pd.DataFrame(rows).set_index("Y")
    t["x"] = t.end_march - t.avg5
    return t


def load_px():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & c.contract_month.isin([3, 4]) & (c.contract_year <= 2022)]
    return c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                   values="settle")


def spreads():
    """Daily H-J per delivery year, from 1 Aug of Y-1 to expiry (or the cutoff for Y=2022)."""
    px = load_px()
    out = {}
    for Y in range(2011, 2023):
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        h, j = px[(Y, 3)], px[(Y, 4)]
        sp = (h - j).dropna()
        sp = sp[sp.index >= pd.Timestamp(Y - 1, 8, 1)]
        out[Y] = pd.DataFrame({"S": sp, "H": h[sp.index], "J": j[sp.index]})
    return out


def expiry_table(sp=None):
    """Expiry-day spread and the mean of the last 5 settles, for Y <= 2021 only."""
    sp = spreads() if sp is None else sp
    rows = []
    for Y, d in sp.items():
        if Y >= 2022:
            continue
        ltd = d.index.max()
        assert ltd.month == 2, (Y, ltd)
        rows.append({"Y": Y, "ltd": ltd, "S_exp": d.S.iloc[-1], "S_exp5": d.S.iloc[-5:].mean(),
                     "J_exp": d.J.iloc[-1], "H_exp": d.H.iloc[-1]})
    return pd.DataFrame(rows).set_index("Y")


def winter_label(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


def weekly_panel(sp=None, first_Y=2011, last_origin_md=(2, 14)):
    """One row per storage week F (Nov 1 .. Feb 14 origins) with the decision settle at F+7.
    Includes the expiry outcome (Y <= 2021) and 4-week-ahead spread (capped at expiry)."""
    sp = spreads() if sp is None else sp
    ex = expiry_table(sp)
    s = load_storage()
    rows = []
    for F in s.index:
        if F.month not in (11, 12, 1, 2):
            continue
        if F.month == 2 and F.day > last_origin_md[1]:
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y < first_Y or Y not in sp:
            continue
        d = sp[Y]
        pos = d.index.searchsorted(F + pd.Timedelta(days=7))
        if pos >= len(d):
            continue
        obs = d.index[pos]
        if obs > CUTOFF:
            continue
        ltd = ex.ltd.get(Y, pd.NaT)
        if pd.notna(ltd) and obs >= ltd:
            continue
        r = {"origin": F, "obs": obs, "Y": Y, "winter": winter_label(Y), "S": d.S.iloc[pos],
             "J": d.J.iloc[pos], "H": d.H.iloc[pos], "storage": s[F]}
        # weeks to expiry (2022 contract: nominal LTD = 3 business days before 1 March)
        ltd_nom = ltd if pd.notna(ltd) else pd.Timestamp(Y, 3, 1) - pd.offsets.BDay(3)
        r["wte"] = (ltd_nom - obs).days / 7
        # 1 week earlier settle (for spike definition)
        p1 = d.index.searchsorted(obs - pd.Timedelta(days=7))
        r["S_prev1w"] = d.S.iloc[p1] if p1 < pos else np.nan
        # outcomes
        r["S_exp"] = ex.S_exp.get(Y, np.nan)
        p4 = d.index.searchsorted(obs + pd.Timedelta(days=28))
        if p4 < len(d) and d.index[p4] <= CUTOFF:
            r["S_4w"], r["date_4w"] = d.S.iloc[p4], d.index[p4]
        elif pd.notna(ltd):  # 4 weeks run past expiry -> exit at expiry
            r["S_4w"], r["date_4w"] = d.S.iloc[-1], d.index[-1]
        else:
            r["S_4w"], r["date_4w"] = np.nan, pd.NaT
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("obs").reset_index(drop=True)
    assert df.obs.max() <= CUTOFF and df.origin.max() <= CUTOFF
    return df


def load_m2():
    m = pd.read_csv(ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv",
                    parse_dates=["origin", "target"])
    m = m[m.origin <= CUTOFF].copy()
    m["Y"] = m.target.dt.year
    return m


PCOLS = [f"p{k}" for k in range(1, 100)]
