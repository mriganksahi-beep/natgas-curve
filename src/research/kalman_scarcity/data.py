"""Weekly panel for the Kalman scarcity study (agent: kalman_scarcity).

Every loader cuts data at END = 2021-12-31 immediately after reading (BRIEF rule 1).

Observation date: for each Friday F, the first trading day on or after F (same convention
as src/model3_data.py), so the storage report for week F-7 (out Thursday 10:30 ET) is
public before the settlement used.

Target winter Y (Dec Y-1 .. Mar Y, plus Apr Y): the earliest Y whose March contract is
more than 5 trading days from expiry on the observation date.

Observed spreads ($/MMBtu), each dropped when any leg is within 5 trading days of expiry:
  DecApr, JanApr, FebApr, MarApr : month minus April of the same winter
  Strip : mean(live winter months Dec..Mar) minus mean(Apr..Oct of year Y)
tau_i : years from the observation date to the first day of delivery of the winter leg
        (for the strip: mean over the live winter months).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
CACHE = ROOT / "data/processed/research_kalman_scarcity"
END = pd.Timestamp("2021-12-31")
NAMES = ["DecApr", "JanApr", "FebApr", "MarApr", "Strip"]
DROP_DAYS = 5


def load_curve():
    c = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c["date"] <= END]  # holdout cut, first thing
    c = c[c["settle"] > 0]   # zero settlements are placeholders
    return c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle")


def load_storage():
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s["week_ending"] <= END]
    return s.set_index("week_ending")["storage_bcf"].sort_index()


def load_hdd():
    w = pd.read_csv(RAW / "weekly_degree_days_own.csv", parse_dates=["week_ending"])
    w = w[w["week_ending"] <= END]
    return w.set_index("week_ending").sort_index()


def expiry_dates(px):
    """Last trading day = 3rd trading day before the 1st of the delivery month.
    Trading calendar from the data (<= END); business days as fallback beyond END."""
    days = px.index
    out = {}
    for (y, m) in px.columns:
        first = pd.Timestamp(y, m, 1)
        prior = days[days < first]
        if first <= END and len(prior) >= 3:
            out[(y, m)] = prior[-3]
        else:
            out[(y, m)] = first - pd.offsets.BDay(3)
    return out


def prior_years_avg(series, date, years=5, min_years=2, tol_days=4):
    """Average of `series` at the same calendar week in up to `years` prior years
    (nearest date within tol_days). Needs at least min_years values, else NaN."""
    vals = []
    for k in range(1, years + 1):
        target = date - pd.DateOffset(years=k)
        i = series.index.get_indexer([target], method="nearest")[0]
        if i >= 0 and abs((series.index[i] - target).days) <= tol_days:
            vals.append(series.iloc[i])
    return np.mean(vals) if len(vals) >= min_years else np.nan


def build_panel(force=False):
    f = CACHE / "panel.pkl"
    if f.exists() and not force:
        return pd.read_pickle(f)
    px = load_curve()
    days = px.index
    exp = expiry_dates(px)
    pos_of = {d: i for i, d in enumerate(days)}

    def tdays_to_expiry(d, key):
        e = exp[key]
        if e <= days[-1]:
            return pos_of.get(e, days.searchsorted(e)) - pos_of[d]
        return len(pd.bdate_range(d, e)) - 1

    fridays = pd.date_range("2010-06-04", END, freq="W-FRI")
    rows = []
    for F in fridays:
        p = days.searchsorted(F)
        if p >= len(days):
            continue
        d = days[p]
        row_px = px.loc[d]
        # target winter
        Y = d.year if d.month <= 3 else d.year + 1
        while (Y, 3) not in exp or tdays_to_expiry(d, (Y, 3)) <= DROP_DAYS:
            Y += 1
        rec = {"friday": F, "date": d, "winter": Y}
        apr = row_px.get((Y, 4), np.nan)
        rec["april"] = apr
        live = []
        for name, key in zip(NAMES[:4], [(Y - 1, 12), (Y, 1), (Y, 2), (Y, 3)]):
            ok = key in exp and tdays_to_expiry(d, key) > DROP_DAYS
            v = row_px.get(key, np.nan) if ok else np.nan
            rec[name] = v - apr
            rec["tau_" + name] = ((pd.Timestamp(key[0], key[1], 1) - d).days / 365.25) if ok else np.nan
            rec["px_" + name] = v
            if ok and np.isfinite(v):
                live.append((v, rec["tau_" + name]))
        summer = [row_px.get((Y, m), np.nan) for m in range(4, 11)]
        if live and np.all(np.isfinite(summer)):
            rec["Strip"] = np.mean([v for v, _ in live]) - np.mean(summer)
            rec["tau_Strip"] = np.mean([t for _, t in live])
        else:
            rec["Strip"], rec["tau_Strip"] = np.nan, np.nan
        rec["n_live_winter"] = len(live)
        rows.append(rec)
    P = pd.DataFrame(rows).set_index("friday")
    P["roll"] = P["winter"].diff().fillna(0) > 0
    # Storage context known at date d: latest reported week ends F-7
    st = load_storage()
    chg = st.diff()
    hdd = load_hdd()["hdd"]
    P["rep_week"] = P.index - pd.Timedelta(days=7)
    P["storage_known"] = st.reindex(P["rep_week"]).values
    P["surplus_known"] = [P.at[F, "storage_known"] - prior_years_avg(st, w) if w in st.index else np.nan
                          for F, w in zip(P.index, P["rep_week"])]
    # surprise of the week reported this week (week ending F-7), published before d
    P["surprise_pub"] = [chg.get(w, np.nan) - prior_years_avg(chg, w) if w in chg.index else np.nan
                         for w in P["rep_week"]]
    # weather of the week ending F (observed by Friday F), anomaly vs prior 10 years
    P["hdd"] = hdd.reindex(P.index).values
    P["hdd_anom"] = [hdd.get(F, np.nan) - prior_years_avg(hdd, F, years=10, min_years=5) for F in P.index]
    # Ex-post target: end-March storage of the target winter minus prior-years avg (<=2021 only)
    em = st[(st.index.month == 3) & (st.index.day >= 25)]
    em = em.groupby(em.index.year).last()
    sur = {}
    for y in em.index:
        prev = [em.get(y - k) for k in range(1, 6) if (y - k) in em.index]
        if len(prev) >= 2:
            sur[y] = em[y] - np.mean(prev)
    P["endmar_surplus"] = P["winter"].map(sur)
    P["endmar_n_prior"] = P["winter"].map({y: sum((y - k) in em.index for k in range(1, 6)) for y in em.index})
    assert P["date"].max() <= END
    P.to_pickle(f)
    return P


if __name__ == "__main__":
    P = build_panel(force=True)
    print(P.shape, P.index.min(), P.index.max())
    print(P[NAMES].describe().round(3))
    print(P.groupby("winter")[NAMES].count())
    print(P[["surplus_known", "surprise_pub", "hdd_anom", "endmar_surplus"]].describe().round(1))
