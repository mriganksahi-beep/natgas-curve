"""Physical (P-measure) distribution of end-March storage, as of each decision date.

1. Model 2 (src/model2.py) - read from the shared dev cache of its walk-forward runs
   (data/processed/shared/model2_winter_percentiles_dev.csv, origins Nov 2016 - Dec 2021). p1..p99 are
   used as 99 equally weighted scenarios. Model 2 cannot be run earlier: Model 1's features need 5 prior
   years of storage (storage starts Jan 2010, so features start Feb 2015) and its walk-forward miss pool
   starts in 2016.
2. "P-lite" - a reduced weather-scenario simulator that can run from 2012, built only from data <= origin:
   weekly storage change in Nov-Mar weeks regressed (OLS, expanding window, refitted at every origin) on
   HDD, HDD^2, sin/cos(day of year), holiday dummy and a linear time trend; future weeks use the HDD of
   every past weather year (analog years, weather on record from Dec 2004, all strictly before the origin),
   plus 4-week blocks of the fit's own in-sample Nov-Mar residuals (inflated by sqrt(n/(n-p))).
   Scenarios = analog years x 200 residual paths. It is validated against Model 2 on the overlap.
Both are as-of the storage week F (public F+6); the decision price is at F+7 (data.decision_dates).
"""
import numpy as np
import pandas as pd

from src.model1 import holiday_in_week, load_raw
from src.research.v5_implied_storage import data as D

WINTER = {11, 12, 1, 2, 3}


def _season_end(F):
    Y = F.year + 1 if F.month >= 11 else F.year
    d = pd.Timestamp(Y, 3, 31)
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d  # last Friday of March (EIA week ending), as Model 2's target


class PLite:
    def __init__(self, n_resid=200, seed=0):
        s, w = load_raw()
        s = s[s.index <= D.CUT]["storage_bcf"]
        w = w[(w.index <= D.CUT) & (w.forecast_days == 0)][["hdd"]]
        self.s, self.w = s, w["hdd"]
        df = pd.DataFrame({"lvl": s}).join(self.w.rename("hdd"), how="inner")
        df["chg"] = df.lvl.diff()
        df = df.dropna()
        df = df[df.index.month.isin(WINTER)]
        self.df = self._design(df)
        self.n_resid = n_resid
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _design(df):
        doy = df.index.dayofyear
        out = df.copy()
        out["hdd2"] = out.hdd ** 2 / 100
        out["sin"], out["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        out["hol"] = [int(holiday_in_week(t)) for t in df.index]
        out["trend"] = (df.index - pd.Timestamp("2015-01-01")).days / 365.25
        out["c"] = 1.0
        return out

    COLS = ["c", "hdd", "hdd2", "sin", "cos", "hol", "trend"]

    def simulate(self, F):
        T = _season_end(F)
        tr = self.df[self.df.index <= F]
        X, y = tr[self.COLS].values, tr.chg.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        e = (e - e.mean()) * np.sqrt(len(y) / (len(y) - len(beta)))
        weeks = pd.date_range(F + pd.Timedelta(days=7), T, freq="7D")
        fut = self._design(pd.DataFrame({"hdd": np.zeros(len(weeks))}, index=weeks))
        # analog HDD paths from past years (same calendar week +-3 days), strictly before F
        paths = []
        for k in range(1, 30):
            row = []
            for d in weeks:
                a = d - pd.DateOffset(years=k)
                p = self.w.index.searchsorted(a)
                cand = [q for q in (p - 1, p) if 0 <= q < len(self.w) and abs(self.w.index[q] - a).days <= 3]
                if not cand or self.w.index[cand[0]] >= F:
                    row = None
                    break
                row.append(self.w.iloc[cand[0]])
            if row is not None:
                paths.append(row)
        A = np.array(paths)                       # (n_analog, n_weeks)
        Xf = fut[self.COLS].values
        base = Xf @ beta                          # without the hdd terms (hdd = 0)
        bh, bh2 = beta[1], beta[2]
        mean_chg = base[None, :] + bh * A + bh2 * A ** 2 / 100   # (n_analog, n_weeks)
        # residual paths: 4-week blocks
        n = len(weeks)
        R = np.empty((self.n_resid, n))
        for i in range(self.n_resid):
            f = 0
            while f < n:
                j = self.rng.integers(0, len(e) - 4)
                t = min(4, n - f)
                R[i, f:f + t] = e[j:j + t]
                f += t
        lv = self.s[F] + mean_chg.sum(1)[:, None] + R.sum(1)[None, :]
        return {"origin": F, "target": T, "levels": lv.ravel(), "n_analog": len(A)}


def m2_scenarios():
    """dict origin -> 99 equally weighted end-March levels, plus the 5-yr path benchmark."""
    m = D.load_m2()
    return {r.origin: (r[D.PCOLS].values.astype(float), r.benchmark) for _, r in m.iterrows()}
