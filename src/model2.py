"""Model 2: where will storage stand at the end of the season?

Targets: the storage level in the last EIA week of March (end of withdrawal season)
and of October (end of injection season).

Method (weather-scenario simulation):
  From an origin week t, step Model 1 forward one week at a time to the season end.
  * Weather for each future week comes from the same calendar week in each past year
    ("analog years"), giving one weather path per year on record.
  * Model 1 is not perfect, so each simulated week also gets one of Model 1's own past
    out-of-sample misses, drawn in 4-week blocks from the same half of the year
    (misses come in short streaks and are larger in winter).
  * Last week's simulated change feeds the next week's lag features, exactly as the
    real data would.
  Every combination of weather year x miss path is one scenario; the scenarios give a
  distribution for the end-of-season level. Median = point forecast; 10th-90th
  percentiles = 80% range.

Only information available at the origin is used: Model 1 is fit on weeks <= t, misses
come from walk-forward forecasts dated <= t, and analog weather comes from past years.

Benchmark: today's level plus the 5-year-average change for each remaining week.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.model1 import FEATURES_V2, build_features, holiday_in_week, load_raw, ridge, walk_forward

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1
WINTER = {11, 12, 1, 2, 3}
VAL_START, HOLDOUT_START = pd.Timestamp("2018-01-01"), pd.Timestamp("2022-01-01")
RECENTRE, DEMEAN_RESID = True, True
FEEDBACK = True  # False: lag features stay at their values at the origin  # v2 fixes; both False reproduces v1


def season_end(t: pd.Timestamp, weeks: pd.DatetimeIndex):
    """Target week for an origin: last EIA week of March (if t in Nov-Mar) or of October."""
    if t.month in WINTER:
        year = t.year + 1 if t.month >= 11 else t.year
        lo, hi = pd.Timestamp(year, 3, 25), pd.Timestamp(year, 3, 31)
    else:
        lo, hi = pd.Timestamp(t.year, 10, 25), pd.Timestamp(t.year, 10, 31)
    if t.month in (4,) and t.day < 1:
        return None
    cand = weeks[(weeks >= lo) & (weeks <= hi)]
    if len(cand):
        return cand[-1]
    # target week not in data yet (future season): compute its Friday date
    d = hi
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d


def nearest(index: pd.DatetimeIndex, date, tol=3):
    pos = index.searchsorted(date)
    best = None
    for p in (pos - 1, pos):
        if 0 <= p < len(index) and abs(index[p] - date) <= pd.Timedelta(days=tol):
            if best is None or abs(index[p] - date) < abs(index[best] - date):
                best = p
    return None if best is None else index[best]


class Simulator:
    def __init__(self, n_resid_paths=200, block=4, seed=0):
        self.s, self.w = load_raw()
        self.df = build_features(self.s, self.w)
        self.n_resid_paths, self.block = n_resid_paths, block
        self.rng = np.random.default_rng(seed)
        self._cache, self._ncache = {}, {}
        # Model 1 walk-forward misses from 2016 onward: the pool of "realistic errors"
        wf = walk_forward(self.df, ridge(ALPHA), pd.Timestamp("2016-01-01"), features=FEATURES_V2)
        self.resid = (self.df.loc[wf.index, "chg"] - wf).rename("resid")

    def _calendar_row(self, d):
        """Calendar inputs for a week not yet in the data (live use), built exactly as in
        Model 1: same calendar date in each of the previous 5 years, nearest week +-3 days."""
        chg = self.s["storage_bcf"].diff()
        out = {}
        for name, series in (("chg_5y", chg), ("hdd_5y", self.w["hdd"]), ("cdd_5y", self.w["cdd"])):
            vals = []
            for k in range(1, 6):
                a = nearest(series.index, d - pd.DateOffset(years=k))
                if a is not None and not np.isnan(series[a]):
                    vals.append(series[a])
            if len(vals) < 5:
                raise ValueError(f"not enough history for {name} at {d.date()}")
            out[name] = float(np.mean(vals))
        doy = d.dayofyear
        out["sin"], out["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        out["holiday"] = int(holiday_in_week(d))
        return pd.Series(out, name=d)

    def _future_rows(self, t, T):
        """Calendar-only inputs for each future week (all from past years -> no leakage)."""
        cols = ["chg_5y", "hdd_5y", "cdd_5y", "sin", "cos", "holiday"]
        weeks = pd.date_range(t + pd.Timedelta(days=7), T, freq="7D")
        rows = [self.df.loc[d, cols] if d in self.df.index else self._calendar_row(d) for d in weeks]
        fut = pd.DataFrame(rows, index=weeks)[cols]
        if fut.isna().any().any():
            raise ValueError("calendar inputs missing for some future week")
        return fut

    def _normal(self, d, years=10):
        """Mean hdd/cdd at the same calendar week over the `years` years before date d."""
        if (d, years) in self._ncache:
            return self._ncache[(d, years)]
        vals = []
        for k in range(1, years + 1):
            a = nearest(self.w.index, d - pd.DateOffset(years=k))
            if a is not None:
                vals.append(self.w.loc[a, ["hdd", "cdd"]].values.astype(float))
        out = np.mean(vals, axis=0) if len(vals) >= years // 2 else None
        self._ncache[(d, years)] = out
        return out

    def _analog_weather(self, t, weeks):
        """hdd/cdd paths: one per past year whose same-calendar weeks are all on record.

        v2: each analog year keeps its own departures from normal, but is re-centred on
        the latest 10-year normal: scenario = analog - normal(analog week) + normal(target
        week). Without this, older and colder winters bias every scenario toward bigger
        withdrawals (winters have warmed over the sample).
        """
        widx = self.w.index
        paths = {}
        for k in range(1, 40):
            rows = []
            for d in weeks:
                a = nearest(widx, d - pd.DateOffset(years=k))
                if a is None or a >= t:
                    rows = None
                    break
                x = self.w.loc[a, ["hdd", "cdd"]].values.astype(float)
                if RECENTRE:
                    na_, nd_ = self._normal(a), self._normal_target(d, t)
                    if na_ is None or nd_ is None:
                        rows = None
                        break
                    x = np.maximum(0.0, x - na_ + nd_)
                rows.append(x)
            if rows is None:
                if d - pd.DateOffset(years=k) < widx.min():
                    break
                continue
            paths[k] = np.array(rows, dtype=float)
        return paths  # {years back: array (n_weeks, 2)}

    def _normal_target(self, d, t):
        """Normal for a future week d, using only years whose same week is before origin t."""
        key = (d, t)
        if key not in self._cache:
            self._cache[key] = self._normal(d)
        return self._cache[key]

    def _resid_paths(self, t, weeks):
        r = self.resid[self.resid.index <= t]
        if DEMEAN_RESID:  # the pool represents spread, not bias (early misses came from
            r = r - r.mean()  # models trained on 1-2 years and run about -12 Bcf/week)
        vals, months = r.values, r.index.month
        n = len(weeks)
        out = np.empty((self.n_resid_paths, n))
        for p in range(self.n_resid_paths):
            filled = 0
            while filled < n:
                season = WINTER if weeks[filled].month in WINTER else set(range(4, 11))
                starts = np.flatnonzero(np.isin(months[: len(vals) - self.block + 1], list(season)))
                j = self.rng.choice(starts)
                take = min(self.block, n - filled)
                out[p, filled:filled + take] = vals[j:j + take]
                filled += take
        return out

    def simulate(self, t, with_resid=True):
        weeks_all = self.df.index
        T = season_end(t, weeks_all)
        if T is None or T <= t:
            return None
        train = self.df[self.df.index <= t]
        model = ridge(ALPHA)().fit(train[FEATURES_V2], train["chg"])

        fut = self._future_rows(t, T)
        weeks = fut.index
        analogs = self._analog_weather(t, weeks)
        A = np.stack(list(analogs.values()))  # (n_analog, n_weeks, 2)
        R = self._resid_paths(t, weeks) if with_resid else np.zeros((1, len(weeks)))
        na, nr, nw = A.shape[0], R.shape[0], len(weeks)
        hdd = np.repeat(A[:, :, 0], nr, axis=0)  # scenario = analog x resid path
        cdd = np.repeat(A[:, :, 1], nr, axis=0)
        res = np.tile(R, (na, 1))
        nS = hdd.shape[0]

        last = self.df.loc[t]
        hdd_prev = np.full(nS, last["hdd"])
        devs = list((train["chg"] - train["chg_5y"]).iloc[-4:].values)  # realized
        dev_hist = np.tile(np.array(devs, dtype=float), (nS, 1))  # (nS, 4)
        level = np.full(nS, float(self.s.loc[t, "storage_bcf"]))
        for i, d in enumerate(weeks):
            f = fut.loc[d]
            X = pd.DataFrame({
                "chg_5y": f.chg_5y, "hdd": hdd[:, i], "cdd": cdd[:, i],
                "hdd_anom": hdd[:, i] - f.hdd_5y, "cdd_anom": cdd[:, i] - f.cdd_5y,
                "hdd_sq": hdd[:, i] ** 2 / 100, "cdd_sq": cdd[:, i] ** 2 / 100,
                "sin": f.sin, "cos": f.cos,
                "hdd_x_cos": hdd[:, i] * f.cos, "cdd_x_cos": cdd[:, i] * f.cos,
                "prev_dev": dev_hist[:, -1], "prev_dev4": dev_hist.mean(axis=1),
                "hdd_prev": hdd_prev, "holiday": f.holiday,
            })[FEATURES_V2]
            chg = model.predict(X) + res[:, i]
            level = level + chg
            if FEEDBACK:  # simulated misses feed next week's lag features
                dev_hist = np.column_stack([dev_hist[:, 1:], chg - f.chg_5y])
            hdd_prev = hdd[:, i]

        bench = float(self.s.loc[t, "storage_bcf"] + fut["chg_5y"].sum())
        actual = float(self.s.loc[T, "storage_bcf"]) if T in self.s.index else np.nan
        q = np.percentile(level, [10, 50, 90])
        return {"origin": t, "target": T, "weeks_to_go": nw, "n_analogs": na,
                "n_scenarios": nS, "p10": q[0], "median": q[1], "p90": q[2],
                "benchmark": bench, "actual": actual, "season": "winter" if t.month in WINTER else "summer",
                "levels": level}


def live_end_of_season(now=None):
    """Forecast end-of-season storage from the latest reported week; append to the log."""
    from datetime import datetime, timezone
    log_path = ROOT / "reports/live_end_of_season.csv"
    sim = Simulator()
    t = sim.df.index.max()  # latest week with reported storage and weather
    r = sim.simulate(t)
    if r is None:
        print("No season target from", t.date())
        return None
    now = now or datetime.now(timezone.utc)
    row = {"forecast_made_utc": pd.Timestamp(now).strftime("%Y-%m-%d %H:%M"),
           "origin_week": t.date().isoformat(), "target_week": r["target"].date().isoformat(),
           "weeks_to_go": r["weeks_to_go"], "p10_bcf": round(r["p10"]), "median_bcf": round(r["median"]),
           "p90_bcf": round(r["p90"]), "five_year_path_bcf": round(r["benchmark"]),
           "prob_below_five_year_path": round(float((r["levels"] < r["benchmark"]).mean()), 2),
           "n_scenarios": r["n_scenarios"], "model": "Model 2 v2"}
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=list(row))
    if ((log["origin_week"] == row["origin_week"]) & (log["target_week"] == row["target_week"])).any():
        print("End-of-season forecast already logged for origin", row["origin_week"])
        return None
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)
    print(f"End-of-season ({row['target_week']}) from {row['origin_week']}: median {row['median_bcf']} Bcf, "
          f"80% range {row['p10_bcf']}-{row['p90_bcf']}; 5-yr path {row['five_year_path_bcf']}")
    return row


def backtest(start=VAL_START, with_resid=True, seed=0):
    sim = Simulator(seed=seed)
    rows = []
    for t in sim.df.index[sim.df.index >= start]:
        T = season_end(t, sim.df.index)
        if T is None or T <= t or T not in sim.s.index:
            continue
        # origins: from the first week of the season up to the week before the target
        season_start = (pd.Timestamp(t.year if t.month >= 11 else t.year - 1, 11, 1)
                        if t.month in WINTER else pd.Timestamp(t.year, 4, 1))
        if t < season_start:
            continue
        r = sim.simulate(t, with_resid=with_resid)
        if r:
            r.pop("levels")
            rows.append(r)
    out = pd.DataFrame(rows)
    out["miss_model"] = out["actual"] - out["median"]
    out["miss_bench"] = out["actual"] - out["benchmark"]
    out["inside80"] = (out["actual"] >= out["p10"]) & (out["actual"] <= out["p90"])
    out["width80"] = out["p90"] - out["p10"]
    return out


def summarize(bt):
    b = bt.copy()
    b["period"] = np.where(b["origin"] >= HOLDOUT_START, "holdout 2022+", "validation 2018-21")
    b["horizon"] = pd.cut(b["weeks_to_go"], [0, 4, 8, 12, 16, 30],
                          labels=["1-4 wks", "5-8 wks", "9-12 wks", "13-16 wks", "17+ wks"])
    g = b.groupby(["period", "horizon"], observed=True)
    return pd.DataFrame({
        "origins": g.size(),
        "model MAE (Bcf)": g["miss_model"].apply(lambda x: x.abs().mean()),
        "5-yr path MAE (Bcf)": g["miss_bench"].apply(lambda x: x.abs().mean()),
        "model better (%)": g.apply(lambda x: 100 * (x.miss_model.abs() < x.miss_bench.abs()).mean()),
        "inside 80% range (%)": g["inside80"].mean() * 100,
        "80% width (Bcf)": g["width80"].mean(),
        "bias (Bcf)": g["miss_model"].mean(),
    }).round(1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        live_end_of_season()
        sys.exit()
    # python -m src.model2          -> current backtest (v2b: 200 miss paths, origins from Nov 2016)
    # python -m src.model2 live     -> log a live end-of-season forecast
    tag, start = "v2b_from2016", pd.Timestamp("2016-11-01")
    bt = backtest(start=start)
    bt.to_csv(ROOT / f"data/processed/model2_backtest_{tag}.csv", index=False)
    print(summarize(bt[bt.origin >= VAL_START]).to_string())
