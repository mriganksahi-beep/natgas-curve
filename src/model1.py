"""Model 1: forecast next week's storage change from that week's weather.

Timing: the EIA week ends on Friday; the number is released the following Thursday.
By Thursday morning we know that week's weather, so a forecast for week t may use
weather for week t, but storage only up to week t-1.

Walk-forward: for each test week we fit only on weeks strictly before it.
Model choices (features, penalty, tree settings) are picked on 2018-2021 and the
headline is scored on 2022 onward, which those choices never saw.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALIGN = "calendar"  # "weeks52" reproduces v1/v2
VAL_START, HOLDOUT_START = pd.Timestamp("2018-01-01"), pd.Timestamp("2022-01-01")


def same_week_prior_years(s: pd.Series, years=5) -> pd.Series:
    """Average of s at the same calendar week in each of the previous `years` years.
    Matches the nearest date within +-3 days of exactly k*52 weeks earlier."""
    out = pd.Series(np.nan, index=s.index)
    idx = s.index
    for t in idx:
        vals = []
        for k in range(1, years + 1):
            # same calendar date k years earlier (v3). v1/v2 used t - 52k weeks, which
            # drifts about a day per year, so the "same week" 5 years back was up to 6 days off.
            target = t - pd.DateOffset(years=k) if ALIGN == "calendar" else t - pd.Timedelta(weeks=52 * k)
            pos = idx.searchsorted(target)
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(idx)]
            best = min(cands, key=lambda p: abs(idx[p] - target)) if cands else None
            if best is not None and abs(idx[best] - target) <= pd.Timedelta(days=3):
                vals.append(s.iloc[best])
        if len(vals) == years:
            out[t] = np.mean(vals)
    return out


def load_raw():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"]).set_index("week_ending")
    own = ROOT / "data/raw/weekly_degree_days_own.csv"  # our own pull (src/fetch.py)
    live = ROOT / "data/raw/weekly_degree_days_live.csv"  # weeks added by the live job
    if own.exists():
        w = pd.read_csv(own, parse_dates=["week_ending"]).set_index("week_ending")
        if live.exists():
            lv = pd.read_csv(live, parse_dates=["week_ending"]).set_index("week_ending")[["hdd", "cdd"]]
            w = pd.concat([w, lv[~lv.index.isin(w.index)]]).sort_index()
        w["forecast_days"] = 0  # observed weather (archive, or recent-days analysis)
    else:  # development copy from the HughFOD repo (not redistributed)
        w = pd.read_csv(ROOT / "data/raw/weekly_degree_days.csv", parse_dates=["week_ending"]).set_index("week_ending")
    return s, w


def build_features(s=None, w=None, keep_last_unknown=False) -> pd.DataFrame:
    """keep_last_unknown: keep a final week whose storage is not yet reported (live use)."""
    if s is None:
        s, w = load_raw()
    w = w[w["forecast_days"] == 0][["hdd", "cdd"]]  # history only: no forecast weather

    df = s.join(w, how="inner")
    df["chg"] = df["storage_bcf"].diff()  # target: this week's storage change
    df["chg_5y"] = same_week_prior_years(df["chg"])  # the market's naive benchmark
    df["hdd_5y"] = same_week_prior_years(df["hdd"])
    df["cdd_5y"] = same_week_prior_years(df["cdd"])
    # weather anomalies vs the same week of the last 5 years
    df["hdd_anom"] = df["hdd"] - df["hdd_5y"]
    df["cdd_anom"] = df["cdd"] - df["cdd_5y"]
    # extreme-cold / extreme-heat curvature: demand bends up in severe weather
    df["hdd_sq"] = df["hdd"] ** 2 / 100
    df["cdd_sq"] = df["cdd"] ** 2 / 100
    # season (winter heating response differs from summer power-burn response)
    doy = df.index.dayofyear
    df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["hdd_x_cos"] = df["hdd"] * df["cos"]
    df["cdd_x_cos"] = df["cdd"] * df["cos"]
    # last week's surprise vs its own 5-year norm (released before this week's forecast):
    # captures slow-moving supply drift (production growth, LNG exports)
    df["prev_dev"] = (df["chg"] - df["chg_5y"]).shift(1)
    df["prev_dev4"] = (df["chg"] - df["chg_5y"]).shift(1).rolling(4).mean()
    df["hdd_prev"] = df["hdd"].shift(1)
    # holiday weeks: industrial gas demand drops (Thanksgiving, Christmas, New Year)
    df["holiday"] = [int(holiday_in_week(t)) for t in df.index]
    if keep_last_unknown:
        feats = [c for c in df.columns if c not in ("storage_bcf", "chg")]
        return df.dropna(subset=feats)
    return df.dropna()


def holiday_in_week(week_end: pd.Timestamp) -> bool:
    days = pd.date_range(week_end - pd.Timedelta(days=6), week_end)
    for d in days:
        if (d.month, d.day) in ((12, 25), (1, 1)):
            return True
        if d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:  # Thanksgiving
            return True
    return False


FEATURES = ["chg_5y", "hdd", "cdd", "hdd_anom", "cdd_anom", "hdd_sq", "cdd_sq",
            "sin", "cos", "hdd_x_cos", "cdd_x_cos", "prev_dev", "prev_dev4", "hdd_prev"]
FEATURES_V2 = FEATURES + ["holiday"]


def walk_forward(df, make_model, start, refit_every=1, features=None):
    features = features or FEATURES
    """Expanding-window forecasts for every week >= start, fitting only on earlier weeks."""
    preds, model, last_fit = {}, None, None
    test_weeks = df.index[df.index >= start]
    for i, t in enumerate(test_weeks):
        train = df[df.index < t]
        assert train.index.max() < t  # no future rows
        if model is None or i % refit_every == 0:
            model = make_model().fit(train[features], train["chg"])
        preds[t] = float(model.predict(df.loc[[t], features])[0])
    return pd.Series(preds, name="pred")


def ridge(alpha):
    return lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def gbm(lr, leaves):
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves,
                                                 max_iter=300, min_samples_leaf=20,
                                                 l2_regularization=1.0, random_state=0)


def interval_coverage(df, pred, start, q=(0.1, 0.9)):
    """80% range from quantiles of *past* walk-forward errors only."""
    err = (df["chg"] - pred).dropna()
    hits = []
    for t in pred.index[pred.index >= start]:
        past = err[err.index < t]
        if len(past) < 52:
            continue
        lo, hi = past.quantile(q[0]), past.quantile(q[1])
        hits.append(lo <= err[t] <= hi)
    return 100 * np.mean(hits)


def score(df, pred, mask):
    wk = pred.index[mask]
    y, p, b = df.loc[wk, "chg"], pred.loc[wk], df.loc[wk, "chg_5y"]
    e = y - p
    return {
        "MAE (Bcf)": e.abs().mean(),
        "RMSE (Bcf)": np.sqrt((e ** 2).mean()),
        "bias (Bcf)": e.mean(),
        "beats 5-yr avg (%)": 100 * ((e.abs() < (y - b).abs()).mean()),
        "surprise sign right (%)": 100 * (np.sign(p - b) == np.sign(y - b)).mean(),
        "weeks": len(y),
    }


def main(features=None, tag="v1"):
    global FEATURES
    if features is not None:
        FEATURES = features
    df = build_features()
    print(f"Weeks with all features: {len(df)}  ({df.index.min().date()} to {df.index.max().date()})")

    # --- 1. choose settings on 2018-2021 only ---------------------------------------
    val_mask = lambda p: (p.index >= VAL_START) & (p.index < HOLDOUT_START)
    choices = {}
    for a in [0.001, 0.01, 0.03, 0.1, 1, 3, 10, 30, 100]:
        p = walk_forward(df, ridge(a), VAL_START)
        choices[("ridge", a)] = score(df, p, val_mask(p))["MAE (Bcf)"]
    for lr in [0.03, 0.1]:
        for lv in [7, 15]:
            p = walk_forward(df, gbm(lr, lv), VAL_START, refit_every=4)
            choices[("gbm", (lr, lv))] = score(df, p, val_mask(p))["MAE (Bcf)"]
    val = pd.Series(choices).rename("validation MAE 2018-21")
    print("\nValidation (2018-2021) MAE by setting:\n", val.round(2).to_string())
    best_a = val.loc["ridge"].idxmin()
    best_g = val.loc["gbm"].idxmin()

    # --- 2. run chosen settings walk-forward from 2018, score holdout 2022+ -----------
    preds = {
        "5-year average": df["chg_5y"][df.index >= VAL_START],
        "Last week's change": df["chg"].shift(1)[df.index >= VAL_START],
        f"Ridge (alpha={best_a})": walk_forward(df, ridge(best_a), VAL_START),
        f"Gradient boosting {best_g}": walk_forward(df, gbm(*best_g), VAL_START, refit_every=4),
    }
    rows = {}
    for name, p in preds.items():
        hold = p.index >= HOLDOUT_START
        r = score(df, p, hold)
        r["80% range covered (%)"] = interval_coverage(df, p, HOLDOUT_START)
        rows[name] = r
        # season split
        win = hold & p.index.month.isin([11, 12, 1, 2, 3])
        r["winter MAE"] = score(df, p, win)["MAE (Bcf)"]
        r["summer MAE"] = score(df, p, hold & ~p.index.month.isin([11, 12, 1, 2, 3]))["MAE (Bcf)"]
    table = pd.DataFrame(rows).T.round(1)
    print("\nHoldout 2022 onward (choices never saw these weeks):\n", table.to_string())

    out = pd.DataFrame(preds)
    out["actual"] = df["chg"]
    out.to_csv(ROOT / f"data/processed/model1_predictions_{tag}.csv")
    table.to_csv(ROOT / f"reports/model1_holdout_{tag}.csv")
    val.to_csv(ROOT / f"reports/model1_validation_{tag}.csv")
    return df, out, table


if __name__ == "__main__":
    # python -m src.model1          -> current model (v3: holiday flag, calendar-aligned 5-yr average)
    # python -m src.model1 v1       -> original feature set, for the record
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "v1":
        main(FEATURES, "v1")
    else:
        main(FEATURES_V2, "v3_own")
