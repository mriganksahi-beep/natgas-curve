"""Frozen news / premium decomposition of daily gas-curve moves (study v4, agent news_channel).

    decompose(panel, instrument, mode="walkforward") -> DataFrame (one row per eligible panel row)

Definition (frozen 23 Nov 2026 after the choice window 2010-11..2016-17; see
reports/agents/v4_news_channel.md):

    move_d       = instrument's same-contract change from the previous trading day dp to d
                   (r_front, r_second, r_strip: 100*dlog, %;  d_hj: $/MMBtu)
    x_{k,d}      = SPEC[instrument]["terms"]:
                   r_front, r_second, r_strip: news_w1, news_w2, news_w3 (three betas)
                   d_hj: news_tot = news_w0+w1+w2+w3 (one beta)
                   (the code also supports x = news * (max(hj_dp,0)+c) state scaling; it was tried and
                   rejected walk-forward, so the frozen SPEC uses scale=None everywhere)
    move_d       = a + sum_k b_k x_{k,d} + e_d           OLS, fitted on Nov-Feb rows only
    news_part_d  = sum_k b_k x_{k,d}                      (NO intercept: drift is not news)
    premium_part = move_d - news_part_d                   (includes the intercept/drift)

Walk-forward ("walkforward", the default and the only mode for development work): the coefficients
used for a row of winter W (W = calendar year of the winter's January; Jul-Dec belong to W+1) are
fitted on the Nov-Feb rows of winters < W only, refitted once per winter, and need at least
MIN_TRAIN_WINTERS prior winters (otherwise news_part is NaN, vintage "none"). Coefficients are applied
to every eligible row of winter W, i.e. Oct-Apr wherever the news columns exist (in_core flags Nov-Feb,
the fitting months; Oct/Mar/Apr rows are extrapolation).

Frozen ("frozen"): applies the coefficients in data/processed/research_v4_news_channel/frozen_coefs.json
(fitted once on all dev Nov-Feb rows, dates <= 2021-12-31) unchanged. This is the mode proposed for
the locked 2000-2010 panel: the coefficients never saw any pre-2010 row.

Eligibility: news_w1..w3 (and any term used) non-missing, move non-missing; for d_hj additionally the
spread must belong to the current winter (hj_year == W if hj_year exists, else hj_days_left <= 125
trading days if that exists, else Oct-Feb rows).

Columns required: date, the move column, the SPEC terms (news_w1..w3; news_tot for d_hj), and for
d_hj also hj (and hj_year or hj_days_left if present, to keep only the current winter's spread).
The locked panel has all of these for r_front, r_second, d_hj. No options, CPC, strip or storage.

Weekly companion: decompose_weekly(panel, instrument) does the same on Friday-to-Friday sums of daily
moves and daily news (weeks with >= 4 eligible days), with WEEKLY_SPEC: the three news windows for the
percent instruments, and for d_hj news_tot * (max(hj at the previous Friday settle, 0) + 0.10). Daily
news is priced partly the day before (06Z/12Z runs we do not have) and partly the day after, so the
weekly fit captures 2.5-4x the daily sensitivity; use it for horizons >= 1 week.

Returned columns: date, instrument, winter, in_core, move, news_part, premium_part, vintage,
vintage_last_winter, n_fit, intercept (fitted drift of that vintage; NOT part of news_part),
train_mean (mean move of the vintage's training rows; benchmark for out-of-sample R^2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/processed/research_v4_news_channel"
PANEL_DEV = ROOT / "data/processed/v4/panel_dev.csv"
CUTOFF = pd.Timestamp("2021-12-31")
FROZEN = CACHE / "frozen_coefs.json"

FIT_MONTHS = (11, 12, 1, 2)
MIN_TRAIN_WINTERS = 2
NEWS3 = ["news_w1", "news_w2", "news_w3"]

# ---------------------------------------------------------------- the frozen specification
SPEC = {
    "r_front":  {"terms": NEWS3, "scale": None},
    "r_second": {"terms": NEWS3, "scale": None},
    "r_strip":  {"terms": NEWS3, "scale": None},
    "d_hj":     {"terms": ["news_tot"], "scale": None},   # one beta; hj-scaled variants failed walk-forward
}
INSTRUMENTS = list(SPEC)

# Weekly (Friday-to-Friday) companion: sums of daily moves and daily news over the week's eligible days.
# d_hj uses the v2 tightness scaling, g = max(spread at the previous Friday settle, 0) + 0.10.
WEEKLY_SPEC = {
    "r_front":  {"terms": NEWS3, "scale": None},
    "r_second": {"terms": NEWS3, "scale": None},
    "r_strip":  {"terms": NEWS3, "scale": None},
    "d_hj":     {"terms": ["news_tot"], "scale": {"c": 0.10}},
}
MIN_DAYS_WEEK = 4


# ---------------------------------------------------------------- helpers
def load_dev_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL_DEV, parse_dates=["date"])
    return p[p.date <= CUTOFF].reset_index(drop=True)          # dev cut, immediately


def winter_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d.dt.year + (d.dt.month >= 7)).astype(int)


def _with_tot(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel.date)
    if "news_tot" not in panel:                                  # locked panel may lack it
        panel["news_tot"] = panel[["news_w0", "news_w1", "news_w2", "news_w3"]].sum(axis=1, min_count=4)
    return panel


def eligible(panel: pd.DataFrame, instrument: str, spec: dict) -> pd.Series:
    W = winter_of(panel.date)
    ok = panel[instrument].notna()
    for t in spec["terms"]:
        ok &= panel[t].notna()
    if instrument == "d_hj":
        if "hj_year" in panel:
            ok &= panel.hj_year == W
        elif "hj_days_left" in panel:
            ok &= panel.hj_days_left <= 125
        else:
            ok &= panel.date.dt.month.isin([10, 11, 12, 1, 2])
        ok &= panel.hj.notna()
    return ok


def design(panel: pd.DataFrame, instrument: str, spec: dict) -> pd.DataFrame:
    X = panel[spec["terms"]].astype(float).copy()
    if spec.get("scale"):
        sc = spec["scale"]
        state = sc.get("state", "hj_prev")
        if state == "hj_prev":
            s = panel.hj - panel.d_hj                            # spread level at dp (pre-move)
        else:
            s = panel[state]
        g = s.clip(lower=0.0) + sc["c"]
        if sc.get("norm"):
            g = g / sc["norm"]
        X = X.mul(g, axis=0)
    return X


def ols(X: np.ndarray, y: np.ndarray, weights=None):
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b                                                     # [intercept, slopes...]


# ---------------------------------------------------------------- main entry point
def decompose(panel: pd.DataFrame, instrument: str, mode: str = "walkforward",
              spec: dict | None = None, min_train_winters: int = MIN_TRAIN_WINTERS,
              fit_months=FIT_MONTHS, coefs: dict | None = None) -> pd.DataFrame:
    """Split each eligible row's move into news_part and premium_part (see module docstring)."""
    spec = spec or SPEC[instrument]
    panel = _with_tot(panel)
    ok = eligible(panel, instrument, spec)
    ok &= design(panel, instrument, spec).notna().all(axis=1)    # scaling state must exist too
    p = panel[ok].reset_index(drop=True)
    X = design(p, instrument, spec).to_numpy()
    y = p[instrument].to_numpy(float)
    W = winter_of(p.date).to_numpy()
    fitrow = p.date.dt.month.isin(fit_months).to_numpy()
    news = np.full(len(p), np.nan)
    vint = np.array(["none"] * len(p), dtype=object)
    vlast = np.full(len(p), -1)
    nfit = np.zeros(len(p), int)
    icpt = np.full(len(p), np.nan)
    ybar = np.full(len(p), np.nan)
    if mode == "walkforward":
        for w in np.unique(W):
            tr = fitrow & (W < w)
            if len(np.unique(W[tr])) < min_train_winters:
                continue
            b = ols(X[tr], y[tr])
            m = W == w
            news[m] = X[m] @ b[1:]
            icpt[m] = b[0]
            ybar[m] = y[tr].mean()
            vint[m] = f"wf_through_W{W[tr].max()}"
            vlast[m] = W[tr].max()
            nfit[m] = tr.sum()
    elif mode == "frozen":
        c = coefs or json.loads(FROZEN.read_text())[instrument]
        assert c["terms"] == spec["terms"] and c["scale"] == spec["scale"], "spec differs from frozen file"
        news = X @ np.asarray(c["slopes"], float)
        icpt[:] = c["intercept"]
        vint[:] = c["vintage"]
        vlast[:] = c["last_winter"]
        nfit[:] = c["n_fit"]
    else:
        raise ValueError(mode)
    out = pd.DataFrame({"date": p.date, "instrument": instrument, "winter": W,
                        "in_core": p.date.dt.month.isin(FIT_MONTHS).to_numpy(), "move": y,
                        "news_part": news, "premium_part": y - news, "vintage": vint,
                        "vintage_last_winter": vlast, "n_fit": nfit,
                        "intercept": icpt, "train_mean": ybar})
    if mode == "walkforward":
        assert_no_lookahead(out)
    return out


def weekly_table(panel: pd.DataFrame, instrument: str, spec: dict) -> pd.DataFrame:
    """Friday-to-Friday sums over eligible days; state hj_dp taken at the week's first eligible day."""
    panel = _with_tot(panel)
    base = {"terms": spec["terms"], "scale": None}
    p = panel[eligible(panel, instrument, base)].copy()
    p["wk"] = p.date.dt.to_period("W-FRI")
    p["hj_dp"] = p.hj - p.d_hj if "hj" in p and "d_hj" in p else np.nan
    agg = {instrument: "sum", "date": "last", "hj_dp": "first", **{t: "sum" for t in spec["terms"]}}
    g = p.groupby("wk").agg(agg)
    g["ndays"] = p.groupby("wk").size()
    g = g[g.ndays >= MIN_DAYS_WEEK].reset_index(drop=True)
    if spec.get("scale"):
        gs = g.hj_dp.clip(lower=0.0) + spec["scale"]["c"]
        for t in spec["terms"]:
            g[t] = g[t] * gs
    return g


def decompose_weekly(panel: pd.DataFrame, instrument: str, mode: str = "walkforward",
                     spec: dict | None = None, min_train_winters: int = MIN_TRAIN_WINTERS,
                     fit_months=FIT_MONTHS, coefs: dict | None = None) -> pd.DataFrame:
    """Weekly companion of decompose(): one row per Friday-ending week (>= 4 eligible days).
    Same walk-forward-by-winter rule (winter and fit months taken from the week's last day)."""
    spec = spec or WEEKLY_SPEC[instrument]
    g = weekly_table(panel, instrument, spec)
    X = g[spec["terms"]].to_numpy(float)
    y = g[instrument].to_numpy(float)
    W = winter_of(g.date).to_numpy()
    fitrow = g.date.dt.month.isin(fit_months).to_numpy()
    news = np.full(len(g), np.nan); icpt = news.copy(); ybar = news.copy()
    vint = np.array(["none"] * len(g), dtype=object); vlast = np.full(len(g), -1)
    if mode == "walkforward":
        for w in np.unique(W):
            tr = fitrow & (W < w)
            if len(np.unique(W[tr])) < min_train_winters:
                continue
            b = ols(X[tr], y[tr]); m = W == w
            news[m] = X[m] @ b[1:]; icpt[m] = b[0]; ybar[m] = y[tr].mean()
            vint[m] = f"wf_weekly_through_W{W[tr].max()}"; vlast[m] = W[tr].max()
    elif mode == "frozen":
        c = coefs or json.loads(FROZEN.read_text())["weekly"][instrument]
        assert c["terms"] == spec["terms"] and c["scale"] == spec["scale"], "spec differs from frozen file"
        news = X @ np.asarray(c["slopes"], float); icpt[:] = c["intercept"]
        vint[:] = c["vintage"]; vlast[:] = c["last_winter"]
    else:
        raise ValueError(mode)
    out = pd.DataFrame({"week_end": g.date, "instrument": instrument, "winter": W,
                        "in_core": g.date.dt.month.isin(FIT_MONTHS).to_numpy(), "ndays": g.ndays,
                        "move": y, "news_part": news, "premium_part": y - news, "vintage": vint,
                        "vintage_last_winter": vlast, "intercept": icpt, "train_mean": ybar})
    if mode == "walkforward":
        assert_no_lookahead(out)
    return out


def fit_frozen_weekly(panel: pd.DataFrame, instrument: str) -> dict:
    spec = WEEKLY_SPEC[instrument]
    g = weekly_table(panel[pd.to_datetime(panel.date) <= CUTOFF], instrument, spec)
    g = g[g.date.dt.month.isin(FIT_MONTHS)]
    b = ols(g[spec["terms"]].to_numpy(float), g[instrument].to_numpy(float))
    W = winter_of(g.date)
    return {"terms": spec["terms"], "scale": spec["scale"], "intercept": float(b[0]),
            "slopes": [float(v) for v in b[1:]], "n_fit": int(len(g)), "first_winter": int(W.min()),
            "last_winter": int(W.max()), "vintage": f"frozen_weekly_dev_W{int(W.min())}-W{int(W.max())}"}


def assert_no_lookahead(out: pd.DataFrame) -> None:
    """Every row with a news_part must use coefficients fitted only on winters strictly before its own."""
    has = out.news_part.notna()
    assert (out.loc[has, "vintage_last_winter"] < out.loc[has, "winter"]).all(), "look-ahead in vintage"
    assert (out.loc[~has, "vintage"] == "none").all()


def fit_frozen(panel: pd.DataFrame, instrument: str, spec: dict | None = None) -> dict:
    spec = spec or SPEC[instrument]
    panel = _with_tot(panel)
    ok = eligible(panel, instrument, spec) & panel.date.dt.month.isin(FIT_MONTHS) & (panel.date <= CUTOFF)
    ok &= design(panel, instrument, spec).notna().all(axis=1)
    p = panel[ok]
    b = ols(design(p, instrument, spec).to_numpy(), p[instrument].to_numpy(float))
    W = winter_of(p.date)
    return {"terms": spec["terms"], "scale": spec["scale"], "intercept": float(b[0]),
            "slopes": [float(v) for v in b[1:]], "n_fit": int(len(p)),
            "first_winter": int(W.min()), "last_winter": int(W.max()), "fit_months": list(FIT_MONTHS),
            "vintage": f"frozen_dev_W{int(W.min())}-W{int(W.max())}_to_2021-12-31"}


def _self_test(panel: pd.DataFrame) -> None:
    """Sanity: walk-forward vintages precede each row's winter; perturbing a later winter's moves
    must leave earlier winters' news_part unchanged."""
    for ins in INSTRUMENTS:
        a = decompose(panel, ins)
        assert_no_lookahead(a)
        q = panel.copy()
        late = winter_of(q.date) >= 2019
        q.loc[late, ins] = q.loc[late, ins] * -5 + 3            # vandalise the future
        b = decompose(q, ins)
        early = a.winter < 2019
        assert np.allclose(a.news_part[early].fillna(0), b.news_part[early].fillna(0)), ins
        aw, bw = decompose_weekly(panel, ins), decompose_weekly(q, ins)
        e = aw.winter < 2019
        assert np.allclose(aw.news_part[e].fillna(0), bw.news_part[e].fillna(0)), ins + " weekly"
    print("self-test passed: no row uses coefficients from its own or a later winter")


def main():
    panel = load_dev_panel()
    CACHE.mkdir(parents=True, exist_ok=True)
    frozen = {ins: fit_frozen(panel, ins) for ins in INSTRUMENTS}
    frozen["weekly"] = {ins: fit_frozen_weekly(panel, ins) for ins in INSTRUMENTS}
    frozen["_meta"] = {"created": "2026-11-23", "data_cut": "2021-12-31",
                       "definition": "news_part = sum_k slope_k * news_wk * g; g = 1, or max(hj - d_hj, 0) + c "
                                     "for d_hj; premium_part = move - news_part (intercept excluded)"}
    FROZEN.write_text(json.dumps(frozen, indent=2))
    parts = [decompose(panel, ins) for ins in INSTRUMENTS]
    dec = pd.concat(parts, ignore_index=True)
    dec = dec[dec.date <= CUTOFF]
    dec.to_csv(CACHE / "decomposed_dev.csv", index=False, float_format="%.6g")
    wk = pd.concat([decompose_weekly(panel, ins) for ins in INSTRUMENTS], ignore_index=True)
    wk[wk.week_end <= CUTOFF].to_csv(CACHE / "decomposed_weekly_dev.csv", index=False, float_format="%.6g")
    _self_test(panel)
    for k, v in frozen.items():
        if k == "weekly":
            for kk, vv in v.items():
                print("weekly", kk, vv["terms"], vv["scale"], np.round(vv["slopes"], 5), vv["n_fit"])
        elif k != "_meta":
            print("daily", k, v["terms"], np.round(v["slopes"], 5), v["n_fit"])
    print(dec.groupby("instrument").agg(n=("move", "size"), n_news=("news_part", "count")))


if __name__ == "__main__":
    main()
