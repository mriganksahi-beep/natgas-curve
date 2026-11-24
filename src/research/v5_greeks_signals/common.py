"""v5_greeks_signals: shared loaders and scoring helpers.

Development data only (date <= 2021-12-31). Options come only from data/processed/v5/ (already cut)
and the raw near-money files filtered to <= 2021-12-31 on load. Never reads *_locked folders.

Scoring conventions (used by every candidate signal):
  * A signal s_t is known at the settle of day t. It is standardised walk-forward: expanding mean/sd
    refitted on the first trading day of each month using data strictly before that month.
  * The direction (+1 / -1) of each rule is CHOSEN on 2010-2016 (sign of the choose-window predictive
    slope) and frozen for the 2017-2021 CHECK window.
  * Futures rule: position_t = direction * sign(z_t) contracts, held h days as an overlapping portfolio
    (position = mean of the last h daily positions). Daily P&L = pos_{t-1} * dF_t * 10,000 ($/contract).
    Costs: $0.002/MMBtu per side (2 ticks) per unit of position change, i.e. $20/contract.
  * Delta-hedged straddle rule: same, with daily P&L = pos_{t-1} * dh_pnl_t * 10,000; costs 1.5% of
    straddle premium per unit of position change plus the futures cost of the daily re-hedge
    (0.002 * |gamma0 * dF| per straddle).
  * Statistics: Newey-West HAC t-stats with lags >= horizon overlap (min 5), and a 21-day moving
    block bootstrap CI for the mean daily P&L.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
V5 = ROOT / "data/processed/v5"
CACHE = ROOT / "data/processed/research_v5_greeks_signals"
FIG = ROOT / "reports/agents/figures"
CACHE.mkdir(parents=True, exist_ok=True)
CHOOSE = (pd.Timestamp("2010-01-01"), pd.Timestamp("2016-12-31"))
CHECK = (pd.Timestamp("2017-01-01"), pd.Timestamp("2021-12-31"))
FUT_COST = 0.002          # $/MMBtu per side
OPT_HALF = 0.015          # fraction of premium per side
UNIT = 10_000             # MMBtu per contract


# ------------------------------------------------------------------ loaders
def panel() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date", "prev_date"])
    p = p[p.date <= CUT].set_index("date").sort_index()
    # $ changes of the rolled front / second (same contract day over day)
    for k in ("front", "second"):
        prev = p[f"px_{k}"] / np.exp(p[f"r_{k}"] / 100)
        p[f"dF_{k}"] = p[f"px_{k}"] - prev
    p["roll_front"] = (p.front != p.front.shift()).astype(int)
    p["roll_second"] = (p.second != p.second.shift()).astype(int)
    return p


def curve() -> pd.DataFrame:
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle", "open_interest"])
    c = c[(c.date <= CUT) & (c.settle > 0)]
    c["cid"] = c.contract_year * 100 + c.contract_month
    return c.pivot_table(index="date", columns="cid", values="settle").sort_index()


def surface() -> pd.DataFrame:
    s = pd.read_csv(V5 / "surface.csv", parse_dates=["date"])
    s = s[s.date <= CUT].sort_values(["date", "T"])
    s["rank"] = s.groupby("date").cumcount() + 1
    return s


def dh() -> pd.DataFrame:
    d = pd.read_csv(V5 / "dh_straddle.csv", parse_dates=["date", "date1"])
    d = d[d.date1 <= CUT].sort_values(["date", "T"])
    d["rank"] = d.groupby("date").cumcount() + 1
    return d


def weekly() -> pd.DataFrame:
    w = pd.read_csv(V5 / "weekly.csv", parse_dates=["date", "expiry"])
    return w[w.date <= CUT]


# ------------------------------------------------------------------ walk-forward helpers
def month_starts(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.to_period("M"), index=idx)


def wf_z(s: pd.Series, min_obs: int = 120) -> pd.Series:
    """Expanding z-score, parameters refitted monthly using data before the month."""
    s = s.astype(float)
    per = s.index.to_period("M")
    out = pd.Series(np.nan, index=s.index)
    for m in per.unique():
        hist = s[per < m].dropna()
        if len(hist) < min_obs:
            continue
        mu, sd = hist.mean(), hist.std()
        if sd > 0:
            sel = per == m
            out[sel] = (s[sel] - mu) / sd
    return out


def wf_beta_resid(y: pd.Series, X: pd.DataFrame, min_obs: int = 120) -> pd.Series:
    """Residual of y on X (with intercept) using expanding OLS refitted monthly (past data only)."""
    df = pd.concat([y.rename("y"), X], axis=1)
    per = df.index.to_period("M")
    out = pd.Series(np.nan, index=df.index)
    for m in per.unique():
        hist = df[per < m].dropna()
        if len(hist) < min_obs:
            continue
        A = np.column_stack([np.ones(len(hist)), hist[X.columns].values])
        b = np.linalg.lstsq(A, hist.y.values, rcond=None)[0]
        cur = df[per == m]
        pred = b[0] + cur[X.columns].values @ b[1:]
        out[cur.index] = cur.y.values - pred
    return out


def wf_quantile(s: pd.Series, min_obs: int = 12) -> pd.Series:
    """Percentile of each value among strictly earlier observations (event-level series)."""
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        h = v[:i][np.isfinite(v[:i])]
        if len(h) >= min_obs and np.isfinite(v[i]):
            out[i] = (h < v[i]).mean()
    return pd.Series(out, index=s.index)


# ------------------------------------------------------------------ statistics
def hac_ols(y, X, lags: int):
    """OLS with Newey-West SEs. X without constant (added). Returns (beta, t) arrays incl. const."""
    import statsmodels.api as sm
    df = pd.concat([pd.Series(y, name="y"), pd.DataFrame(X)], axis=1).dropna()
    if len(df) < 30:
        return np.full(df.shape[1], np.nan), np.full(df.shape[1], np.nan), len(df)
    m = sm.OLS(df.y, sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(int(lags), 1)})
    return m.params.values, m.tvalues.values, len(df)


def hac_mean_t(x: pd.Series, lags: int = 5):
    x = pd.Series(x).dropna()
    if len(x) < 30:
        return np.nan
    b, t, _ = hac_ols(x.values, np.empty((len(x), 0)), lags)
    return t[0]


def block_boot_ci(x, block: int = 21, n: int = 2000, seed: int = 0):
    x = np.asarray(pd.Series(x).dropna())
    if len(x) < 2 * block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :len(x)]
    m = x[idx].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def fwd_sum(s: pd.Series, h: int) -> pd.Series:
    """Sum of s over t+1..t+h (aligned at t)."""
    return s[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


# ------------------------------------------------------------------ tradable scoring
def overlap_pos(raw_pos: pd.Series, h: int) -> pd.Series:
    return raw_pos.fillna(0).rolling(h, min_periods=1).mean()


def futures_pnl(pos: pd.Series, dF: pd.Series, roll: pd.Series | None = None) -> pd.Series:
    """Daily $ P&L per contract of position pos (set at close t) on dF (change t -> t+1)."""
    pos = pos.reindex(dF.index).fillna(0)
    gross = pos.shift(1) * dF * UNIT
    cost = pos.diff().abs().fillna(pos.abs()) * FUT_COST * UNIT
    if roll is not None:   # rolling a held position = 2 sides
        cost = cost + pos.shift(1).abs() * roll.reindex(dF.index).fillna(0) * 2 * FUT_COST * UNIT
    return (gross - cost).fillna(0)


def straddle_pnl(pos: pd.Series, st: pd.DataFrame) -> pd.Series:
    """st indexed by date d (position date) with columns pnl, P0, gamma0, dF (F1-F0) for the
    straddle held d -> d+1. P&L is booked at d+1 index (date1) but we keep it at d for alignment."""
    pos = pos.reindex(st.index).fillna(0)
    gross = pos * st.pnl * UNIT
    trade = pos.diff().abs().fillna(pos.abs())
    cost = trade * OPT_HALF * st.P0 * UNIT + pos.abs() * FUT_COST * (st.gamma0 * st.dF).abs() * UNIT
    return (gross - cost).fillna(0)


def summarise(pnl: pd.Series, lags: int = 5) -> dict:
    out = {}
    for name, (a, b) in (("choose", CHOOSE), ("check", CHECK)):
        x = pnl[(pnl.index >= a) & (pnl.index <= b)]
        n_active = (x != 0).sum()
        out[f"{name}_mean"] = x.mean()
        out[f"{name}_ann"] = x.mean() * 252
        out[f"{name}_sr"] = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
        out[f"{name}_t"] = hac_mean_t(x, lags)
        out[f"{name}_n"] = int(n_active)
    return out


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


class no_costs:
    """Context manager: zero all cost parameters (for gross P&L)."""
    def __enter__(self):
        global FUT_COST, OPT_HALF
        self.saved = (FUT_COST, OPT_HALF)
        FUT_COST, OPT_HALF = 0.0, 0.0

    def __exit__(self, *a):
        global FUT_COST, OPT_HALF
        FUT_COST, OPT_HALF = self.saved
