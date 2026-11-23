"""Shared helpers for the v4_news_channel analysis: OOS R^2, winter bootstrap, derived columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import decompose, load_dev_panel, winter_of, CACHE, ROOT, CUTOFF  # noqa: F401

CHOOSE = (2013, 2017)   # OOS evaluation winters in the choice window: 2012-13 .. 2016-17
CHECK = (2018, 2022)    # 2017-18 .. 2021-22 (cut 2021-12-31)
RNG = np.random.default_rng(20261123)


def prepared_panel() -> pd.DataFrame:
    p = load_dev_panel()
    p["W"] = winter_of(p.date)
    p["month"] = p.date.dt.month
    p["dow"] = p.date.dt.dayofweek
    p["hj_prev"] = p.hj - p.d_hj
    # perfect-at-d hindsight revision: realised window sum minus dp's forecast for the same dates
    for k in (1, 2, 3):
        p[f"hs_w{k}"] = p[f"real_w{k}"] - (p[f"lvl_w{k}"] - p[f"news_w{k}"])
    p["real12"] = p.real_w1 + p.real_w2
    p["iv_prev"] = p.iv_front.shift(1)
    p["px_prev"] = p.px_front * np.exp(-p.r_front / 100)
    p["is_mon"] = (p.dow == 0).astype(float)
    p["is_storage"] = p.storage_surprise_5y.notna().astype(float)
    return p


def r2_oos(df: pd.DataFrame) -> float:
    e1 = df.move - df.intercept - df.news_part
    e0 = df.move - df.train_mean
    return 1 - (e1 ** 2).sum() / (e0 ** 2).sum()


def r2_news_only(df: pd.DataFrame) -> float:
    """Share of the variance of moves removed by news_part alone (no intercept): 1 - var(prem)/var(move)."""
    return 1 - ((df.premium_part - df.premium_part.mean()) ** 2).sum() / ((df.move - df.move.mean()) ** 2).sum()


def boot_r2(df: pd.DataFrame, fn=r2_oos, B: int = 2000) -> tuple[float, float, float]:
    """Winter-block bootstrap of a pooled statistic over the winters present in df."""
    ws = df.winter.unique()
    g = {w: df[df.winter == w] for w in ws}
    # pre-aggregate sufficient stats for speed when fn is r2_oos
    if fn is r2_oos:
        s1 = {w: ((d.move - d.intercept - d.news_part) ** 2).sum() for w, d in g.items()}
        s0 = {w: ((d.move - d.train_mean) ** 2).sum() for w, d in g.items()}
        a1 = np.array([s1[w] for w in ws]); a0 = np.array([s0[w] for w in ws])
        idx = RNG.integers(0, len(ws), (B, len(ws)))
        bs = 1 - a1[idx].sum(1) / a0[idx].sum(1)
    else:
        bs = np.array([fn(pd.concat([g[w] for w in RNG.choice(ws, len(ws))])) for _ in range(B)])
    return fn(df), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def window(df: pd.DataFrame, win: tuple[int, int]) -> pd.DataFrame:
    return df[(df.winter >= win[0]) & (df.winter <= win[1])]


def fmt(t) -> str:
    v, lo, hi = t
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
