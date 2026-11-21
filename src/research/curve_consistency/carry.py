"""Part 1: adjacent-month spreads, estimated full carry c, implied convenience yields."""
import numpy as np
import pandas as pd

from .data import NPOS, CUTOFF

IS = ("2011-01-01", "2017-12-31")
OOS = ("2018-01-01", "2021-12-31")
AUTUMN = ("09-15", "11-15")   # storage near full


def adjacent_pairs(d):
    """One row per (date, near contract) for adjacent listed months within the first NPOS positions."""
    P = d[d.pos <= NPOS].sort_values(["di", "pos"])
    nxt = P.groupby("di").shift(-1)
    pr = P.assign(far=nxt.settle, kfar=nxt.k)
    pr = pr[pr.kfar == pr.k + 1].copy()
    pr["s"] = pr.far - pr.settle                    # $/MMBtu per month (adjacent months)
    pr["near_m"] = pr.contract_month
    return pr[["date", "di", "k", "pos", "near_m", "settle", "far", "s", "dte"]]


def autumn_front(pr):
    m = pr.date.dt.strftime("%m-%d")
    return pr[(pr.pos == 1) & m.between(*AUTUMN) & (pr.dte > 5)]


def estimate_c(pr, q=0.95):
    """Fixed c from 2011-2017 autumns, and a walk-forward c_Y from autumns of years < Y."""
    au = autumn_front(pr)
    fixed = {qq: au[au.date.between(*IS)].s.quantile(qq) for qq in (0.5, 0.9, 0.95, 0.99, 1.0)}
    years = range(2011, 2022)
    wf = {y: au[au.date.dt.year < y].s.quantile(q) for y in years}
    by_year = au.groupby(au.date.dt.year).s.quantile(q)
    return fixed, pd.Series(wf), by_year


def convenience(pr, c, r):
    """y = c + r*near/12 - (far - near). Negative y = contango beyond full carry."""
    out = pr.copy()
    if isinstance(c, pd.Series):            # walk-forward by year
        cc = out.date.dt.year.map(c)
    else:
        cc = c
    out["carry"] = cc + r * out.settle / 12.0
    out["y"] = out.carry - out.s
    return out


def exceedance_table(pr, c, label):
    rows = []
    for r in (0.02, 0.05):
        cy = convenience(pr, c, r)
        for per, (a, b) in (("2011-17", IS), ("2018-21", OOS)):
            g = cy[cy.date.between(a, b) & cy.y.notna()]
            ex = g[g.y < 0]
            rows.append(dict(c=label, r=r, period=per, pairs=len(g),
                             pct_exceed=100 * len(ex) / len(g),
                             days_with_any=100 * ex.di.nunique() / g.di.nunique(),
                             mean_excess_c=100 * (-ex.y).mean() if len(ex) else 0.0,
                             p95_excess_c=100 * (-ex.y).quantile(.95) if len(ex) else 0.0,
                             max_excess_c=100 * (-ex.y).max() if len(ex) else 0.0))
    return pd.DataFrame(rows)


def exceed_by(pr, c, r=0.02):
    cy = convenience(pr, c, r)
    cy["ex"] = cy.y < 0
    by_pair = cy.groupby("near_m").agg(pct=("ex", "mean"), med_y=("y", "median"),
                                       q05_y=("y", lambda s: s.quantile(.05)))
    by_year = cy.groupby(cy.date.dt.year).ex.mean()
    return by_pair, by_year, cy
