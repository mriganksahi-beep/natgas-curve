"""Step 1: reversal regressions of future same-contract moves on today's news and premium parts.

Daily split (h = 1, 2, 5, 10, 20 trading days) and weekly split (Friday rows, h = 1, 2, 4 weeks = 5/10/20
trading days). Rows: Nov-Feb (fit months of the split), news_part available (walk-forward from 2012-13).
Specs: naive (raw move), split (news, prem), split + controls (news at d-1, future news d+1..d+h, Monday).
HAC Newey-West lags = h + 2 days (weekly: k + 1 weeks); winter-block bootstrap 1,000 draws.
Descriptive: future news is NOT tradable information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CACHE, CHOOSE, CHECK, HS, reg, win

WINDOWS = {"choose": CHOOSE, "check": CHECK, "all": (CHOOSE[0], CHECK[1])}


def daily_rows(p, name):
    return p[(p[f"core_{name}"] == True) & p[f"news_{name}"].notna()].copy()  # noqa: E712


def weekly_rows(p, name):
    return p[p[f"wkprem_{name}"].notna() & p.month.isin([11, 12, 1, 2])].copy()


def run_daily(p, B=1000):
    out = []
    for name in ("front", "second", "hj"):
        d = daily_rows(p, name)
        for h in HS:
            y = f"fut{h}_{name}"
            specs = {"naive": [f"move_{name}"],
                     "naive_ctrl": [f"move_{name}", f"nlag_{name}", f"nfut{h}_{name}", "mon"],
                     "split": [f"news_{name}", f"prem_{name}"],
                     "split_ctrl": [f"news_{name}", f"prem_{name}", f"nlag_{name}", f"nfut{h}_{name}", "mon"]}
            for wn, w in WINDOWS.items():
                for sn, xs in specs.items():
                    r = reg(win(d, w), y, xs, lags=h + 2, B=B)
                    r = {k.replace(f"_{name}", "").replace(f"nfut{h}", "nfut"): v for k, v in r.items()}
                    out.append({"freq": "daily", "ins": name, "h": h, "window": wn, "spec": sn, **r})
    return pd.DataFrame(out)


def run_weekly(p, B=1000):
    out = []
    for name in ("front", "second", "hj"):
        d = weekly_rows(p, name)
        for k, h in ((1, 5), (2, 10), (4, 20)):
            y = f"fut{h}_{name}"
            specs = {"naive": [f"wkmove_{name}"],
                     "naive_ctrl": [f"wkmove_{name}", f"wknlag_{name}", f"wknfut{h}_{name}"],
                     "split": [f"wknews_{name}", f"wkprem_{name}"],
                     "split_ctrl": [f"wknews_{name}", f"wkprem_{name}", f"wknlag_{name}", f"wknfut{h}_{name}"]}
            for wn, w in WINDOWS.items():
                for sn, xs in specs.items():
                    r = reg(win(d, w), y, xs, lags=k + 1, B=B)
                    r = {kk.replace(f"_{name}", "").replace(f"wknfut{h}", "nfut").replace("wk", ""): v
                         for kk, v in r.items()}
                    out.append({"freq": "weekly", "ins": name, "h": h, "window": wn, "spec": sn, **r})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- heterogeneity
def hetero(p, B=500):
    """Premium slope by group (shared news + controls), daily h=10 and weekly 2 weeks."""
    out = []
    for name in ("second", "hj", "front"):
        for freq, h, rows, mv, nw, pr, nl, nf, lags in (
                ("daily", 10, daily_rows(p, name), f"move_{name}", f"news_{name}", f"prem_{name}",
                 f"nlag_{name}", f"nfut10_{name}", 12),
                ("daily", 5, daily_rows(p, name), f"move_{name}", f"news_{name}", f"prem_{name}",
                 f"nlag_{name}", f"nfut5_{name}", 7),
                ("weekly", 10, weekly_rows(p, name), f"wkmove_{name}", f"wknews_{name}", f"wkprem_{name}",
                 f"wknlag_{name}", f"wknfut10_{name}", 3)):
            if name == "front" and h == 10:
                continue
            y = f"fut{h}_{name}"
            d = rows.dropna(subset=[y, nw, pr]).copy()
            ch = win(d, CHOOSE)
            q90 = ch[pr].abs().quantile(0.9)
            t1, t2 = ch.hj_prev.quantile([1 / 3, 2 / 3])
            groups = {
                "size": {"top10": d[pr].abs() >= q90, "rest": d[pr].abs() < q90},
                "sign": {"up": d[pr] > 0, "down": d[pr] <= 0},
                "tight": {"loose": d.hj_prev < t1, "mid": (d.hj_prev >= t1) & (d.hj_prev < t2), "tight": d.hj_prev >= t2},
                "month": {m: d.month == mm for m, mm in (("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2))},
            }
            for gname, gs in groups.items():
                xs = [nw, nl, nf] if freq == "daily" else [nw, nl, nf]
                cols = []
                for lab, m in gs.items():
                    c = f"prem__{lab}"
                    d[c] = d[pr].where(m, 0.0)
                    d[f"ind__{lab}"] = m.astype(float)
                    cols.append(c)
                inds = [f"ind__{lab}" for lab in list(gs)[1:]]
                for wn, w in WINDOWS.items():
                    r = reg(win(d, w), y, cols + xs + inds, lags=lags, B=B)
                    for lab in gs:
                        c = f"prem__{lab}"
                        out.append({"ins": name, "freq": freq, "h": h, "group": gname, "level": lab, "window": wn,
                                    "n_level": int((gs[lab] & d.W.between(*w)).sum()),
                                    "b": r.get(f"b_{c}"), "t": r.get(f"t_{c}"),
                                    "lo": r.get(f"lo_{c}"), "hi": r.get(f"hi_{c}")})
                    for c in cols + xs + inds:
                        pass
            d.drop(columns=[c for c in d if c.startswith(("prem__", "ind__"))], inplace=True)
    return pd.DataFrame(out)


def episodes(p):
    """Largest premium shocks (second month and H-J), their future moves, and concentration of the slope."""
    rows, conc = [], []
    for name, h in (("second", 10), ("hj", 10), ("second", 5), ("hj", 5)):
        d = daily_rows(p, name).dropna(subset=[f"fut{h}_{name}"])
        pr, y = d[f"prem_{name}"], d[f"fut{h}_{name}"]
        c = (pr - pr.mean()) * (y - y.mean())
        slope = c.sum() / ((pr - pr.mean()) ** 2).sum()
        top = c.abs().sort_values(ascending=False)
        # leave-one-winter-out slopes
        lowo = {}
        for w in sorted(d.W.unique()):
            m = d.W != w
            a, b = pr[m], y[m]
            lowo[int(w)] = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()) ** 2).sum())
        # slope without the 10 largest |prem| days
        big = pr.abs().sort_values(ascending=False).index[:10]
        m = ~d.index.isin(big)
        a, b = pr[m], y[m]
        s_ex = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()) ** 2).sum())
        conc.append({"ins": name, "h": h, "n": len(d), "slope_all_dev": slope,
                     "share_num_top10_days": float(c[top.index[:10]].sum() / c.sum()),
                     "slope_ex_top10_prem_days": s_ex,
                     **{f"lowo_{k}": v for k, v in lowo.items()}})
        if h == 10:
            for i in pr.abs().sort_values(ascending=False).index[:12]:
                rows.append({"ins": name, "date": d.at[i, "date"].date(), "move": d.at[i, f"move_{name}"],
                             "news": d.at[i, f"news_{name}"], "prem": pr[i],
                             "fut5": d.at[i, f"fut5_{name}"], "fut10": y[i], "fut20": d.at[i, f"fut20_{name}"],
                             "rev10": -y[i] / pr[i]})
    return pd.DataFrame(rows), pd.DataFrame(conc)


def robustness(p, B=500):
    """Premium slope under sample cuts: all dev, ex winter 2020 (GEFS v11 operational), ex 2014 and 2019
    (the two episode winters), Monday rows only (cleanly timed weekend news), up/down not here (hetero)."""
    out = []
    cuts = {"all": lambda d: d, "ex_W2020": lambda d: d[d.W != 2020],
            "ex_W2014_W2019": lambda d: d[~d.W.isin([2014, 2019])], "monday_only": lambda d: d[d.mon == 1]}
    for name in ("second", "hj", "front"):
        for freq, h in (("daily", 5), ("daily", 10), ("weekly", 10), ("weekly", 20)):
            if name == "front" and h >= 10:
                continue
            for cn, cut in cuts.items():
                if freq == "weekly" and cn == "monday_only":
                    continue
                if freq == "daily":
                    d = cut(daily_rows(p, name)); y = f"fut{h}_{name}"
                    specs = {"split": [f"news_{name}", f"prem_{name}"],
                             "split_ctrl": [f"news_{name}", f"prem_{name}", f"nlag_{name}", f"nfut{h}_{name}"]
                             + ([] if cn == "monday_only" else ["mon"])}
                    lags, pr = h + 2, f"prem_{name}"
                else:
                    d = cut(weekly_rows(p, name)); y = f"fut{h}_{name}"
                    specs = {"split": [f"wknews_{name}", f"wkprem_{name}"],
                             "split_ctrl": [f"wknews_{name}", f"wkprem_{name}", f"wknlag_{name}", f"wknfut{h}_{name}"]}
                    lags, pr = h // 5 + 1, f"wkprem_{name}"
                for sn, xs in specs.items():
                    for wn, w in WINDOWS.items():
                        r = reg(win(d, w), y, xs, lags=lags, B=B)
                        out.append({"ins": name, "freq": freq, "h": h, "cut": cn, "spec": sn, "window": wn,
                                    "n": r.get("n"), "b_prem": r.get(f"b_{pr}"), "t_prem": r.get(f"t_{pr}"),
                                    "lo": r.get(f"lo_{pr}"), "hi": r.get(f"hi_{pr}")})
    return pd.DataFrame(out)


def main():
    p = pd.read_pickle(CACHE / "daily.pkl")
    a = pd.concat([run_daily(p), run_weekly(p)], ignore_index=True)
    a.to_csv(CACHE / "step1_reversal.csv", index=False, float_format="%.4g")
    hetero(p).to_csv(CACHE / "step1_hetero.csv", index=False, float_format="%.4g")
    robustness(p).to_csv(CACHE / "step1_robustness.csv", index=False, float_format="%.4g")
    ep, conc = episodes(p)
    ep.to_csv(CACHE / "step1_episodes.csv", index=False, float_format="%.4g")
    conc.to_csv(CACHE / "step1_concentration.csv", index=False, float_format="%.4g")
    print("done step1")


if __name__ == "__main__":
    main()
