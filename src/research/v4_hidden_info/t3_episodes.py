"""Tests 3 and 4: ex-post classification of big premium days (hidden news vs never justified) and
whether any ex-ante observable separates them; up vs down asymmetry.

Ex-post justification of a day's premium_part P (price units) uses agent 1's FROZEN WEEKLY betas
(full-dev, descriptive; weekly betas because daily ones are attenuated ~2.5-4x by timing smear):
  front/second: J = sum_k b_k * X_wk  (k = 1..3), with X = later GEFS revisions over d+1..d+5 (FN5_wk)
                or the realised surprise vs the d forecast (sur_wk = real_wk - lvl_wk)
  hj:           J = b * (max(hj_dp, 0) + 0.10) * X_tot (news_tot summed d+1..d+5, or sur_tot)
Class: hidden_news if max(J_F, J_R)/P >= 0.5; partial if in [0.2, 0.5); premium otherwise
(includes wrong-sign justification). Uses realised outcomes -> ex post only.
Writes t3_episodes.csv, t3_bigdays.csv, t3_auc.csv, t4_asym.csv."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CACHE, NC, INS, RNG, daily

FZ = json.loads((NC / "frozen_coefs.json").read_text())["weekly"]


def justify(p: pd.DataFrame, name: str) -> pd.DataFrame:
    q = p.copy()
    if name == "hj":
        g = FZ["d_hj"]["slopes"][0] * (q.hj_prev.clip(lower=0) + 0.10)
        q["J_F"] = g * q.FN5
        q["J_R"] = g * q.sur_tot
    else:
        b = FZ[INS[name]]["slopes"]
        q["J_F"] = sum(b[k - 1] * q[f"FN5_w{k}"] for k in (1, 2, 3))
        q["J_R"] = sum(b[k - 1] * q[f"sur_w{k}"] for k in (1, 2, 3))
    P = q[f"P_{name}"]
    q["P"] = P
    q["fracF"] = q.J_F / P
    q["fracR"] = q.J_R / P
    best = q[["fracF", "fracR"]].max(axis=1, skipna=False)
    q["cls"] = np.where(best >= 0.5, "hidden_news", np.where(best >= 0.2, "partial", "premium"))
    q.loc[best.isna(), "cls"] = "n/a"
    # retention after drift adjustment (mean core-day move, descriptive full-dev): 1 = fully kept, 0 = reversed
    drift = q.loc[q[f"in_core_{name}"] == True, f"move_{name}"].mean()  # noqa: E712
    q["retain10"] = 1 + (q[f"fmove10_{name}"] - 10 * drift) / q[f"move_{name}"]
    q["retain5"] = 1 + (q[f"fmove5_{name}"] - 5 * drift) / q[f"move_{name}"]
    q["no_news"] = (q[f"news_part_{name}"].abs() < 0.1 * q[f"move_{name}"].abs()).astype(int)
    return q


SHOW = ["date", "ins", "move", "news_part", "P", "J_F", "J_R", "fracF", "fracR", "cls", "retain5", "retain10",
        "FN1", "FN5", "sur_tot", "news_tot", "lvl_tot", "hj_prev", "d_iv", "d_nv", "cpc610", "dow"]


def pick(q, name, dates):
    r = q[q.date.isin(pd.to_datetime(dates))].copy()
    r["ins"] = name
    r["move"] = r[f"move_{name}"]
    r["news_part"] = r[f"news_part_{name}"]
    return r[SHOW]


def auc(score, lab):
    s, l = np.asarray(score, float), np.asarray(lab, int)
    ok = ~np.isnan(s)
    s, l = s[ok], l[ok]
    if l.sum() == 0 or l.sum() == len(l):
        return np.nan
    r = pd.Series(s).rank().values
    n1 = l.sum(); n0 = len(l) - n1
    return (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wshare(df, JR, JF):
    """Dollar-weighted justified share: sum(sign(P)*J) / sum(|P|)."""
    sg = np.sign(df.P)
    return (sg * JR).sum() / df.P.abs().sum(), (sg * JF).sum() / df.P.abs().sum()


def dollar_table(p: pd.DataFrame) -> pd.DataFrame:
    """Share of big (top-decile |P|) premium moves later justified, up vs down, winter bootstrap and a
    placebo that pairs each big day with the J of a core day 20..195 rows away (circular)."""
    rows = []
    for name in INS:
        q = justify(p, name)
        q = q[(q[f"in_core_{name}"] == True) & q.P.notna() & q.J_R.notna() & q.J_F.notna()]  # noqa: E712
        q = q.reset_index(drop=True)
        big = q.P.abs() >= q.P.abs().quantile(0.9)
        pl = []
        for k in range(20, 200, 7):
            JR = q.J_R.shift(k).fillna(q.J_R.shift(k - len(q)))
            JF = q.J_F.shift(k).fillna(q.J_F.shift(k - len(q)))
            pl.append(wshare(q[big], JR[big], JF[big]))
        pl = np.array(pl)
        for lab, sel in (("all_big", big), ("up", big & (q.P > 0)), ("down", big & (q.P < 0)),
                         ("all_days", q.P.notna())):
            b = q[sel]
            ws = b.W.unique()
            bs = np.array([wshare(*(lambda bb: (bb, bb.J_R, bb.J_F))(pd.concat([b[b.W == w] for w in RNG.choice(ws, len(ws))])))
                           for _ in range(1000)])
            r, f_ = wshare(b, b.J_R, b.J_F)
            rows.append({"ins": name, "set": lab, "n": len(b), "share_realised": r,
                         "R_lo": np.percentile(bs[:, 0], 2.5), "R_hi": np.percentile(bs[:, 0], 97.5),
                         "share_later_gefs": f_, "F_lo": np.percentile(bs[:, 1], 2.5), "F_hi": np.percentile(bs[:, 1], 97.5),
                         "placebo_R_p95": np.percentile(pl[:, 0], 95) if lab == "all_big" else np.nan,
                         "placebo_F_p95": np.percentile(pl[:, 1], 95) if lab == "all_big" else np.nan})
        # up - down difference
        b = q[big]
        ws = b.W.unique()
        dd = []
        for _ in range(1000):
            bb = pd.concat([b[b.W == w] for w in RNG.choice(ws, len(ws))])
            u, d_ = bb[bb.P > 0], bb[bb.P < 0]
            if len(u) and len(d_):
                dd.append(np.array(wshare(u, u.J_R, u.J_F)) - np.array(wshare(d_, d_.J_R, d_.J_F)))
        dd = np.array(dd)
        rows.append({"ins": name, "set": "up_minus_down", "n": len(b),
                     "R_lo": np.percentile(dd[:, 0], 2.5), "R_hi": np.percentile(dd[:, 0], 97.5),
                     "F_lo": np.percentile(dd[:, 1], 2.5), "F_hi": np.percentile(dd[:, 1], 97.5),
                     "share_realised": np.nan, "share_later_gefs": np.nan})
    return pd.DataFrame(rows)


def main():
    p = daily()
    eps = []
    qh = justify(p, "hj")
    # named episodes + Uri (largest |d_hj| and |front| core days 1-24 Feb 2021) + 10 largest |d_hj|
    feb21 = qh[(qh.date >= "2021-02-01") & (qh.date <= "2021-02-24") & qh.P.notna()]
    uri = [feb21.loc[feb21.move_hj.abs().idxmax(), "date"], feb21.loc[feb21.move_front.abs().idxmax(), "date"],
           pd.Timestamp("2021-02-12")]
    top = qh[qh.P.notna()].assign(a=lambda x: x.move_hj.abs()).nlargest(10, "a").date.tolist()
    named = ["2014-01-24", "2018-11-13", "2018-11-14", "2018-11-15"] + [str(x.date()) for x in uri]
    dates = sorted(set(pd.to_datetime(named + [str(x.date()) for x in top])))
    for name in ("hj", "front"):
        eps.append(pick(justify(p, name), name, dates))
    ep = pd.concat(eps)
    ep["tag"] = ep.date.map(lambda d: ("named " if str(d.date()) in named else "") + ("top10" if d in top else ""))
    ep.to_csv(CACHE / "t3_episodes.csv", index=False, float_format="%.4g")

    # systematic: top decile |P| core days per instrument
    big, aucs, asym = [], [], []
    for name in INS:
        q = justify(p, name)
        q = q[(q[f"in_core_{name}"] == True) & q.P.notna() & (q.cls != "n/a")].copy()  # noqa: E712
        thr = q.P.abs().quantile(0.9)
        q["big"] = q.P.abs() >= thr
        b = q[q.big].copy()
        b["ins"] = name
        b["up"] = (b.P > 0).astype(int)
        b["just"] = (b.cls == "hidden_news").astype(int)
        b["signR"] = (np.sign(b.J_R) == np.sign(b.P)).astype(int)
        big.append(b)
        sgn = np.sign(b.P)
        feats = {"absP": b.P.abs(), "d_vol_signed": sgn * (b.d_nv if name == "hj" else b.d_iv),
                 "lvl_tot_signed": sgn * b.lvl_tot, "hj_prev": b.hj_prev, "cpc610_signed": sgn * b.cpc610,
                 "monday": b.mon, "no_gefs_news": b.no_news, "news_tot_signed": sgn * b.news_tot,
                 "up_move": b.up}
        for f, v in feats.items():
            for nm, (a, z) in {"choose": (2013, 2017), "check": (2018, 2022)}.items():
                m = (b.W >= a) & (b.W <= z)
                aucs.append({"ins": name, "feature": f, "window": nm, "n": int(m.sum()),
                             "n_just": int(b.just[m].sum()), "auc": auc(v[m], b.just[m])})
        # asymmetry with winter bootstrap of the difference in justified share (up - down)
        ws = b.W.unique()
        diffs, diffs_s = [], []
        for _ in range(2000):
            bb = pd.concat([b[b.W == w] for w in RNG.choice(ws, len(ws))])
            diffs.append(bb[bb.up == 1].just.mean() - bb[bb.up == 0].just.mean())
            diffs_s.append(bb[bb.up == 1].signR.mean() - bb[bb.up == 0].signR.mean())
        rest = q[~q.big]
        asym.append({"ins": name, "thr_absP": thr, "n_up": int(b.up.sum()), "n_down": int((1 - b.up).sum()),
                     "just_up": b[b.up == 1].just.mean(), "just_down": b[b.up == 0].just.mean(),
                     "diff": b[b.up == 1].just.mean() - b[b.up == 0].just.mean(),
                     "diff_lo": np.nanpercentile(diffs, 2.5), "diff_hi": np.nanpercentile(diffs, 97.5),
                     "signR_up": b[b.up == 1].signR.mean(), "signR_down": b[b.up == 0].signR.mean(),
                     "signR_diff_lo": np.nanpercentile(diffs_s, 2.5), "signR_diff_hi": np.nanpercentile(diffs_s, 97.5),
                     "just_all_big": b.just.mean(),
                     "signR_rest": (np.sign(rest.J_R) == np.sign(rest.P)).mean(),
                     "medfracR_up": b[b.up == 1].fracR.median(), "medfracR_down": b[b.up == 0].fracR.median(),
                     "retain10_med_up": b[b.up == 1].retain10.median(),
                     "retain10_med_down": b[b.up == 0].retain10.median(),
                     "retain10_med_just": b[b.just == 1].retain10.median(),
                     "retain10_med_prem": b[b.cls == "premium"].retain10.median()})
    pd.concat(big)[SHOW[:1] + ["ins", "W", "P", "J_F", "J_R", "fracF", "fracR", "cls", "retain10", "up"]].to_csv(
        CACHE / "t3_bigdays.csv", index=False, float_format="%.4g")
    pd.DataFrame(aucs).to_csv(CACHE / "t3_auc.csv", index=False, float_format="%.3g")
    pd.DataFrame(asym).to_csv(CACHE / "t4_asym.csv", index=False, float_format="%.4g")
    dt = dollar_table(p)
    dt.to_csv(CACHE / "t3_dollar_share.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        print(ep.round(3).to_string())
        a = pd.DataFrame(aucs).pivot_table(index=["ins", "feature"], columns="window", values="auc")
        print(a.round(2)); print(pd.DataFrame(aucs).groupby(["ins", "window"])[["n", "n_just"]].first())
        print(pd.DataFrame(asym).round(3).T)
        print(dt.round(3))


if __name__ == "__main__":
    main()
