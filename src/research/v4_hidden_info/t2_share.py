"""Test 2: how much of premium_part is 'anticipated news' (later GEFS revisions, realised surprise)?

DIAGNOSIS ONLY: the projection uses FUTURE information (GEFS runs d+1..d+3, realised weather d+1..d+15).
pure_premium_DIAG = premium_part - LOWO projection on [late news (d-1), future GEFS (d+1..d+3),
realised surprise vs the d forecast]. Not tradable.

Tradable proxy: walk-forward (expanding by winter, >= 2 training winters) fit of the anticipated
component A on ex-ante interactions of premium_part; proxy = P - A_hat. Compared with A_hat = c*P.
Writes t2_share.csv, t2_weekly_share.csv, t2_reversal.csv, t2_proxy.csv, daily_components.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import CACHE, INS, CHOOSE, CHECK, daily, weekly, ols_fit, RNG

WIN = ["w0", "w1", "w2", "w3"]
BLOCKS = {
    "L_late": [f"news_{w}_l1" for w in WIN],
    "F1_next_run": [f"news_{w}_f1" for w in WIN],
    "F23_runs_d2_d3": [f"F23_{w}" for w in WIN],
    "R_realised_surprise": ["sur_w1", "sur_w2", "sur_w3"],
}
COMBOS = {
    "L": ["L_late"], "F1": ["F1_next_run"], "F1-3": ["F1_next_run", "F23_runs_d2_d3"],
    "R": ["R_realised_surprise"], "F1-3+R": ["F1_next_run", "F23_runs_d2_d3", "R_realised_surprise"],
    "L+F1-3+R": list(BLOCKS),
}


def cols(combo):
    return [c for b in COMBOS[combo] for c in BLOCKS[b]]


def lowo_pred(Y, X, ws):
    pr = np.full(len(Y), np.nan)
    for w in np.unique(ws):
        tr = ws != w
        b, _ = ols_fit(Y[tr], X[tr])
        pr[ws == w] = np.column_stack([np.ones((ws == w).sum()), X[ws == w]]) @ b
    return pr


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def share_table(d: pd.DataFrame, P: str, name: str, B=1000) -> list[dict]:
    allc = cols("L+F1-3+R")
    d = d.dropna(subset=[P] + allc)
    ws = d.W.values
    uw = np.unique(ws)
    idx = {w: np.where(ws == w)[0] for w in uw}
    Y = d[P].values
    rows = []
    for combo in COMBOS:
        X = d[cols(combo)].values
        _, r2i = ols_fit(Y, X)
        bs = []
        for _ in range(B):
            ii = np.concatenate([idx[w] for w in RNG.choice(uw, len(uw))])
            bs.append(ols_fit(Y[ii], X[ii])[1])
        lo = lowo_pred(Y, X, ws)
        row = {"ins": name, "combo": combo, "n": len(d), "R2_in": r2i, "lo": np.percentile(bs, 2.5),
               "hi": np.percentile(bs, 97.5), "R2_lowo": r2(Y, lo)}
        for nm, (a, b) in {"choose": CHOOSE, "check": CHECK}.items():
            m = (ws >= a) & (ws <= b)
            row[f"R2_lowo_{nm}"] = r2(Y[m], lo[m])
        rows.append(row)
    return rows


def components(d: pd.DataFrame, name: str) -> pd.DataFrame:
    """LOWO projections: late-news part, anticipated part (F+R), pure_premium_DIAG (residual)."""
    P = f"P_{name}"
    allc = cols("L+F1-3+R")
    d = d.dropna(subset=[P] + allc).copy()
    ws = d.W.values
    Y = d[P].values
    X = d[allc].values
    out = np.full((len(d), 3), np.nan)
    nL = len(BLOCKS["L_late"])
    for w in np.unique(ws):
        tr, te = ws != w, ws == w
        b, _ = ols_fit(Y[tr], X[tr])
        out[te, 0] = X[te, :nL] @ b[1:1 + nL]
        out[te, 1] = X[te, nL:] @ b[1 + nL:]
    out[:, 2] = Y - out[:, 0] - out[:, 1]
    return pd.DataFrame({"date": d.date.values, "W": ws, "ins": name, "P": Y, "late_news": out[:, 0],
                         "anticipated": out[:, 1], "pure_premium_DIAG": out[:, 2]})


def reversal(p: pd.DataFrame, comp: pd.DataFrame, name: str) -> list[dict]:
    d = comp.merge(p[["date", f"fmove5_{name}", f"fmove10_{name}", f"news_part_{name}"]], on="date")
    rows = []
    for h in (5, 10):
        y = f"fmove{h}_{name}"
        for xs in (["P"], ["anticipated", "late_news", "pure_premium_DIAG"]):
            e = d.dropna(subset=[y] + xs + [f"news_part_{name}"])
            X = sm.add_constant(e[xs + [f"news_part_{name}"]])
            m = sm.OLS(e[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
            for x in xs:
                rows.append({"ins": name, "h": h, "x": x, "slope": m.params[x], "t_hac": m.tvalues[x], "n": len(e)})
    return rows


def proxy(p: pd.DataFrame, comp: pd.DataFrame, name: str) -> dict:
    """Walk-forward ex-ante proxy for the anticipated part (target uses future info; features do not)."""
    P = f"P_{name}"
    d = comp.merge(p[["date", "mon", "news_tot", "lvl_tot", "hj_prev", "d_iv", "d_nv", f"fmove5_{name}",
                      f"news_part_{name}"]], on="date")
    sP = d.P.std()
    z = d.P / sP
    feats = pd.DataFrame({"P": d.P, "P_mon": d.P * d.mon, "P_absnews": d.P * d.news_tot.abs() / 20,
                          "P_lvl": d.P * d.lvl_tot / 50, "P_absP": d.P * z.abs(),
                          "P_hj": d.P * d.hj_prev.clip(lower=0),
                          "P_div": d.P * (d.d_nv if name == "hj" else d.d_iv).fillna(0) * 10}).fillna(0.0)
    ws = d.W.values
    A = d.anticipated.values
    ahat_full, ahat_P = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for w in np.unique(ws):
        tr = ws < w
        if len(np.unique(ws[tr])) < 2:
            continue
        te = ws == w
        # target on training rows must come from training-only projection: refit P ~ blocks on tr
        Xb = p.set_index("date").loc[d.date[tr], cols("L+F1-3+R")].values
        b, _ = ols_fit(d.P.values[tr], Xb)
        nL = len(BLOCKS["L_late"])
        Atr = Xb[:, nL:] @ b[1 + nL:]
        bf, _ = ols_fit(Atr, feats.values[tr])
        ahat_full[te] = np.column_stack([np.ones(te.sum()), feats.values[te]]) @ bf
        bp, _ = ols_fit(Atr, feats[["P"]].values[tr])
        ahat_P[te] = np.column_stack([np.ones(te.sum()), feats[["P"]].values[te]]) @ bp
    out = {"ins": name}
    for nm, (a, b) in {"choose": CHOOSE, "check": CHECK}.items():
        m = (ws >= a) & (ws <= b) & ~np.isnan(ahat_full)
        out[f"n_{nm}"] = int(m.sum())
        out[f"R2_Ahat_P_{nm}"] = r2(A[m], ahat_P[m])
        out[f"R2_Ahat_full_{nm}"] = r2(A[m], ahat_full[m])
        pr = d.P.values - ahat_full
        out[f"corr_P_pure_{nm}"] = np.corrcoef(d.P.values[m], d.pure_premium_DIAG.values[m])[0, 1]
        out[f"corr_proxy_pure_{nm}"] = np.corrcoef(pr[m], d.pure_premium_DIAG.values[m])[0, 1]
        # reversal slope of the next-5-day move on P vs on the proxy (same rows)
        y = d[f"fmove5_{name}"].values
        ok = m & ~np.isnan(y)
        for lab, x in (("P", d.P.values), ("proxy", pr)):
            mm = sm.OLS(y[ok], sm.add_constant(np.column_stack([x[ok], d[f"news_part_{name}"].values[ok]]))).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5})
            out[f"rev5_{lab}_{nm}"] = mm.params[1]
            out[f"rev5_t_{lab}_{nm}"] = mm.tvalues[1]
    return out


def weekly_share(w: pd.DataFrame) -> list[dict]:
    rows = []
    for name, ins in INS.items():
        d = w[(w.instrument == ins) & (w.in_core == True)].dropna(subset=["premium_part"])  # noqa: E712
        d = d.copy()
        d["W"] = d.winter
        d["sur_w1"] = d.real_w1 - d.lvl_w1; d["sur_w2"] = d.real_w2 - d.lvl_w2; d["sur_w3"] = d.real_w3 - d.lvl_w3
        for lab, xs in {"F_next_week": ["FN5", "FN5_w1", "FN5_w3"], "R": ["sur_w1", "sur_w2", "sur_w3"],
                        "F+R": ["FN5", "FN5_w1", "FN5_w3", "sur_w1", "sur_w2", "sur_w3"]}.items():
            e = d.dropna(subset=xs)
            Y, X = e.premium_part.values, e[xs].values
            _, r2i = ols_fit(Y, X)
            rows.append({"ins": name, "combo": lab, "n": len(e), "R2_in": r2i,
                         "R2_lowo": r2(Y, lowo_pred(Y, X, e.W.values))})
    return rows


def main():
    p = daily()
    for w in WIN:
        p[f"F23_{w}"] = p[f"news_{w}_f2"] + p[f"news_{w}_f3"]
    share, comps, rev, prox = [], [], [], []
    for name in INS:
        d = p[p[f"in_core_{name}"] == True]  # noqa: E712
        share += share_table(d, f"P_{name}", name)
        c = components(d, name)
        comps.append(c)
        rev += reversal(p, c, name)
        prox.append(proxy(p, c, name))
    pd.DataFrame(share).to_csv(CACHE / "t2_share.csv", index=False, float_format="%.4g")
    pd.concat(comps).to_csv(CACHE / "daily_components.csv", index=False, float_format="%.6g")
    pd.DataFrame(rev).to_csv(CACHE / "t2_reversal.csv", index=False, float_format="%.4g")
    pd.DataFrame(prox).to_csv(CACHE / "t2_proxy.csv", index=False, float_format="%.4g")
    ws = pd.DataFrame(weekly_share(weekly()))
    ws.to_csv(CACHE / "t2_weekly_share.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(share).round(3)); print(ws.round(3))
        print(pd.DataFrame(rev).round(3)); print(pd.DataFrame(prox).round(3).T)
        for c in comps:
            print(c.ins.iloc[0], "var shares:", (c[["late_news", "anticipated", "pure_premium_DIAG"]].var() / c.P.var()).round(3).to_dict())


if __name__ == "__main__":
    main()
