"""Evaluation of the composite RND (src/tool/rnd_composite.py) against the direct G4X engine and realised S_T.

Development only: dates <= 2021-12-31; winters Y = 2011..2021 (March option expiry <= Feb 2021) are scored.
Winter 2021-22 (Y = 2022) is computed for Nov-Dec 2021 dates but NOT scored (its outcome is in 2022).
Direct densities: the v5_implied_storage cache rnd_daily_<Y>.pkl (3/2/1-component Bachelier mixtures fitted
to G4X March settles), read only, Y = 2013..2021.

    python -m src.tool.rnd_composite_eval step1 [Y ...]   # daily table per winter (cached)
    python -m src.tool.rnd_composite_eval mixgrid          # choose the decoupling-mixture parameters (2014-16)
    python -m src.tool.rnd_composite_eval analyse          # tables + figures
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from src.tool import rnd_composite as C

OUT = C.CFG["cache"]
ISD = C.ROOT / "data/processed/research_v5_implied_storage"
FAMS = ("gauss", "t4", "mix")
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
SCORED = range(2011, 2022)


def realised():
    out = {}
    for Y, f in C._futs().items():
        oe = C.expiries(Y)["H_oexp"]
        if oe <= C.CFG["cut"] and oe in f.index:
            out[Y] = float(f.H[oe] - f.J[oe])
    return out


def crps_sample(s, y):
    ss = np.sort(s)
    n = len(ss)
    e1 = np.abs(ss - y).mean()
    e2 = 2.0 * ((2 * np.arange(1, n + 1) - n - 1) * ss).sum() / (n * n)
    return e1 - 0.5 * e2


def stats_sample(s, F, y):
    r = {f"pa{int(x*100)}": float((s > F + x).mean()) for x in C.THRESH}
    qv = np.quantile(s, QS)
    r.update({f"q{int(p*100)}": float(v) for p, v in zip(QS, qv)})
    r["mean"] = float(s.mean())
    r["sd"] = float(s.std())
    if y is not None:
        r["crps"] = float(crps_sample(s, y))
        r["pit"] = float((s <= y).mean())
    return r


# ----------------------------------------------------------------------------------------------- step 1
def step1_winter(Y, force=False):
    p = OUT / f"daily_{Y}.pkl"
    if p.exists() and not force:
        return pd.read_pickle(p)
    t0 = time.time()
    yr = realised().get(Y)
    rows = []
    rho_tab = {f: [] for f in FAMS}
    for d in C._winter_days(Y):
        L = C.legs(d, Y)
        if L is None:
            continue
        F = L["FH"] - L["FJ"]
        T = L["TH"]
        cq = C.cso_quote(Y, d, F, T)
        e = C.ewma_corr(Y, d)
        base = {"date": d, "Y": Y, "tdH": L["tdH"], "T": T, "F": F, "y": yr, "FH": L["FH"], "FJ": L["FJ"],
                "ivH": L["iv_H_atm"], "ivJ": L["iv_J_atm"], "ivJh": L["iv_J_atm_h"], "q": L["q"], "q_src": L["q_src"],
                "nH": L["H"]["smile"]["n"], "nJ": L["J"]["smile"]["n"], "j_proxy": L["J"]["proxy"],
                "floored": L["J"]["floored"], "bfly_ok": L["H"]["smile"]["bfly_ok"] and L["J"]["smile"]["bfly_ok"],
                "neg_mass": max(L["H"]["neg_mass"], L["J"]["neg_mass"]),
                "rmseH": L["H"]["smile"]["rmse_iv"], "rmseJ": L["J"]["smile"]["rmse_iv"],
                "cso_valid": cq["valid"], "cso_reason": cq["reason"], "nv": cq["nv"], "n_oi_prev": cq["n_oi_prev"],
                "cso_nq": cq["n_quotes"], "rho_ewma": e}
        for fam in FAMS:
            r_c, cal = (np.nan, "n/a")
            if cq["valid"]:
                r_c, cal = C.calibrate_rho(L["H"], L["J"], cq["nv"] * np.sqrt(2 * T / np.pi), fam)
            rho_tab[fam].append({"date": d, "tdH": L["tdH"], "rho_cso": r_c, "rho_ewma": e})
            med, off = C._past_rho(Y, fam, L["tdH"])
            cand = {"cso": r_c if cq["valid"] else np.nan,
                    "hist": (e - off) if np.isfinite(e) and np.isfinite(off) else (e if np.isfinite(e) else np.nan),
                    "fixed": med if np.isfinite(med) else C.DEFAULT_RHO}
            cand["auto"] = next(v for v in (cand["cso"], cand["hist"], cand["fixed"]) if np.isfinite(v))
            src_auto = "cso" if np.isfinite(cand["cso"]) else ("hist" if np.isfinite(cand["hist"]) else "fixed")
            for mode, rho in cand.items():
                if not np.isfinite(rho):
                    continue
                rho = float(np.clip(rho, -0.5, 0.99999))
                s = C.simulate(L["H"], L["J"], rho, fam)
                r = dict(base, fam=fam, mode=mode, rho=rho, cal=cal if mode == "cso" else "",
                         src=src_auto if mode == "auto" else mode, off=off)
                r.update(stats_sample(s, F, yr))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    if C.expiries(Y)["H_oexp"] <= C.CFG["cut"]:
        for fam in FAMS:   # same format as rnd_composite.cso_rho_table -> used walk-forward by later winters
            pd.DataFrame(rho_tab[fam], columns=["date", "tdH", "rho_cso", "rho_ewma"]).to_pickle(
                C._cache_path(f"rho_{fam}_{Y}"))
    print(Y, len(df), f"{time.time() - t0:.0f}s", flush=True)
    return df


# ----------------------------------------------------------------------------------------------- direct engine
def mix_pdf(theta, x):
    w, m, s = theta
    return (w[None, :] * norm.pdf((x[:, None] - m[None, :]) / s[None, :]) / s[None, :]).sum(1)


def mix_cdf(theta, x):
    w, m, s = theta
    return (w[None, :] * ndtr((x[:, None] - m[None, :]) / s[None, :])).sum(1)


def _A(mu, var):
    sd = np.sqrt(var)
    return 2 * sd * norm.pdf(mu / sd) + mu * (2 * ndtr(mu / sd) - 1)


def mix_crps(theta, y):
    """Closed-form CRPS of a normal mixture (Grimit et al. 2006)."""
    w, m, s = theta
    t1 = (w * _A(y - m, s ** 2)).sum()
    t2 = (w[:, None] * w[None, :] * _A(m[:, None] - m[None, :], s[:, None] ** 2 + s[None, :] ** 2)).sum()
    return float(t1 - 0.5 * t2)


def mix_quantile(theta, p, lo=-200.0, hi=200.0):
    from scipy.optimize import brentq
    return brentq(lambda x: mix_cdf(theta, np.array([x]))[0] - p, lo, hi, xtol=1e-6)


def direct_table():
    p = OUT / "direct_daily.pkl"
    if p.exists():
        return pd.read_pickle(p)
    yr = realised()
    flags = pd.read_csv(ISD / "defaults_flag.csv", parse_dates=["date"])[["date", "Y", "default"]]
    rows = []
    for Y in range(2013, 2023):
        r = pd.read_pickle(ISD / f"rnd_daily_{Y}.pkl")
        r = r[r.date <= C.CFG["cut"]]
        for _, row in r.iterrows():
            th = row.theta
            F = row.F
            w, m, s = th
            o = {"date": row.date, "Y": Y, "F_dir": F, "nq": row.nq, "ncomp": row.ncomp, "rel_rmse": row.rel_rmse,
                 "smax_w": float(w[np.argmax(s)]), "smax": float(s.max())}
            for t in C.THRESH:
                o[f"pa{int(t*100)}"] = float(1 - mix_cdf(th, np.array([F + t]))[0])
            for qq in QS:
                o[f"q{int(qq*100)}"] = mix_quantile(th, qq)
            o["mean"] = float(w @ m)
            o["sd"] = float(np.sqrt(w @ (s ** 2 + m ** 2) - (w @ m) ** 2))
            y = yr.get(Y)
            if y is not None:
                o["crps"] = mix_crps(th, y)
                o["pit"] = float(mix_cdf(th, np.array([y]))[0])
            rows.append(o)
    df = pd.DataFrame(rows).merge(flags, on=["date", "Y"], how="left")
    df.to_pickle(p)
    return df


# ----------------------------------------------------------------------------------------------- mixture grid
def mixgrid():
    """Choose (P_MIX, RHO_LO) on 2014-16 dense days: upper-tail distance to the direct density."""
    p = OUT / "mixgrid.pkl"
    if p.exists():
        return pd.read_pickle(p)
    dt = direct_table()
    dense = dt[(dt.Y.between(2014, 2016)) & (dt.nq >= 15) & (dt.default != True)]
    dense = dense[dense.date.dt.month.isin([11, 12, 1, 2])].iloc[::4]
    grid = [(0.0, 0.6), (0.05, 0.3), (0.05, 0.6), (0.10, 0.3), (0.10, 0.6), (0.10, 0.8), (0.20, 0.6), (0.20, 0.8),
            (0.30, 0.8)]
    rows = []
    for _, r in dense.iterrows():
        L = C.legs(r.date, r.Y)
        if L is None:
            continue
        F = L["FH"] - L["FJ"]
        cq = C.cso_quote(r.Y, r.date, F, L["TH"])
        if not cq["valid"]:
            continue
        for pm, rl in grid:
            C.P_MIX, C.RHO_LO = pm, rl
            rho, cal = C.calibrate_rho(L["H"], L["J"], cq["nv"] * np.sqrt(2 * L["TH"] / np.pi), "mix")
            s = C.simulate(L["H"], L["J"], rho, "mix")
            o = {"date": r.date, "Y": r.Y, "p": pm, "rho_lo": rl, "rho": rho, "cal": cal}
            for t in C.THRESH:
                k = f"pa{int(t*100)}"
                o[k] = float((s > F + t).mean())
                o[k + "_dir"] = r[k]
            rows.append(o)
    C.P_MIX, C.RHO_LO = 0.05, 0.30
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    return df


# ----------------------------------------------------------------------------------------------- blend evaluation
def _crps_grid(x, cdf, y):
    return float(np.trapezoid((cdf - (x >= y)) ** 2, x))


def blendeval():
    """On days with both a direct fit and a composite reading (Y 2013..2021): CRPS of blends on a common grid."""
    p = OUT / "blendeval.pkl"
    if p.exists():
        return pd.read_pickle(p)
    dt = direct_table()
    dt = dt[dt.Y.isin(SCORED) & dt.date.dt.month.isin([11, 12, 1, 2])]
    yr = realised()
    rows = []
    for Y, g in dt.groupby("Y"):
        rd = pd.read_pickle(ISD / f"rnd_daily_{Y}.pkl").set_index("date")
        for _, r in g.iterrows():
            out = C.spread_density_composite(r.date, Y, "auto", "gauss", return_sample=True)
            if out is None:
                continue
            y = yr[Y]
            F = out["F"]
            x = np.linspace(min(F, y) - 4.0, max(F, y) + 10.0, 14001)
            ss = np.sort(out["sample"])
            cc = np.searchsorted(ss, x, side="right") / len(ss)
            cd = mix_cdf(rd.loc[r.date, "theta"], x)
            o = {"date": r.date, "Y": Y, "n_oi_prev": out["quality"]["cso_n_oi_prev"], "nq": r.nq,
                 "default": r.default, "cso_valid": out["quality"]["cso_valid"], "src": out["rho_source"]}
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                o[f"crps_w{int(w*100)}"] = _crps_grid(x, w * cd + (1 - w) * cc, y)
                o[f"pit_w{int(w*100)}"] = float(np.interp(y, x, w * cd + (1 - w) * cc))
                for t in C.THRESH:
                    o[f"pa{int(t*100)}_w{int(w*100)}"] = float(1 - np.interp(F + t, x, w * cd + (1 - w) * cc))
            o["y"], o["F"] = y, F
            rows.append(o)
        print(Y, flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(p)
    return df


# ----------------------------------------------------------------------------------------------- vs shipped direct engine
RND = C.ROOT / "data/processed/research_tool_rnd"
AGRID = np.round(np.arange(-0.5, 3.0001, 0.01), 2)


def engine_table():
    """Shipped direct engine (src/tool/rnd_direct.py, method 'sln') daily cache final_<Y>.pkl, read only."""
    out = []
    for Y in range(2013, 2023):
        f = pd.read_pickle(RND / f"final_{Y}.pkl")
        out.append(f[f.method == "sln"])
    e = pd.concat(out, ignore_index=True)
    e = e[e.date <= C.CFG["cut"]]
    return e


def vs_engine():
    """Per engine day: composite (gauss, auto) CDF on the engine's grid -> cross term int (F_d - F_c)^2 dx,
    so CRPS of any CDF blend w F_d + (1-w) F_c = w CRPS_d + (1-w) CRPS_c - w(1-w) cross."""
    p = OUT / "vs_engine.pkl"
    if p.exists():
        return pd.read_pickle(p)
    e = engine_table()
    e = e[e.date.dt.month.isin([11, 12, 1, 2]) & e.has_quotes.astype(bool)]
    rows = []
    for _, r in e.iterrows():
        out = C.spread_density_composite(r.date, r.Y, "auto", "gauss", return_sample=True)
        if out is None or not isinstance(r.cdf_abs, np.ndarray):
            continue
        ss = np.sort(out["sample"])
        cc = np.searchsorted(ss, AGRID, side="right") / len(ss)
        cross = float(np.trapezoid((r.cdf_abs - cc) ** 2, AGRID))
        y = r.get("ST", np.nan)
        rows.append({"date": r.date, "Y": r.Y, "cross": cross, "crps_comp": crps_sample(ss, y) if np.isfinite(y) else np.nan,
                     "pit_comp": float((ss <= y).mean()) if np.isfinite(y) else np.nan,
                     "F_comp": out["F"], "n_oi_prev": out["quality"]["cso_n_oi_prev"]})
    df = pd.DataFrame(rows).merge(e.drop(columns=["cdf_abs"]), on=["date", "Y"])
    df.to_pickle(p)
    return df


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "step1":
        ys = [int(a) for a in sys.argv[2:]] or list(range(2011, 2023))
        for Y in ys:
            step1_winter(Y)
    elif cmd == "direct":
        print(direct_table().shape)
    elif cmd == "vs_engine":
        print(vs_engine().shape)
    elif cmd == "blendeval":
        print(blendeval().shape)
    elif cmd == "mixgrid":
        print(mixgrid().groupby(["p", "rho_lo"]).mean(numeric_only=True))
