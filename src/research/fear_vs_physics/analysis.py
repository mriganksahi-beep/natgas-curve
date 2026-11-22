"""Fear vs physics: level regressions, drift decomposition, walk-forward, rule grid, Jan 2014."""
import numpy as np
import pandas as pd

from . import data as D

FEAR = ["rv2", "rv4", "rv8", "maxrise", "fvol"]
PHYS = ["neg_dev_now", "neg_x_an", "p_tail"]        # signed so that + = physically tighter
TRAIN_Y = range(2011, 2018)                          # 2010-11 .. 2016-17
CHECK_Y = range(2018, 2023)                          # 2017-18 .. Dec 2021


def add_indices(p, ref=None):
    """Fear / physics composite = mean of z-scores. z uses `ref` rows (training window) if given."""
    p = p.copy()
    p["neg_dev_now"], p["neg_x_an"] = -p.dev_now, -p.x_an
    ref = p if ref is None else ref.assign(neg_dev_now=-ref.dev_now, neg_x_an=-ref.x_an)
    for blk, cols in (("Fear", FEAR), ("Phys", PHYS)):
        z = [(p[c] - ref[c].mean()) / ref[c].std() for c in cols]
        p[blk] = pd.concat(z, axis=1).mean(axis=1, skipna=True)
    return p


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, 1 - e.var() / y.var()


def fit(df, y, xs):
    d = df.dropna(subset=[y] + xs)
    return ols(d[y].values, d[xs].values), d


def boot(df, y, xs, B=2000, seed=0):
    """Winter-block bootstrap CIs for OLS coefficients (+ R2)."""
    (b0, r0), d = fit(df, y, xs)
    rng = np.random.default_rng(seed)
    ws = d.winter.unique()
    g = {w: d[d.winter == w] for w in ws}
    out = []
    for _ in range(B):
        s = pd.concat([g[w] for w in rng.choice(ws, len(ws))])
        if s[xs].std().min() < 1e-9:
            continue
        b, r = ols(s[y].values, s[xs].values)
        out.append(np.r_[b, r])
    out = np.array(out)
    lo, hi = np.quantile(out, [0.025, 0.975], axis=0)
    names = ["const"] + xs + ["R2"]
    return pd.DataFrame({"coef": np.r_[b0, r0], "lo": lo, "hi": hi,
                         "p_sign": np.r_[(np.sign(out[:, :-1]) != np.sign(b0)).mean(0) * 2, np.nan]},
                        index=names).assign(n=len(d), winters=len(ws))


def level_tables(p):
    rows = {}
    for name, xs in {"wte": ["wte"], "wte+Fear": ["wte", "Fear"], "wte+Phys": ["wte", "Phys"],
                     "wte+Fear+Phys": ["wte", "Fear", "Phys"],
                     "wte+rv4+p_tail": ["wte", "rv4", "p_tail"]}.items():
        rows[name] = boot(p, "S", xs)
    return rows


def decompose(p, train=None):
    """S = c + a*wte + f*Fear + q*Phys + e. Components measured relative to training means."""
    train = p if train is None else train
    (b, _), d = fit(train, "S", ["wte", "Fear", "Phys"])
    m = d[["wte", "Fear", "Phys"]].mean()
    out = p.copy()
    out["S_time"] = b[1] * (out.wte - m.wte)
    out["S_fear"] = b[2] * (out.Fear - m.Fear)
    out["S_phys"] = b[3] * (out.Phys - m.Phys)
    out["S_base"] = b[0] + b[1] * m.wte + b[2] * m.Fear + b[3] * m.Phys
    out["S_res"] = out.S - out.S_base - out.S_time - out.S_fear - out.S_phys
    return out, b


def drift_tables(p):
    dec, b = decompose(p)
    res = {}
    for y in ("dS_exp", "dS_4w"):
        res[(y, "S")] = boot(dec, y, ["S"])
        res[(y, "Fear")] = boot(dec, y, ["Fear"])
        res[(y, "Phys")] = boot(dec, y, ["Phys"])
        res[(y, "Fear+Phys")] = boot(dec, y, ["Fear", "Phys"])
        res[(y, "S+Fear+Phys")] = boot(dec, y, ["S", "Fear", "Phys"])
        res[(y, "decomp")] = boot(dec, y, ["S_fear", "S_phys", "S_time", "S_res"])
    return res, dec, b


# ------------------------------------------------------------------ walk-forward
def walk_forward(p, ycol, first_Y=2013):
    """For each Y, indices z-scored and all models fitted on winters < Y with outcome; predict Y."""
    preds = []
    for Y in sorted(p.Y.unique()):
        if Y < first_Y:
            continue
        tr_raw = p[(p.Y < Y)]
        te_raw = p[p.Y == Y]
        tr = add_indices(tr_raw, ref=tr_raw)
        te = add_indices(te_raw, ref=tr_raw)
        dec_tr, _ = decompose(tr)
        dec_te, _ = decompose(te, train=tr)
        trd = dec_tr.dropna(subset=[ycol])
        if trd.empty or te[ycol].isna().all():
            continue
        out = te_raw[["origin", "obs", "Y", "winter", "S", ycol]].copy()
        out["zero"] = 0.0
        out["mean"] = trd[ycol].mean()
        for name, xs in {"S": ["S"], "Fear": ["Fear"], "Phys": ["Phys"], "Fear+Phys": ["Fear", "Phys"],
                         "S+Fear+Phys": ["S", "Fear", "Phys"],
                         "decomp": ["S_fear", "S_phys", "S_time", "S_res"]}.items():
            (b, _), _ = fit(trd, ycol, xs)
            src = dec_te
            out[name] = b[0] + src[xs].values @ b[1:]
        preds.append(out)
    return pd.concat(preds).dropna(subset=[ycol])


def oos_table(wf, ycol, models=("zero", "mean", "S", "Fear", "Phys", "Fear+Phys", "S+Fear+Phys", "decomp"),
              B=2000, seed=1):
    rng = np.random.default_rng(seed)
    ws = wf.winter.unique()
    rows = []
    se = {m: (wf[ycol] - wf[m]) ** 2 for m in models}
    for m in models:
        r2z = 1 - se[m].sum() / se["zero"].sum()
        r2s = 1 - se[m].sum() / se["S"].sum()
        diffs = []
        for _ in range(B):
            pick = rng.choice(ws, len(ws))
            idx = np.concatenate([np.where(wf.winter.values == w)[0] for w in pick])
            diffs.append(1 - se[m].values[idx].sum() / se["S"].values[idx].sum())
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        rows.append({"model": m, "rmse": np.sqrt(se[m].mean()), "R2_vs_zero": r2z, "R2_vs_S": r2s,
                     "R2_vs_S_lo": lo, "R2_vs_S_hi": hi})
    return pd.DataFrame(rows).assign(n=len(wf), winters=len(ws))


# ------------------------------------------------------------------ Model 2 subsample
def m2_tables(p):
    q = p.dropna(subset=["p_tail_m2"]).copy()
    q = add_indices(q, ref=p)
    out = {}
    for y in ("S", "dS_exp", "dS_4w"):
        base = ["wte"] if y == "S" else ["S"]
        out[(y, "Fear+Phys")] = boot(q, y, base + ["Fear", "Phys"])
        out[(y, "Fear+Phys+M2")] = boot(q, y, base + ["Fear", "Phys", "p_tail_m2"])
        out[(y, "Fear+M2")] = boot(q, y, base + ["Fear", "p_tail_m2"])
    return out


# ------------------------------------------------------------------ rules
def trades(p, mask, hold):
    ycol = "S_exp" if hold == "exp" else "S_4w"
    t = p[mask & p[ycol].notna()].copy()
    t["pnl"] = -(t[ycol] - t.S) * D.MULT - D.ROUND_TRIP
    t["mae"] = t.max_adverse_exp * D.MULT if hold == "exp" else np.nan
    return t


def summ(t, B=2000, seed=2):
    if t.empty:
        return {"trades": 0, "winters": 0}
    rng = np.random.default_rng(seed)
    ws = t.winter.unique()
    g = {w: t.pnl[t.winter == w].values for w in ws}
    bs = [np.concatenate([g[w] for w in rng.choice(ws, len(ws))]).mean() for _ in range(B)] if len(ws) > 1 else [np.nan]
    pw = t.groupby("winter").pnl.mean()
    return {"trades": len(t), "winters": len(ws), "mean": t.pnl.mean(), "lo": np.nanquantile(bs, 0.025),
            "hi": np.nanquantile(bs, 0.975), "hit": (t.pnl > 0).mean(), "worst": t.pnl.min(),
            "worst_winter_mean": pw.min(), "winters_pos": f"{(pw > 0).sum()}/{len(pw)}",
            "sum_per_winter": t.pnl.sum() / len(ws)}


FEAR_GRID = [0.0, 0.010, 0.015, 0.020, 0.030, 0.050]          # rv4, $/day
PHYS_GRID = [1.01, 0.05, 0.10, 0.20, 0.30, 0.50]               # max analog P_tail (1.01 = no filter)


def rule_grid(p, fear_col="rv4", phys_col="p_tail", fgrid=FEAR_GRID, pgrid=PHYS_GRID):
    rows = []
    for hold in ("exp", "4w"):
        for qf in fgrid:
            for qp in pgrid:
                m = (p[fear_col] >= qf) & (p[phys_col].fillna(0) <= qp)
                for per, Ys in (("train", TRAIN_Y), ("check", CHECK_Y)):
                    s = summ(trades(p[p.Y.isin(Ys)], m[p.Y.isin(Ys)], hold))
                    rows.append({"hold": hold, "fear_min": qf, "phys_max": qp, "period": per, **s})
    return pd.DataFrame(rows)


def select(grid, hold, min_trades=10, min_winters=3):
    tr = grid[(grid.hold == hold) & (grid.period == "train") & (grid.trades >= min_trades) & (grid.winters >= min_winters)]
    best = tr.sort_values("mean", ascending=False).iloc[0]
    return best.fear_min, best.phys_max


def benchmarks(p, sp):
    out = []
    for hold in ("exp", "4w"):
        for per, Ys in (("train", TRAIN_Y), ("check", CHECK_Y), ("all", range(2011, 2023))):
            q = p[p.Y.isin(Ys)]
            out.append({"rule": "short every week", "hold": hold, "period": per,
                        **summ(trades(q, pd.Series(True, index=q.index), hold))})
            out.append({"rule": "short when S > 0.15", "hold": hold, "period": per,
                        **summ(trades(q, q.S > 0.15, hold))})
            first = q.groupby("Y").obs.transform("min") == q.obs
            out.append({"rule": "short first Nov week only", "hold": hold, "period": per,
                        **summ(trades(q, first, hold))})
    return pd.DataFrame(out)


def nov1_to_expiry(sp):
    rows = []
    for Y, S in sp.items():
        if Y > 2021:
            continue
        w = S[S.index >= pd.Timestamp(Y - 1, 11, 1)]
        rows.append({"Y": Y, "entry": w.iloc[0], "exit": w.iloc[-1],
                     "pnl": -(w.iloc[-1] - w.iloc[0]) * D.MULT - D.ROUND_TRIP,
                     "mae": (w.max() - w.iloc[0]) * D.MULT})
    return pd.DataFrame(rows)


def winter_pnl(p, mask, hold):
    t = trades(p, mask, hold)
    return t.groupby("winter").pnl.agg(["count", "mean", "sum", "min"])


# ------------------------------------------------------------------ walk-forward signals
def wf_signals(raw, first_Y=2013):
    """FP: fear-implied spread = fit of S on (wte, Fear), winters < Y.
    PP: physics-implied spread = fit of S on (wte, Phys), winters < Y.
    FX = S - PP: the part of the spread not backed by physics (the 'unbacked' premium)."""
    out = []
    for Y in sorted(raw.Y.unique()):
        if Y < first_Y:
            continue
        tr_raw, te_raw = raw[raw.Y < Y], raw[raw.Y == Y]
        tr, te = add_indices(tr_raw, ref=tr_raw), add_indices(te_raw, ref=tr_raw)
        (bf, _), _ = fit(tr, "S", ["wte", "Fear"])
        (bp, _), _ = fit(tr, "S", ["wte", "Phys"])
        o = te_raw[["origin"]].copy()
        o["FP"] = bf[0] + bf[1] * te.wte + bf[2] * te.Fear
        o["PP"] = bp[0] + bp[1] * te.wte + bp[2] * te.Phys
        out.append(o)
    s = pd.concat(out)
    r = raw.merge(s, on="origin", how="left")
    r["FX"] = r.S - r.PP
    return r


def rule_grid_generic(p, sig, sgrid, pgrid=(1.01,), phys_col="p_tail", Ys_train=TRAIN_Y):
    rows = []
    for hold in ("exp", "4w"):
        for qf in sgrid:
            for qp in pgrid:
                m = (p[sig] >= qf) & (p[phys_col].fillna(0) <= qp)
                for per, Ys in (("train", Ys_train), ("check", CHECK_Y)):
                    q = p[p.Y.isin(Ys)]
                    rows.append({"signal": sig, "hold": hold, "fear_min": qf, "phys_max": qp, "period": per,
                                 **summ(trades(q, m[q.index], hold))})
    return pd.DataFrame(rows)


def wf_resid_signal(raw, xs_fn, first_Y=2013, name="X"):
    """Generic walk-forward 'S minus fitted S' signal. xs_fn(tr_raw, te_raw) -> (Xtr, Xte) arrays."""
    out = []
    for Y in sorted(raw.Y.unique()):
        if Y < first_Y:
            continue
        tr, te = raw[raw.Y < Y], raw[raw.Y == Y]
        Xtr, Xte = xs_fn(tr, te)
        ok = np.isfinite(Xtr).all(axis=1)
        b, _ = ols(tr.S.values[ok], Xtr[ok])
        out.append(pd.DataFrame({"origin": te.origin.values, name: te.S.values - (b[0] + Xte @ b[1:])}))
    return raw.merge(pd.concat(out), on="origin", how="left")


def phys_variant(cols):
    def f(tr, te):
        trs, tes = [], []
        for c, sgn in cols:
            mu, sd = (sgn * tr[c]).mean(), (sgn * tr[c]).std()
            trs.append(((sgn * tr[c]) - mu) / sd)
            tes.append(((sgn * te[c]) - mu) / sd)
        ztr = pd.concat(trs, axis=1).mean(axis=1).values
        zte = pd.concat(tes, axis=1).mean(axis=1).values
        return np.column_stack([tr.wte, ztr]), np.column_stack([te.wte, zte])
    return f


def time_only(tr, te):
    return tr[["wte"]].values, te[["wte"]].values


def frozen_coefs(raw):
    """PP fitted on every dev winter (2010-11 .. Dec 2021) for the holdout test."""
    p = add_indices(raw)
    ref = raw.assign(neg_dev_now=-raw.dev_now, neg_x_an=-raw.x_an)
    (b, r2), d = fit(p, "S", ["wte", "Phys"])
    _, bl = decompose(p)
    return {"const": b[0], "wte": b[1], "Phys": b[2], "R2": r2, "n": len(d),
            "z": {c: {"mean": float(ref[c].mean()), "sd": float(ref[c].std())} for c in PHYS + FEAR},
            "level_decomp_const_wte_Fear_Phys": [float(x) for x in bl],
            "level_decomp_means_wte_Fear_Phys": [float(p.wte.mean()), float(p.Fear.mean()), float(p.Phys.mean())],
            "FEAR": FEAR,
            "PHYS": PHYS, "threshold_FX": 0.10, "hold_weeks": 4, "cost_ticks_per_side": 4}
