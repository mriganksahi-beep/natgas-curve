"""Frozen test V6 (fear_vs_physics, H-FvP). Descriptive per reports/synthesis_v3.md.

Spec (frozen in reports/agents/fear_vs_physics.md section 6, coefficients in
data/processed/research_fear_vs_physics/frozen_holdout.json; nothing is refitted on 2022+):
  * Weekly panel exactly as src/research/fear_vs_physics/data.py, with only the end date moved:
    origins F = EIA weeks 1 Nov .. 14 Feb, decision settle = first trading day >= F+7.
  * Phys = mean of frozen z-scores of (-dev_now, -x_an, p_tail);  PP = c + b_wte*wte + b_P*Phys.
  * FX = S - PP. Short 1 H-J when FX >= $0.10; hold 4 weeks (exit at the H expiry if sooner);
    4 ticks/contract/side = $160 per spread round trip.
  * Twin: S > $0.15 on the same weekly decision points, same hold and costs.
  * PASS: mean net/trade > 0 AND (FX worst trade > twin worst trade OR neither has a trade < -$2,000).
    INCONCLUSIVE if fewer than 2 winters trade. Otherwise FAIL.
  * Secondary (not pass/fail): slopes of dS_4w (and dS_exp) on S_fear, S_phys, S_time, residual, with the
    level decomposition frozen on dev data, for (a) the original fear index (frozen json) and
    (b) the review's level-neutral fear: nrv4 = s.d. over 20 days of dS_t / max(S_{t-1} - floor, 0.005),
    floor = -$0.20 (scenario_risk's floor bound). (b)'s level coefficients and z-scores are fitted here on
    rows with obs <= 2021-12-31 only. The floor was fixed on dev data, before any holdout run: it reproduces
    the review's beta_P (-0.56 [-0.91,-0.19] vs review -0.58 [-0.91,-0.21]).
  Segments: dev (obs <= 2021-12-31), Jan-Feb 2022 (reported separately), holdout winters 2022-23..2025-26.
  Winter 2026-27 (origins Nov 2026) is excluded: it is the live test.

Run:
  python -m src.holdout.fear_vs_physics --end 2021-12-31   # dev check (reproduce agent numbers)
  python -m src.holdout.fear_vs_physics --end 2026-11-17   # holdout, run once
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.fear_vs_physics import data as D

ROOT = Path(__file__).resolve().parents[2]
FROZ = ROOT / "data/processed/research_fear_vs_physics/frozen_holdout.json"
DEV_END = pd.Timestamp("2021-12-31")
MAX_Y = 2026
FLOOR = -0.20
THR, TWIN, BIG_LOSS = 0.10, 0.15, -2000.0
FEAR = ["rv2", "rv4", "rv8", "maxrise", "fvol"]


def load(end):
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= end].set_index("week_ending")["storage_bcf"].sort_index()
    h = pd.read_csv(ROOT / "data/raw/weekly_degree_days_own.csv", parse_dates=["week_ending"])
    h = h[h.week_ending <= end].set_index("week_ending")["hdd"].sort_index()
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= end) & c.settle.notna()]
    px = c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle").sort_index()
    assert s.index.max() <= end and h.index.max() <= end and px.index.max() <= end
    return s, h, px


def front_vol(px, end):
    lr = np.log(px).diff()
    last = px.apply(lambda col: col.last_valid_index())
    pos = pd.Series(np.arange(len(px.index)), index=px.index)
    exp_pos = {k: pos[v] for k, v in last.items() if v < end}
    keys = sorted(px.columns)
    r = pd.Series(np.nan, index=px.index)
    for i in range(len(px.index)):
        for k in keys:
            j = px.columns.get_loc(k)
            if exp_pos.get(k, 10 ** 9) - i >= 2 and np.isfinite(px.iat[i, j]):
                r.iat[i] = lr.iat[i, j]
                break
    return r.rolling(20).std() * np.sqrt(252)


def build(end):
    s, hdd, px = load(end)
    fv = front_vol(px, end)
    d_s = s.diff().dropna()
    em = {Y: s.get(D.end_march_week(Y), np.nan) for Y in range(2010, MAX_Y + 1)}
    sp = {}
    for Y in range(2011, MAX_Y + 1):
        if (Y, 3) in px.columns and (Y, 4) in px.columns:
            x = (px[(Y, 3)] - px[(Y, 4)]).dropna()
            sp[Y] = x[x.index >= pd.Timestamp(Y - 1, 8, 1)]
    rows = []
    for F in s.index:
        if F.month not in (11, 12, 1, 2) or (F.month == 2 and F.day > 14):
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y not in sp or Y > MAX_Y:
            continue
        S = sp[Y]
        expired = S.index[-1].month == 2 and S.index[-1].year == Y and S.index[-1] < end - pd.Timedelta(days=5)
        ltd = S.index[-1] if expired else pd.NaT
        pos = S.index.searchsorted(F + pd.Timedelta(days=7))
        if pos >= len(S):
            continue
        obs = S.index[pos]
        if obs > end or (pd.notna(ltd) and obs >= ltd):
            continue
        h, dh = S.iloc[: pos + 1], S.diff().iloc[: pos + 1]
        z = (S.diff() / np.maximum(S.shift(1) - FLOOR, 0.005)).iloc[: pos + 1]
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em and np.isfinite(em[Y - k])]
        a5 = np.mean(prior)
        r = {"origin": F, "obs": obs, "Y": Y, "winter": f"{Y - 1}-{str(Y)[2:]}", "S": S.iloc[pos],
             "rv2": dh.iloc[-10:].std(), "rv4": dh.iloc[-20:].std(), "rv8": dh.iloc[-40:].std(),
             "maxrise": (h - h.shift(5)).iloc[-20:].max(), "fvol": fv.get(obs, np.nan), "nrv4": z.iloc[-20:].std()}
        ltd_nom = ltd if pd.notna(ltd) else pd.Timestamp(Y, 3, 1) - pd.offsets.BDay(3)
        r["wte"] = (ltd_nom - obs).days / 7
        r["dev_now"], _ = D.calendar_dev(F, s)
        ap = D.analog_projection(F, Y, s, hdd, d_s, a5)
        r["x_an"], r["p_tail"] = ap["x_an"], ap["p_tail"]
        r["S_exp"] = S.iloc[-1] if pd.notna(ltd) else np.nan
        p4 = S.index.searchsorted(obs + pd.Timedelta(days=28))
        if p4 < len(S) and S.index[p4] <= end and (pd.isna(ltd) or S.index[p4] <= ltd):
            r["S_4w"], r["date_4w"] = S.iloc[p4], S.index[p4]
        elif pd.notna(ltd):
            r["S_4w"], r["date_4w"] = S.iloc[-1], S.index[-1]
        else:
            r["S_4w"], r["date_4w"] = np.nan, pd.NaT
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("obs").reset_index(drop=True)
    df["dS_exp"], df["dS_4w"] = df.S_exp - df.S, df.S_4w - df.S
    assert df.obs.max() <= end
    return df


def signals(df, fz):
    z = fz["z"]
    zz = lambda col, sgn, key: (sgn * df[col] - z[key]["mean"]) / z[key]["sd"]
    df["Phys"] = pd.concat([zz("dev_now", -1, "neg_dev_now"), zz("x_an", -1, "neg_x_an"),
                            zz("p_tail", 1, "p_tail")], axis=1).mean(axis=1)
    df["Fear"] = pd.concat([(df[c] - z[c]["mean"]) / z[c]["sd"] for c in FEAR], axis=1).mean(axis=1)
    df["PP"] = fz["const"] + fz["wte"] * df.wte + fz["Phys"] * df.Phys
    df["FX"] = df.S - df.PP
    df["seg"] = np.where(df.obs <= DEV_END, "dev", np.where(df.Y == 2022, "JanFeb2022", "holdout"))
    return df


def trades(df, mask):
    t = df[mask & df.S_4w.notna()].copy()
    t["pnl"] = -(t.S_4w - t.S) * D.MULT - D.ROUND_TRIP
    return t


def summ(t, B=2000, seed=0):
    if t.empty:
        return {"trades": 0, "winters": 0}
    rng = np.random.default_rng(seed)
    ws = t.winter.unique()
    g = {w: t.pnl[t.winter == w].values for w in ws}
    bs = [np.concatenate([g[w] for w in rng.choice(ws, len(ws))]).mean() for _ in range(B)] if len(ws) > 1 else [np.nan]
    return {"trades": len(t), "winters": len(ws), "mean": t.pnl.mean(), "lo": np.nanquantile(bs, 0.025),
            "hi": np.nanquantile(bs, 0.975), "worst": t.pnl.min(), "sum": t.pnl.sum()}


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def decomp(df, b, means, fear_col):
    """b = (const, wte, Fear, Phys) frozen level coefs; means = dev means of (wte, Fear, Phys)."""
    o = df.copy()
    o["S_time"] = b[1] * (o.wte - means[0])
    o["S_fear"] = b[2] * (o[fear_col] - means[1])
    o["S_phys"] = b[3] * (o.Phys - means[2])
    base = b[0] + b[1] * means[0] + b[2] * means[1] + b[3] * means[2]
    o["S_res"] = o.S - base - o.S_time - o.S_fear - o.S_phys
    return o


def slopes(o, y, B=2000, seed=1):
    xs = ["S_fear", "S_phys", "S_time", "S_res"]
    d = o.dropna(subset=[y] + xs)
    if d.winter.nunique() < 2:
        return None
    b0 = ols(d[y].values, d[xs].values)
    rng = np.random.default_rng(seed)
    ws = d.winter.unique()
    g = {w: d[d.winter == w] for w in ws}
    bs = []
    for _ in range(B):
        q = pd.concat([g[w] for w in rng.choice(ws, len(ws))])
        if q[xs].std().min() > 1e-9:
            bs.append(ols(q[y].values, q[xs].values))
    lo, hi = np.quantile(np.array(bs), [0.025, 0.975], axis=0)
    return {"n": len(d), "winters": len(ws), **{f"{x}": f"{b0[i + 1]:+.2f} [{lo[i + 1]:+.2f}, {hi[i + 1]:+.2f}]"
                                                 for i, x in enumerate(xs)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    end = pd.Timestamp(ap.parse_args().end)
    out = ROOT / "data/processed/research_fear_vs_physics" / f"holdout_run_{end.date()}"
    out.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(FROZ))
    df = signals(build(end), fz)
    df.to_pickle(out / "panel.pkl")

    # level-neutral fear: z-scores and level fit on dev rows only
    dev = df[df.obs <= DEV_END]
    mu, sd = dev.nrv4.mean(), dev.nrv4.std()
    df["FearN"] = (df.nrv4 - mu) / sd
    dv = df[df.obs <= DEV_END].dropna(subset=["S", "wte", "FearN", "Phys"])
    bN = ols(dv.S.values, dv[["wte", "FearN", "Phys"]].values)
    mN = [dv.wte.mean(), dv.FearN.mean(), dv.Phys.mean()]
    bO, mO = fz["level_decomp_const_wte_Fear_Phys"], fz["level_decomp_means_wte_Fear_Phys"]

    print(f"end {end.date()}: rows {len(df)}, segments {df.seg.value_counts().to_dict()}")
    print("level-neutral fear (dev fit): coefs const/wte/FearN/Phys", np.round(bN, 4), "corr(nrv4,S) dev",
          round(dev.nrv4.corr(dev.S), 2))
    res = {}
    for seg in (["dev"] if end <= DEV_END else ["dev", "JanFeb2022", "holdout"]):
        q = df[df.seg == seg]
        fx, tw = trades(q, q.FX >= THR), trades(q, q.S > TWIN)
        sf, st = summ(fx), summ(tw)
        res[seg] = {"FX": sf, "twin": st}
        print(f"\n== segment {seg}: {len(q)} weeks, {q.winter.nunique()} winters")
        print(" FX >= 0.10 :", {k: (round(v) if isinstance(v, float) and np.isfinite(v) else v) for k, v in sf.items()})
        print(" S > 0.15   :", {k: (round(v) if isinstance(v, float) and np.isfinite(v) else v) for k, v in st.items()})
        if len(fx):
            print(" FX per winter:", fx.groupby("winter").pnl.agg(["count", "sum", "min"]).round(0).to_dict("index"))
        if len(tw):
            print(" twin per winter:", tw.groupby("winter").pnl.agg(["count", "sum", "min"]).round(0).to_dict("index"))
        if seg != "dev" and len(fx):
            print(fx[["obs", "S", "PP", "FX", "p_tail", "dS_4w", "pnl"]].round(3).to_string(index=False))
        if seg == "holdout":
            if sf["winters"] < 2:
                v = "INCONCLUSIVE (fewer than 2 winters trade)"
            else:
                tail_ok = (st.get("trades", 0) > 0 and sf["worst"] > st["worst"]) or \
                          (sf["worst"] >= BIG_LOSS and (st.get("trades", 0) == 0 or st["worst"] >= BIG_LOSS))
                v = "PASS" if (sf["mean"] > 0 and tail_ok) else "FAIL"
                print(f" criterion 1 mean>0: {sf['mean'] > 0}; criterion 2 tail: {tail_ok}")
            res[seg]["verdict"] = v
            print(" VERDICT:", v)
        for nm, b, m, fc in (("original fear index", bO, mO, "Fear"), ("level-neutral fear", bN, mN, "FearN")):
            o = decomp(q, b, m, fc)
            for y in ("dS_4w", "dS_exp"):
                sl = slopes(o, y)
                res[seg][f"{nm}|{y}"] = sl
                print(f" slopes {nm:20s} {y}: {sl}")
    json.dump(res, open(out / "results.json", "w"), indent=1, default=str)
    if end <= DEV_END:
        ref = pd.read_pickle(ROOT / "data/processed/research_fear_vs_physics/panel.pkl")
        m = df.merge(ref[["origin", "S", "S_4w", "p_tail", "x_an"]], on="origin", suffixes=("", "_ref"))
        print("\ndev reproduction: rows", len(df), "vs", len(ref), "| max |dS|", float((m.S - m.S_ref).abs().max()),
              "| max |dp_tail|", float((m.p_tail - m.p_tail_ref).abs().max()),
              "| S_4w equal", bool(np.allclose(m.S_4w, m.S_4w_ref, equal_nan=True)))


if __name__ == "__main__":
    main()
