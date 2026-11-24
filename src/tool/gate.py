"""Frozen gate test for the Winter Squeeze-Risk Monitor (reports/tool/GATE.md). Scores a monitor panel.

Usage:  python -m src.tool.gate data/processed/tool_dev/panel_dev.csv          (reproduces dev reference)
        python -m src.tool.gate data/processed/tool_locked/panel_locked.csv    (the one locked run)
"""
import json
import sys

import numpy as np
import pandas as pd

from src.tool import _map_data as MD

B_BOOT, SEED = 5000, 0


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(df: pd.DataFrame) -> dict:
    df = df[df.usable == True].drop_duplicates(["as_of", "Y"]).copy()  # noqa: E712
    t = MD.winter_table()

    def clim(Y, col, thr, below):
        h = t[(t.index >= 1994) & (t.index <= Y - 1)][col]
        return float(((h < thr) if below else (h > thr)).mean())

    df["def"] = (df.x_real < -400).astype(float)
    df["sq"] = (df.S_T > 0.25).astype(float)
    df["clim_def"] = [clim(Y, "x", -400, True) for Y in df.Y]
    df["clim_sq"] = [clim(Y, "S", 0.25, False) for Y in df.Y]
    r = {"n": int(len(df)), "winters": sorted(map(int, df.Y.unique())),
         "events_def": {int(Y): int(g["def"].iloc[0]) for Y, g in df.groupby("Y")},
         "events_sq": {int(Y): int(g["sq"].iloc[0]) for Y, g in df.groupby("Y")},
         "B_Q_def": brier(df.Q_def400, df["def"]), "B_clim_def": brier(df.clim_def, df["def"]),
         "B_P_def": brier(df.P_def400_raw, df["def"]),
         "B_Q_sq": brier(df.Q_squeeze, df.sq), "B_clim_sq": brier(df.clim_sq, df.sq),
         "B_P_sq": brier(df.P_squeeze, df.sq)}
    r["G1_BSS_def"] = 1 - r["B_Q_def"] / r["B_clim_def"]
    r["G2_BSS_sq"] = 1 - r["B_Q_sq"] / r["B_clim_sq"]
    r["G3_P_minus_Q_def"] = r["B_Q_def"] - r["B_P_def"]
    rng = np.random.default_rng(SEED)
    Ys = df.Y.unique()
    bs = {"G1": [], "G2": [], "G3": []}
    for _ in range(B_BOOT):
        s = pd.concat([df[df.Y == y] for y in rng.choice(Ys, len(Ys))])
        bcd, bcs = brier(s.clim_def, s["def"]), brier(s.clim_sq, s.sq)
        bs["G1"].append(1 - brier(s.Q_def400, s["def"]) / bcd if bcd > 0 else np.nan)
        bs["G2"].append(1 - brier(s.Q_squeeze, s.sq) / bcs if bcs > 0 else np.nan)
        bs["G3"].append(brier(s.Q_def400, s["def"]) - brier(s.P_def400_raw, s["def"]))
    for k, v in bs.items():
        r[f"{k}_ci90"] = [float(x) for x in np.nanpercentile(v, [5, 95])]
    per = []
    for Y, g in df.groupby("Y"):
        per.append({"Y": int(Y), "n": len(g), "def": int(g["def"].iloc[0]), "sq": int(g.sq.iloc[0]),
                    "Q_def400": g.Q_def400.mean(), "P_def400_raw": g.P_def400_raw.mean(),
                    "clim_def": g.clim_def.iloc[0], "Q_squeeze": g.Q_squeeze.mean(),
                    "B_Q_def": brier(g.Q_def400, g["def"]), "B_clim_def": brier(g.clim_def, g["def"]),
                    "B_P_def": brier(g.P_def400_raw, g["def"])})
    r["per_winter"] = per
    # verdicts (rules frozen in reports/tool/GATE.md)
    r["G1_pass"] = bool(r["G1_BSS_def"] > 0)
    r["G2_pass"] = bool(r["G2_BSS_sq"] > 0)
    r["G3_pass"] = bool(r["G3_P_minus_Q_def"] > 0)
    # exploratory (not pass/fail): does the physical fair spread add to the forward in predicting S_T?
    r["X1_corr_F_ST"] = float(np.corrcoef(df.F, df.S_T)[0, 1])
    r["X1_corr_Pmean_ST"] = float(np.corrcoef(df.P_mean_S, df.S_T)[0, 1])
    return r


if __name__ == "__main__":
    out = score(pd.read_csv(sys.argv[1]))
    print(json.dumps(out, indent=1, default=lambda o: round(o, 4) if isinstance(o, float) else str(o)))
