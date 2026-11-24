"""Scores for the physical benchmark -> data/processed/research_tool_physical/*.csv

    python -m src.research.tool_physical.score [chosen_variant]
"""
import json
import sys

import numpy as np
import pandas as pd

from src.research.tool_physical import bench as B

ALL_AN = list(B.AN)


def winter_mean(df, col):
    return df.groupby("Y")[col].mean().mean()


def diff_ci(df, model, ref, col="crps", B_=4000, seed=0):
    """Winter-weighted mean difference model - ref on common dates, winter-block bootstrap CI."""
    a = df[df.model == model].set_index("asof")[[col, "Y"]]
    b = df[df.model == ref].set_index("asof")[col]
    j = a.join(b.rename("ref"), how="inner").dropna()
    j["d"] = j[col] - j["ref"]
    d = j.groupby("Y")["d"].mean()
    if len(d) < 2:
        return np.nan, np.nan, np.nan, len(d)
    rng = np.random.default_rng(seed)
    bs = [d.loc[rng.choice(d.index.values, len(d))].mean() for _ in range(B_)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


def table(df, models, years, ref="clim"):
    rows = []
    for m in models:
        s = df[(df.model == m) & df.Y.isin(years) & df.crps.notna()]
        if s.empty:
            continue
        dm, lo, hi, nw = diff_ci(df[df.Y.isin(years)], m, ref)
        rows.append({"model": m, "winters": s.Y.nunique(), "dates": len(s),
                     "CRPS": winter_mean(s, "crps"), f"dCRPS_vs_{ref}": dm, "lo": lo, "hi": hi,
                     "B200": winter_mean(s, "B200"), "B400": winter_mean(s, "B400"), "B600": winter_mean(s, "B600"),
                     "cov50": winter_mean(s, "cov50"), "cov80": winter_mean(s, "cov80"), "cov90": winter_mean(s, "cov90"),
                     "PIT_mean": s.pit.mean(), "PIT_sd": s.pit.std(),
                     "w80": winter_mean(s.assign(w=s.p90 - s.p10), "w"),
                     "bias_med": winter_mean(s.assign(b=s.x_real - s.med), "b")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    base = ["clim", "plite", "m2"] + ALL_AN + ["an_F"]
    B.centred(["plite", "m2"] + ALL_AN)
    models = base + [m + "_rc" for m in ["plite", "m2"] + ALL_AN]
    df = B.load_all(models)
    df.to_csv(B.CACHE / "scores_long.csv", index=False)
    CH, CK, ALL = range(2012, 2017), range(2017, 2022), range(2012, 2022)
    out = {}
    for nm, yrs in (("choose", CH), ("check", CK), ("all", ALL)):
        t = table(df, models, yrs)
        t.to_csv(B.CACHE / f"table_{nm}.csv", index=False)
        out[nm] = t
        print(f"\n== {nm} ==\n", t.round(3).to_string(index=False))
    # choose the analog variant on the choose window (lowest winter-weighted CRPS)
    ch = out["choose"].set_index("model").loc[ALL_AN, "CRPS"].idxmin()
    (B.CACHE / "chosen.txt").write_text(ch)
    print("\nchosen on 2012-16:", ch)
    # overlap with Model 2 (2017-21 origins where Model 2 exists), vs Model 2 raw
    m2d = set(df[df.model == "m2"]["asof"])
    ov = df[df["asof"].isin(m2d)]
    t = table(ov, ["clim", "plite", "plite_rc", "m2", "m2_rc", ch, ch + "_rc", "an_F"], CK, ref="m2")
    t.to_csv(B.CACHE / "table_m2_overlap.csv", index=False)
    print("\n== Model 2 overlap (ref m2) ==\n", t.round(3).to_string(index=False))
    # by as-of month (all winters)
    rows = []
    for mo in (11, 12, 1, 2):
        s = df[df.month == mo]
        t = table(s, ["clim", "plite", "plite_rc", ch, "an_F", "m2"], ALL)
        t.insert(0, "month", mo)
        rows.append(t)
    bm = pd.concat(rows)
    bm.to_csv(B.CACHE / "table_by_month.csv", index=False)
    print("\n== by month ==\n", bm.round(3).to_string(index=False))
    # by winter
    bw = df[df.crps.notna()].pivot_table(index="Y", columns="model", values="crps").round(0)
    bw.to_csv(B.CACHE / "crps_by_winter.csv")
    print("\n== CRPS by winter ==\n", bw[["clim", "plite", "plite_rc", "m2", ch, "an_F"]].to_string())
    json.dump({"chosen": ch}, open(B.CACHE / "summary.json", "w"))
