"""Step 2: can ex-ante features (known at the d settle) separate spikes that reversed from spikes that ran?

Spikes: days with |move| >= the choose-window 90th percentile (Nov-Feb, news available), second month
and H-J. Outcome (ex post): reversal in units of the trailing vol,
    rev_h = -sign(move_d) * fut_h / sd60_d     (h = 10, 20; > 0 = moved back)
and the binary "reversed" (fut_h undoes >= 50% of the day's move) vs "ran" (fut_h continues the move).
Features are signed so that a higher value should mean MORE reversal. Statistic: Spearman rank
correlation with rev_h and AUC (reversed vs ran), per window; winter-bootstrap CI on all dev.
No fitting beyond the walk-forward split, so nothing here is tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import CACHE, CHOOSE, CHECK, win

FEATURES = {
    # name: (description, function(df, name) -> series, higher = more reversal expected)
    "abs_move_z": "size of the day's move, |move|/sd60 (benchmark: bigger = more reversal?)",
    "raw_cum5_z": "naive: 5-day cumulative raw move in the spike's direction / sd60",
    "prem_share_d": "daily split: premium share of the day's move, prem/move",
    "cprem5_z": "daily split: 5-day cumulative premium in the spike's direction / sd60",
    "neg_wnews5_z": "rolling weekly split: minus 5-day news (weekly betas) in the spike's direction / sd60",
    "wprem5_z": "rolling weekly split: 5-day premium (weekly betas) in the spike's direction / sd60",
    "wprem_share5": "rolling weekly split: premium share of the 5-day move (in spike direction, clipped)",
    "tight": "spread level at d-1 (hj_prev)",
    "iv_front": "front ATM implied vol at d",
    "d_iv5": "5-day change in front implied vol",
    "wprem5_x_tight": "wprem5_z x max(hj_prev, 0)",
    "up": "1 if the spike is up",
    "neg_lvl12": "minus forecast cold level (lvl_w1 + lvl_w2), signed by spike direction",
}


def add_features(p: pd.DataFrame, name: str) -> pd.DataFrame:
    mv = p[f"move_{name}"]
    s = np.sign(mv)
    sd = p[f"sd60_{name}"]
    f = pd.DataFrame(index=p.index)
    f["abs_move_z"] = mv.abs() / sd
    f["raw_cum5_z"] = s * p[f"cmove5_{name}"] / (sd * np.sqrt(5))
    f["prem_share_d"] = (p[f"prem_{name}"] / mv).clip(-2, 2)
    f["cprem5_z"] = s * p[f"cprem5_{name}"] / (sd * np.sqrt(5))
    f["neg_wnews5_z"] = -s * p[f"wnews5_{name}"] / (sd * np.sqrt(5))
    f["wprem5_z"] = s * p[f"wprem5_{name}"] / (sd * np.sqrt(5))
    f["wprem_share5"] = (p[f"wprem5_{name}"] / p[f"cmove5_{name}"]).where(
        np.sign(p[f"cmove5_{name}"]) == s).clip(-2, 2)
    f["tight"] = p.hj_prev
    f["iv_front"] = p.iv_front
    f["d_iv5"] = p.iv_front - p.iv_front.shift(5)
    f["wprem5_x_tight"] = f.wprem5_z * p.hj_prev.clip(lower=0)
    f["up"] = (mv > 0).astype(float)
    f["neg_lvl12"] = -s * (p.lvl_w1 + p.lvl_w2)
    return f


def auc(score, label):
    m = score.notna() & label.notna()
    s, l = score[m].to_numpy(), label[m].to_numpy().astype(bool)
    if l.sum() < 3 or (~l).sum() < 3:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum())


def main(B=400, seed=11):
    p = pd.read_pickle(CACHE / "daily.pkl")
    rng = np.random.default_rng(seed)
    res, spikes_out = [], []
    for name in ("second", "hj"):
        p[f"sd60_{name}"] = p[f"move_{name}"].rolling(60, min_periods=40).std().shift(1)
        base = p[(p[f"core_{name}"] == True) & p[f"news_{name}"].notna()].copy()  # noqa: E712
        thr = win(base, CHOOSE)[f"move_{name}"].abs().quantile(0.9)
        sp = base[base[f"move_{name}"].abs() >= thr].copy()
        F = add_features(p, name).loc[sp.index]
        sgn = np.sign(sp[f"move_{name}"])
        for h in (10, 20):
            # drift: mean fut_h over Nov-Feb days of PRIOR winters (walk-forward), removed in rev{h}adj
            fm = base.groupby("W")[f"fut{h}_{name}"].agg(["sum", "count"])
            drift = (fm["sum"].cumsum().shift(1) / fm["count"].cumsum().shift(1)).reindex(sp.W).to_numpy()
            sp[f"rev{h}"] = -sgn * sp[f"fut{h}_{name}"] / sp[f"sd60_{name}"]
            sp[f"rev{h}adj"] = -sgn * (sp[f"fut{h}_{name}"] - drift) / sp[f"sd60_{name}"]
            frac = -sgn * sp[f"fut{h}_{name}"] / sp[f"move_{name}"].abs()
            sp[f"lab{h}"] = np.where(frac >= 0.5, 1.0, np.where(frac <= 0, 0.0, np.nan))
            sp.loc[sp[f"fut{h}_{name}"].isna(), f"lab{h}"] = np.nan
        sp = pd.concat([sp.drop(columns=[c for c in F.columns if c in sp]), F], axis=1)
        spikes_out.append(sp.assign(ins=name)[["date", "ins", "W", f"move_{name}", f"news_{name}", f"prem_{name}",
                                               "rev10", "rev20", "rev10adj", "rev20adj", "lab10", "lab20"] + list(F.columns)]
                          .rename(columns={f"move_{name}": "move", f"news_{name}": "news", f"prem_{name}": "prem"}))
        for h in (10, 20):
            for feat in F.columns:
                row = {"ins": name, "h": h, "feature": feat, "thr_abs_move": thr}
                for wn, w in (("choose", CHOOSE), ("check", CHECK), ("all", (CHOOSE[0], CHECK[1]))):
                    d = win(sp, w).dropna(subset=[feat, f"rev{h}"])
                    row[f"n_{wn}"] = len(d)
                    row[f"rho_{wn}"] = spearmanr(d[feat], d[f"rev{h}"])[0] if len(d) > 5 else np.nan
                    row[f"auc_{wn}"] = auc(d[feat], d[f"lab{h}"])
                    if wn == "all" and len(d) > 10:
                        ws = d.W.unique(); g = {x: d[d.W == x] for x in ws}
                        bs = []
                        for _ in range(B):
                            dd = pd.concat([g[x] for x in rng.choice(ws, len(ws))])
                            bs.append(spearmanr(dd[feat], dd[f"rev{h}"])[0])
                        row["rho_all_lo"], row["rho_all_hi"] = np.nanpercentile(bs, [2.5, 97.5])
                    if wn == "all":
                        dd = d.dropna(subset=[f"rev{h}adj"])
                        row["rho_all_driftadj"] = spearmanr(dd[feat], dd[f"rev{h}adj"])[0] if len(dd) > 5 else np.nan
                res.append(row)
    out = pd.DataFrame(res)
    out.to_csv(CACHE / "step2_classify.csv", index=False, float_format="%.4g")
    sp = pd.concat(spikes_out, ignore_index=True)
    sp.to_csv(CACHE / "step2_spikes.csv", index=False, float_format="%.4g")
    # base rates
    br = sp.groupby(["ins"]).agg(n=("rev10", "size"), mean_rev10=("rev10", "mean"), mean_rev20=("rev20", "mean"),
                                 share_rev10=("lab10", "mean"), share_rev20=("lab20", "mean"),
                                 mean_rev10adj=("rev10adj", "mean"), mean_rev20adj=("rev20adj", "mean"))
    br.to_csv(CACHE / "step2_baserates.csv", float_format="%.4g")
    print(br)
    print("done step2")


if __name__ == "__main__":
    main()
