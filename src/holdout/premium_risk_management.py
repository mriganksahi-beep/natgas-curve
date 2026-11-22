"""Frozen holdout test V7 = H-PRM (agent premium_risk_management). Descriptive.

Spec (frozen in reports/agents/premium_risk_management.md section 6; not changed after seeing results):
  * Base: short 1 H-J at the settle of the first trading day on/after the first Friday on/after 1 Nov,
    held to the March expiry; daily MTM; 4 ticks/contract/side ($80 per spread per side).
  * Overlay: PHYS-M2 exactly as src/research/scenario_risk/models.py (h=1, ES97.5, walk-forward map and
    eta on earlier winters, Model 2 origin d-7), every Friday 1 Nov - 15 Feb.
    Alarm = ES / $1,530 (frozen). Alarm > 2.0 -> flat at the next settle; re-enter at the next settle
    after a Friday with alarm <= 2.0. Executed with the same code as development
    (overlays.phys_trigger, m=2, re-entry).
  * Kept = mean overlay P&L / mean base P&L; Cut = 1 - worst overlay DD / worst base DD (peak-to-trough
    within winter), over the holdout winters 2022-23 .. 2025-26.
  * PASS iff kept + cut >= 1.10 and kept >= 0.40; FAIL otherwise.
    INCONCLUSIVE if the base worst DD is under $5,000, or the base mean <= 0.
  * Reported alongside: $2k stop with 4-week re-entry (next-settle exit), half-size scaling.

Usage:
  python -m src.holdout.premium_risk_management --end 2021-12-31   # development check
  python -m src.holdout.premium_risk_management --holdout          # the one holdout run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.premium_risk_management import overlays as O
from src.research.scenario_risk import data as SD
from src.research.scenario_risk import models as SRM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/research_premium_risk_management/holdout"
M2_FULL = ROOT / "data/processed/research_implied_storage/model2_percentiles_full_to_2026.csv"
ES_MEDIAN = 1530.0
MULT, SIDE = 10_000.0, 80.0


def configure(cut, tag):
    """Point scenario_risk's loaders at a new cutoff and a private cache (their files are untouched)."""
    cache = OUT / tag
    cache.mkdir(parents=True, exist_ok=True)
    SD.CUTOFF = pd.Timestamp(cut)
    SD.CACHE = cache

    def load_model2():
        m = pd.read_csv(M2_FULL, parse_dates=["origin", "target"])
        m = m[m.origin <= SD.CUTOFF].copy()
        return m.drop(columns=["actual"]).set_index("origin")

    SD.load_model2 = load_model2


def phys_m2_es(seed=20261122):
    SRM.RNG = np.random.default_rng(seed)
    df = SD.add_storage_state(SD.weekly_panel())
    maps = SRM.fit_maps(df)
    m2 = SD.load_model2()
    ev = SD.eval_rows(df, 1)
    rows = []
    for d, r in ev[ev.index >= "2016-11-01"].iterrows():
        mp, dm = maps.get(r.Y), SRM.m2_dE(m2, d, 1)
        if mp is None or dm is None:
            continue
        q = SRM.phys(r.S, mp, dm, 1)
        rows.append(dict(date=d, Y=r.Y, S=r.S, ES=float(np.mean(q[975:])) * MULT))
    return pd.DataFrame(rows).set_index("date"), maps


def winters(years, end):
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= pd.Timestamp(end)) & c.contract_month.isin([3, 4])]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    out = {}
    for Y in years:
        d = pd.DataFrame({"S": px[(Y, 3)] - px[(Y, 4)]}).dropna()
        d = d[d.index >= pd.Timestamp(Y - 1, 8, 1)]
        nov1 = pd.Timestamp(Y - 1, 11, 1)
        fri = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
        ent = d.index[d.index.searchsorted(fri)]
        i0 = d.index.get_loc(ent)
        d = d.iloc[max(0, i0 - 30):].copy()
        d["live"] = d.index >= ent
        out[Y] = d
    return out


def metrics(daily):
    cum0 = np.r_[0.0, daily.cumsum().values]
    return cum0[-1], (cum0 - np.maximum.accumulate(cum0)).min()


def evaluate(W, alarm):
    rows = []
    for Y, w in W.items():
        live = w.live.values.copy()
        live[-1] = False
        pos = {"base": O.base(w),
               "phys_m2": O.phys_trigger(w, alarm, 2.0, True),
               "stop2k_re4": O.stop(w, 2000, 4)}
        for k, p in pos.items():
            pnl, dd = metrics(O.pnl(w, p))
            rows.append(dict(Y=Y, winter=f"{Y - 1}-{str(Y)[2:]}", rule=k, pnl=pnl, dd=dd,
                             exposure=np.asarray(p)[live].mean(), expiry=w.index[-1].date()))
        b = rows[-3]
        rows.append(dict(Y=Y, winter=b["winter"], rule="half_size", pnl=0.5 * b["pnl"], dd=0.5 * b["dd"],
                         exposure=0.5, expiry=b["expiry"]))
    return pd.DataFrame(rows)


def score(T):
    b = T[T.rule == "base"]
    out = {}
    for r, g in T.groupby("rule"):
        kept = g.pnl.mean() / b.pnl.mean()
        cut = 1 - g.dd.min() / b.dd.min()
        out[r] = dict(mean=g.pnl.mean(), worst_dd=g.dd.min(), worst_winter=g.pnl.min(), kept=kept, cut=cut,
                      kept_plus_cut=kept + cut)
    return pd.DataFrame(out).T


def verdict(S, T):
    b = T[T.rule == "base"]
    if b.dd.min() > -5000 or b.pnl.mean() <= 0:
        return "INCONCLUSIVE"
    p = S.loc["phys_m2"]
    return "PASS" if (p.kept_plus_cut >= 1.10 and p.kept >= 0.40) else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.holdout:
        tag, cut, years = "holdout", "2026-03-31", [2023, 2024, 2025, 2026]
    else:
        tag, cut, years = "dev", a.end, list(range(2017, 2023))
    configure(cut, tag)
    es, maps = phys_m2_es()
    es["alarm_frozen"] = es.ES / ES_MEDIAN
    med = [np.median(es.ES.values[:i]) if i >= 10 else np.nan for i in range(len(es))]
    es["alarm_rolling"] = es.ES.values / np.array(med)
    es.to_csv(OUT / tag / "phys_m2_es.csv")
    log = [f"mode={tag} cutoff={cut} winters={years}",
           "maps: " + json.dumps({int(k): {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in maps.items()})]
    W = winters(years, cut)
    T = evaluate(W, es.alarm_frozen)
    S = score(T)
    if not a.holdout:
        # reproduce development numbers: PHYS-M2 ES vs scenario_risk's cached forecasts, and V7 cells
        fc = pd.read_pickle(ROOT / "data/processed/research_scenario_risk/forecasts.pkl")
        p = fc[(fc.model == "PHYS-M2") & (fc.h == 1)].set_index("date")
        p_es = p.q.apply(lambda q: float(np.mean(q[975:]))) * MULT
        j = pd.concat([es.ES, p_es.rename("ES_cached")], axis=1).dropna()
        log.append(f"ES reproduction: n={len(j)} (cached {len(p_es)}), corr={j.corr().iloc[0, 1]:.4f}, "
                   f"median |rel diff|={(j.ES / j.ES_cached - 1).abs().median():.3%}, "
                   f"max={(j.ES / j.ES_cached - 1).abs().max():.3%}; median ES {es.ES.median():.0f}")
        Tr = evaluate(W, es.alarm_rolling)
        log.append("Rolling-median alarm (as in the report), per winter:\n" +
                   Tr.pivot(index="winter", columns="rule", values="pnl").round(0).to_string())
        log.append("Score with rolling-median alarm:\n" + score(Tr).round(3).to_string())
    log.append("Per winter (frozen alarm):\n" + T.round(2).to_string(index=False))
    log.append("Alarm by Friday:\n" + es.round(2).to_string())
    log.append("Score:\n" + S.round(3).to_string())
    log.append(f"VERDICT ({tag}): {verdict(S, T)}")
    T.to_csv(OUT / tag / "per_winter.csv", index=False)
    S.to_csv(OUT / tag / "score.csv")
    txt = "\n\n".join(log)
    (OUT / tag / "output.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
