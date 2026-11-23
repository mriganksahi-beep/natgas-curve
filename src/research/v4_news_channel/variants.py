"""Specification search for the news model, judged ONLY on the choice window (OOS winters 2012-13..2016-17).
Check-window numbers are computed for the report but were not used to choose. Every variant is listed.

python -m src.research.v4_news_channel.variants   (~1 min)
"""
import pandas as pd

from .common import prepared_panel, decompose, window, r2_oos, boot_r2, CHOOSE, CHECK, CACHE, fmt

N4 = ["news_w0", "news_w1", "news_w2", "news_w3"]
N3 = ["news_w1", "news_w2", "news_w3"]
C3 = ["news_w1_c00", "news_w2_c00", "news_w3_c00"]


def variants(ins):
    v = {
        "tot (1 beta)": {"terms": ["news_tot"], "scale": None},
        "w0-w3": {"terms": N4, "scale": None},
        "w1-w3": {"terms": N3, "scale": None},
        "w1-w3 control run": {"terms": C3, "scale": None},
        "w1-w3 x (max(hj_dp,0)+0.05)": {"terms": N3, "scale": {"c": 0.05}},
        "w1-w3 x (max(hj_dp,0)+0.10)": {"terms": N3, "scale": {"c": 0.10}},
        "w1-w3 x (max(hj_dp,0)+0.20)": {"terms": N3, "scale": {"c": 0.20}},
        "w1-w3 x (max(hj_dp,0)+0.40)": {"terms": N3, "scale": {"c": 0.40}},
    }
    v["w1-w3 + Monday interactions"] = {"terms": N3 + [t + "_mon" for t in N3], "scale": None}
    v["tot + Monday interaction"] = {"terms": ["news_tot", "news_tot_mon"], "scale": None}
    # added AFTER seeing the full-dev lead/lag diagnostic (run.py); not eligible for adoption (see report)
    base = ["news_tot"] if ins == "d_hj" else N3
    v["POST-HOC frozen terms + same terms lagged 1 day"] = {"terms": base + [t + "_lag" for t in base], "scale": None}
    if ins != "d_hj":
        v["POST-HOC w1-w3 + news_tot lagged 1 day"] = {"terms": N3 + ["news_tot_lag"], "scale": None}
    if ins != "d_hj":
        v["w1-w3 x iv_dp/0.4 (no locked)"] = {"terms": N3, "scale": {"state": "iv_prev", "c": 0.0}}
    else:
        v["w1-w3 x px_front_dp (level)"] = {"terms": N3, "scale": {"state": "px_prev", "c": 0.0}}
    return v


def run():
    p = prepared_panel()
    for t in N3 + ["news_tot"]:
        p[t + "_mon"] = p[t] * p.is_mon
        p[t + "_lag"] = p[t].shift(1)
    rows = []
    for ins in ["r_front", "r_second", "r_strip", "d_hj"]:
        for name, spec in variants(ins).items():
            for fm_name, fm in [("Nov-Feb", (11, 12, 1, 2)), ("Oct-Mar", (10, 11, 12, 1, 2, 3))]:
                if fm_name == "Oct-Mar" and name not in ("w1-w3", "w1-w3 x (max(hj_dp,0)+0.10)"):
                    continue
                d = decompose(p, ins, spec=spec, fit_months=fm)
                d = d[d.in_core & d.news_part.notna()]
                rows.append({"instrument": ins, "variant": name, "fit_months": fm_name,
                             "choose": r2_oos(window(d, CHOOSE)), "check": r2_oos(window(d, CHECK)),
                             "choose_ci": fmt(boot_r2(window(d, CHOOSE), B=500))})
    t = pd.DataFrame(rows)
    t.to_csv(CACHE / "variants.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    print("variant count:", len(t))


if __name__ == "__main__":
    run()
