"""Gas-weighted GEFS ensemble disagreement for study v5 (weather uncertainty vs option prices).

For the 00Z run initialised on trading day d (known at the d settle), per window
w1 = d+1..d+5, w2 = d+6..d+10, w3 = d+11..d+15:
  unc_w*      mean over the window's days of the gas-weighted (heating-weight) average of the per-city
              5-member temperature standard deviation (deg F). This upper-bounds the s.d. of gas-weighted
              temperature (it assumes perfect cross-city correlation); HDD s.d. equals it when cold.
  unc_n_w*    unc_w* divided by the median per-lead s.d. on development winters (2010-10..2021-12),
              so 1.0 = typical disagreement for that horizon.
  d_unc_w*    unc_w* minus the previous trading day's run over the SAME valid dates.
Outputs: data/processed/v5/weather_unc.csv (date <= 2021-12-31) and data/processed/v5_locked/weather_unc.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.fetch import CITIES

ROOT = Path(__file__).resolve().parents[2]
W = pd.Series({c: v[2] for c, v in CITIES.items()})


def build():
    parts = []
    for f in ("gefs_reforecast_ens5.csv.gz", "gefs_operational_ens5.csv.gz"):
        d = pd.read_csv(ROOT / "data/forecasts" / f, parse_dates=["init_date", "valid_et_date"],
                        usecols=["init_date", "valid_et_date", "lead_days", "city", "temp_sd"])
        parts.append(d[d.init_date >= "2010-01-01"].dropna(subset=["temp_sd"]))
    d = pd.concat(parts)
    d["w"] = d.city.map(W)
    g = (d.temp_sd * d.w).groupby([d.init_date, d.valid_et_date, d.lead_days]).sum() / \
        d.w.groupby([d.init_date, d.valid_et_date, d.lead_days]).sum()
    g = g.rename("sd").reset_index()
    dev = g[(g.valid_et_date >= "2010-10-01") & (g.valid_et_date <= "2021-12-31")]
    norm = dev.groupby("lead_days").sd.median()
    g["sd_n"] = g.sd / g.lead_days.map(norm)
    tab = g.pivot_table(index="init_date", columns="valid_et_date", values="sd")
    tabn = g.pivot_table(index="init_date", columns="valid_et_date", values="sd_n")
    days = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", usecols=["date", "prev_date"], parse_dates=["date", "prev_date"])
    rows = []
    for dd, dp in zip(days.date, days.prev_date):
        if dd not in tab.index:
            continue
        r = {"date": dd}
        for w, (a, b) in {"w1": (1, 5), "w2": (6, 10), "w3": (11, 15)}.items():
            vd = pd.date_range(dd + pd.Timedelta(days=a), dd + pd.Timedelta(days=b))
            new = tab.loc[dd].reindex(vd)
            r[f"unc_{w}"] = new.mean() if new.notna().all() else np.nan
            nn = tabn.loc[dd].reindex(vd)
            r[f"unc_n_{w}"] = nn.mean() if nn.notna().all() else np.nan
            if dp in tab.index:
                old = tab.loc[dp].reindex(vd)
                r[f"d_unc_{w}"] = (new - old).mean() if old.notna().all() and new.notna().all() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/processed/v5_locked/weather_unc.csv", index=False)
    out[out.date <= "2021-12-31"].to_csv(ROOT / "data/processed/v5/weather_unc.csv", index=False)
    return out


if __name__ == "__main__":
    o = build()
    print(o.shape, o.date.min().date(), o.date.max().date())
