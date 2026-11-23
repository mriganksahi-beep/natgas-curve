# Colab cell: scope NYMEX natural gas OPTIONS data on Databento (free calls + cost estimates).
# Nothing here is billed: symbology.resolve and metadata.get_cost are free.
# Paste into Colab after:  !pip -q install databento
import databento as db, pandas as pd
from google.colab import userdata
client = db.Historical(userdata.get("DATABENTO_API_KEY"))

DS = "GLBX.MDP3"
# Candidate parent symbols for NG options on NYMEX (outrights, financial, weeklies, calendar spread options)
cands = ["ON.OPT", "LNE.OPT", "LN.OPT", "NG.OPT", "KD.OPT",          # monthly American / European
         "LN1.OPT", "LN2.OPT", "LN3.OPT", "LN4.OPT", "LN5.OPT",       # weeklies (if listed)
         "G2.OPT", "G3.OPT", "G4.OPT", "G5.OPT", "G6.OPT", "G7.OPT",  # calendar spread options
         "G8.OPT", "G9.OPT", "G10.OPT", "G11.OPT", "G12.OPT", "IA.OPT", "D6.OPT"]
found = {}
for s in cands:
    try:
        r = client.symbology.resolve(dataset=DS, symbols=[s], stype_in="parent", stype_out="instrument_id",
                                     start_date="2024-01-02", end_date="2024-01-03")
        n = len(r.get("result", {}))
        print(f"{s:9s} instruments on 2024-01-02: {n}", "" if n else r.get("not_found"))
        if n:
            found[s] = n
    except Exception as e:
        print(s, "error:", str(e)[:120])

# What are they? One day of definitions for the found parents (cheap) -> names, underlyings.
if found:
    d = client.timeseries.get_range(dataset=DS, schema="definition", symbols=list(found), stype_in="parent",
                                    start="2024-01-02", end="2024-01-03").to_df()
    cols = [c for c in ["raw_symbol", "asset", "security_type", "instrument_class", "underlying", "strike_price",
                        "expiration", "user_defined_instrument", "group"] if c in d.columns]
    print(d[cols].drop_duplicates("raw_symbol").groupby("asset").head(3).to_string())
    print(d.groupby(["asset", "instrument_class"]).size())

# Cost estimates for the full history (2010-06-06 onward) per schema, per found parent
rows = []
for s in found:
    for schema in ["statistics", "ohlcv-1d", "definition"]:
        try:
            c = client.metadata.get_cost(dataset=DS, symbols=[s], stype_in="parent", schema=schema,
                                         start="2010-06-06", end="2026-11-20")
            z = client.metadata.get_billable_size(dataset=DS, symbols=[s], stype_in="parent", schema=schema,
                                                  start="2010-06-06", end="2026-11-20")
            rows.append((s, schema, round(c, 2), round(z / 1e6)))
        except Exception as e:
            rows.append((s, schema, None, str(e)[:60]))
print(pd.DataFrame(rows, columns=["parent", "schema", "usd", "MB"]).to_string())
