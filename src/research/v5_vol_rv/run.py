"""Run the whole v5_vol_rv study: python -m src.research.v5_vol_rv.run [--force]
Steps: base table (~15 s) -> VRP map (~5 s) -> RV forecasts (~15 s) -> trade + variants (~2 min)
-> term structure (~1 min) -> figures (~30 s).  Cache: data/processed/research_v5_vol_rv/."""
from __future__ import annotations

import sys

import pandas as pd

from . import base, vrp, forecast, trade, term, figures


def main(force=False):
    pd.set_option("display.width", 220)
    base.build(force=force)
    vrp.run()
    forecast.run(force=force)
    trade.main(force=force)
    trade.variants(force=False)
    print(trade.regime_winter().round(2).to_string(index=False))
    term.run(force=force)
    figures.fig_vrp(); figures.fig_trade(); figures.fig_term()


if __name__ == "__main__":
    main(force="--force" in sys.argv)
