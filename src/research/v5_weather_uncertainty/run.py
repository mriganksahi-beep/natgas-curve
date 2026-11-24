"""Rerun everything: python -m src.research.v5_weather_uncertainty.run [--force]  (~3 min on 1 CPU)."""
import sys

from . import analysis, data, event, figures, forecast

if __name__ == "__main__":
    force = "--force" in sys.argv
    data.base(force=force)
    forecast.run(force=force)
    forecast.describe_coefs()
    event.run(force=force)
    analysis.main()
    analysis.extras()
    figures.fig_forecast_event()
    figures.fig_trade()
    figures.fig_cells()
