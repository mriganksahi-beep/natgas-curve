"""Run all v4_premium_shocks steps: python -m src.research.v4_premium_shocks.run  (~1.5 min)."""
from . import classify, figures, frozen_test, reversal, rules
from .common import CACHE, build


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    build().to_pickle(CACHE / "daily.pkl")
    reversal.main(); classify.main(); rules.main(); figures.main(); frozen_test.main()


if __name__ == "__main__":
    main()
