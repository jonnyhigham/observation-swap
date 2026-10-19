"""observation-swap: is cross-window recurrence in the units, or in how you observed them?

    from observation_swap import swap_test
    result = swap_test(df, unit="id", window="window",
                       process="receiver", value="duty", weight="n", min_weight=10)
    print(result.report())

See `core.swap_test` for the estimand and `simulate` for the controls that
validate it.
"""
from .core import SwapResult, aggregate_records, swap_test, TRANSFORMS
from . import simulate

__version__ = "0.1.0"
__all__ = ["swap_test", "aggregate_records", "SwapResult", "TRANSFORMS", "simulate"]
