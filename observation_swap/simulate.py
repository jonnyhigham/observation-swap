"""Synthetic generators with a known answer, for validating the test.

Two extremes, both producing cross-window recurrence in the measured statistic:

``behavioural``
    Units differ in a persistent latent trait. Both observation processes see
    that trait with independent noise. Recurrence is real, so swapping the
    process between windows should change almost nothing and the gap should sit
    near zero.

``observational``
    Every unit has the *same* latent trait. What persists is each unit's
    observation conditions, and those conditions are specific to a process, so
    the same bias is reproduced whenever that process measures that unit.
    Recurrence is entirely an artefact, and swapping the process should destroy
    it, giving a large gap.

A mixture is available through ``mixed``. These are the positive and negative
controls for the estimator: a test that cannot separate these two cases is not
measuring what it claims to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["behavioural", "observational", "mixed"]


def _frame(values, weights, n_units, n_windows, processes) -> pd.DataFrame:
    rows = []
    for u in range(n_units):
        for w in range(n_windows):
            for pi, p in enumerate(processes):
                rows.append({
                    "unit": f"u{u:05d}",
                    "window": w,
                    "process": p,
                    "value": float(values[u, w, pi]),
                    "weight": float(weights[u, w, pi]),
                })
    return pd.DataFrame(rows)


def mixed(
    n_units: int = 800,
    n_windows: int = 4,
    *,
    trait_sd: float = 1.0,
    condition_sd: float = 1.0,
    noise_sd: float = 0.35,
    processes: tuple[str, str] = ("A", "B"),
    weight_lognormal_sd: float = 0.6,
    weight_median: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a panel whose recurrence mixes both sources.

    ``trait_sd`` scales a persistent per-unit trait that both processes see.
    ``condition_sd`` scales a persistent per-unit, *per-process* offset: that is
    the observation-conditions artefact, visible only to the process it belongs
    to. Setting one of them to zero gives a pure case.
    """
    rng = np.random.default_rng(seed)
    n_p = len(processes)
    trait = rng.normal(0.0, trait_sd, size=n_units)
    condition = rng.normal(0.0, condition_sd, size=(n_units, n_p))
    noise = rng.normal(0.0, noise_sd, size=(n_units, n_windows, n_p))
    values = trait[:, None, None] + condition[:, None, :] + noise
    weights = np.maximum(
        1.0,
        rng.lognormal(np.log(weight_median), weight_lognormal_sd,
                      size=(n_units, n_windows, n_p)).round(),
    )
    return _frame(values, weights, n_units, n_windows, processes)


def behavioural(n_units: int = 800, n_windows: int = 4, *, seed: int = 0,
                **kw) -> pd.DataFrame:
    """Recurrence entirely from persistent unit differences. Expect gap ~ 0."""
    kw.setdefault("trait_sd", 1.0)
    return mixed(n_units, n_windows, condition_sd=0.0, seed=seed, **kw)


def observational(n_units: int = 800, n_windows: int = 4, *, seed: int = 0,
                  **kw) -> pd.DataFrame:
    """Recurrence entirely from persistent per-process observation conditions.

    Expect a large positive gap: same-process correlation is high because the
    process reproduces its own per-unit bias, while the swapped correlation
    collapses to the units' shared level, which here is zero.
    """
    kw.setdefault("condition_sd", 1.0)
    return mixed(n_units, n_windows, trait_sd=0.0, seed=seed, **kw)
