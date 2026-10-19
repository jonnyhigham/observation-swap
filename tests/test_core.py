"""Tests, including the two controls that establish the estimator works.

The controls matter beyond the package: a reader's first objection to a small
measured gap is that the estimator cannot detect a large one. ``test_positive_control``
answers that directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observation_swap import aggregate_records, simulate, swap_test

KW = dict(unit="unit", window="window", process="process", value="value",
          weight="weight", min_weight=10, n_boot=400, seed=7)


def test_negative_control_behavioural_recurrence_gives_near_zero_gap():
    """Recurrence from persistent unit traits must survive the swap."""
    df = simulate.behavioural(n_units=900, n_windows=4, seed=1)
    r = swap_test(df, **KW)
    assert r.summary["r_same_process_A"] > 0.5, "simulation should show recurrence"
    assert abs(r.gap_share) < 0.10, f"expected a small gap, got {r.gap_share:.3f}"


def test_positive_control_observational_recurrence_gives_large_gap():
    """Recurrence from persistent per-process conditions must not survive it."""
    df = simulate.observational(n_units=900, n_windows=4, seed=2)
    r = swap_test(df, **KW)
    assert r.summary["r_same_process_A"] > 0.5, "simulation should show recurrence"
    assert r.gap_share > 0.80, f"expected a large gap, got {r.gap_share:.3f}"
    assert r.summary["n_pairs_gap_interval_excludes_zero"] == r.summary["n_window_pairs"]


def test_gap_increases_monotonically_with_the_artefact():
    """More observation-conditions variance must mean a larger gap."""
    shares = []
    for cond in (0.0, 0.5, 1.0, 2.0):
        df = simulate.mixed(n_units=700, n_windows=3, trait_sd=1.0,
                            condition_sd=cond, seed=3)
        shares.append(swap_test(df, **KW).gap_share)
    assert all(b > a - 0.02 for a, b in zip(shares, shares[1:])), shares
    assert shares[-1] > shares[0] + 0.4


def test_common_cohort_is_identical_across_all_six_correlations():
    df = simulate.mixed(n_units=300, n_windows=3, seed=4)
    # Make one cell ineligible for some units and check every pair shrinks together.
    mask = (df.unit.isin(df.unit.unique()[:50])) & (df.window == 1) & (df.process == "B")
    df.loc[mask, "weight"] = 1.0
    r = swap_test(df, **KW)
    for p in r.pairs:
        if "estimates" not in p:
            continue
        if 1 in (p["window_1"], p["window_2"]):
            assert p["n_units"] <= 250


def test_transforms_agree_in_direction():
    df = simulate.observational(n_units=600, n_windows=3, seed=5)
    # logit needs values on the unit interval
    df = df.assign(value=1.0 / (1.0 + np.exp(-df.value)))
    out = {t: swap_test(df, **{**KW, "transform": t}).gap_share
           for t in ("identity", "logit", "rank")}
    assert all(v > 0.5 for v in out.values()), out


def test_rejects_a_process_column_without_exactly_two_levels():
    df = simulate.mixed(n_units=100, n_windows=2, seed=6)
    df.loc[df.index[:10], "process"] = "C"
    with pytest.raises(ValueError, match="exactly two levels"):
        swap_test(df, **KW)


def test_rejects_duplicate_cells():
    df = simulate.mixed(n_units=100, n_windows=2, seed=6)
    df = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        swap_test(df, **KW)


def test_missing_column_names_are_reported():
    df = simulate.mixed(n_units=60, n_windows=2, seed=6)
    with pytest.raises(KeyError, match="nope"):
        swap_test(df, **{**KW, "value": "nope"})


def test_aggregate_records_round_trip():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({
        "id": rng.choice([f"u{i}" for i in range(20)], 4000),
        "win": rng.integers(0, 3, 4000),
        "src": rng.choice(["A", "B"], 4000),
        "moving": rng.random(4000) < 0.4,
    })
    agg = aggregate_records(raw, unit="id", window="win", process="src",
                            outcome="moving")
    assert {"id", "win", "src", "moving", "weight"} <= set(agg.columns)
    assert not agg.duplicated(["id", "win", "src"]).any()
    assert agg.weight.sum() == len(raw)


def test_low_cross_process_agreement_warns():
    """If the two processes disagree within window, the swap is hard to read."""
    df = simulate.mixed(n_units=500, n_windows=3, trait_sd=1.0, condition_sd=0.0,
                        noise_sd=3.0, seed=8)
    r = swap_test(df, **KW)
    assert any("same-window agreement" in w for w in r.warnings)


def test_report_renders_and_mentions_the_gap():
    df = simulate.behavioural(n_units=400, n_windows=3, seed=9)
    text = swap_test(df, **KW).report()
    assert "GAP" in text and "What this does not establish" in text


def test_shared_observation_bias_is_indistinguishable_from_a_unit_trait():
    """Interpretation guard: shared measurement bias is not removed by a swap."""
    df = simulate.behavioural(n_units=400, n_windows=2, seed=19)
    # The identical numerical model can represent shared detection conditions;
    # the estimator cannot decide which physical interpretation generated it.
    r = swap_test(df, **{**KW, "n_boot": 20, "sparsity_split": False})
    assert r.summary["r_swapped_A_to_B"] > .8
    assert abs(r.gap_share) < .1
    report = r.report()
    assert "Shared observation bias" in report
    assert "not generally" in report
    assert "carried by the units" not in report


def test_unequal_precision_can_give_a_gap_without_specific_bias():
    z = 2*((np.arange(32)[:, None] >> np.arange(5)) & 1)-1
    rows = []
    for u in range(32):
        for w in range(2):
            for p in range(2):
                val = z[u,0] + .35*(1 if p == 0 else 4)*z[u,1+2*w+p]
                rows.append(dict(unit=u, window=w, process="AB"[p], value=val, weight=100))
    r = swap_test(pd.DataFrame(rows), **{**KW, "n_boot": 0, "sparsity_split": False})
    expected = .5*(1/1.1225 + 1/2.96) - 1/np.sqrt(1.1225*2.96)
    assert r.gap == pytest.approx(expected, abs=1e-12)
    assert r.gap > .06


def test_bootstrap_seed_is_reproducible_across_python_hash_seeds():
    import json
    import os
    import subprocess
    import sys
    code = '''import json
from observation_swap import simulate, swap_test
r=swap_test(simulate.behavioural(60,2,seed=8),unit="unit",window="window",
process="process",value="value",n_boot=30,sparsity_split=False,seed=3)
print(json.dumps(r.pairs[0]["bootstrap"]))'''
    outputs = [subprocess.check_output([sys.executable, "-c", code],
               env={**os.environ, "PYTHONHASHSEED": str(s)}, text=True)
               for s in [1, 123]]
    assert json.loads(outputs[0]) == json.loads(outputs[1])
