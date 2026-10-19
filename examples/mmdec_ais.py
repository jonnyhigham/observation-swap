#!/usr/bin/env python3
"""Reproduce the published AIS result through the packaged API.

This is the package's real-data validation: the numbers it prints should match
the cross-source, cross-window test reported in the originating paper, which was
computed by a standalone script. Agreement shows the general implementation and
the study-specific one estimate the same thing.

Data: MMDEC multi-source AIS positions for the English Channel region,
Zenodo 10.5281/zenodo.17491518 (file ``Dataset_AIS_POS.parquet``, 451 MB,
md5 12b824d26488e381680b2de90090cc2c). The archive labels every row
``eee-land`` (terrestrial) or ``eee-sat`` (satellite), which are the two
observation processes swapped here.

    python examples/mmdec_ais.py /path/to/Dataset_AIS_POS.parquet

Needs pyarrow in addition to the package's own dependencies.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from observation_swap import swap_test

START = pd.Timestamp("2023-07-01T00:00:00")
WINDOW_DAYS = 14
N_WINDOWS = 6
MIN_REPORTS = 10
SOURCES = ("eee-land", "eee-sat")

# Published values, for the comparison printed at the end.
PUBLISHED = {
    "r_same_process_A": 0.771,   # terrestrial, across windows
    "r_same_process_B": 0.782,   # satellite, across windows
    "r_swapped_A_to_B": 0.762,
    "r_swapped_B_to_A": 0.766,
    "gap": 0.0127,
    "r_same_window_w1": 0.964,
}


def build_panel(parquet: Path) -> pd.DataFrame:
    """Per MMSI x window x source: the report-weighted fraction with SOG >= 1 kn."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    end = START + pd.Timedelta(days=WINDOW_DAYS * N_WINDOWS)
    pieces = []
    for i in range(pf.metadata.num_row_groups):
        x = pf.read_row_group(
            i, columns=["Date", "Source", "Mmsi", "SpeedOverGround"]
        ).to_pandas()
        x["Date"] = pd.to_datetime(x["Date"])
        x = x[(x["Date"] >= START) & (x["Date"] < end)]
        x = x[x["Source"].isin(SOURCES) & x["Mmsi"].notna()
              & x["SpeedOverGround"].between(0, 102.2)]
        if x.empty:
            continue
        x["window"] = ((x["Date"] - START).dt.total_seconds()
                       // (WINDOW_DAYS * 86400)).astype(int)
        x["moving"] = (x["SpeedOverGround"] >= 1).astype(int)
        pieces.append(
            x.groupby(["Mmsi", "window", "Source"], observed=True)
            .agg(n=("moving", "size"), moving=("moving", "sum"))
            .reset_index()
        )
    panel = (pd.concat(pieces, ignore_index=True)
             .groupby(["Mmsi", "window", "Source"], as_index=False)
             .agg(n=("n", "sum"), moving=("moving", "sum")))
    panel["fraction_moving"] = panel["moving"] / panel["n"]
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    panel = build_panel(a.parquet)
    print(f"panel: {len(panel):,} MMSI x window x source cells, "
          f"{panel.Mmsi.nunique():,} distinct MMSIs\n")

    result = swap_test(
        panel,
        unit="Mmsi",
        window="window",
        process="Source",
        value="fraction_moving",
        weight="n",
        min_weight=MIN_REPORTS,
        processes=SOURCES,
        n_boot=a.n_boot,
        seed=20261019,
    )
    print(result.report())

    print("\n" + "=" * 60)
    print("Agreement with the published standalone analysis")
    print(f"{'quantity':24s} {'package':>9s} {'published':>10s} {'diff':>8s}")
    worst = 0.0
    for key, pub in PUBLISHED.items():
        got = result.summary[key]
        worst = max(worst, abs(got - pub))
        print(f"{key:24s} {got:9.4f} {pub:10.4f} {got - pub:+8.4f}")
    tol = 0.001
    print(f"\nlargest absolute difference {worst:.4f} "
          f"({'within' if worst <= tol else 'OUTSIDE'} the {tol} rounding tolerance)")
    raise SystemExit(0 if worst <= tol else 1)


if __name__ == "__main__":
    main()
