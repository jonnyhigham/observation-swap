"""Command-line entry point: run the swap test on a CSV or Parquet table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .core import swap_test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="observation-swap",
        description="Test whether cross-window recurrence survives swapping the "
                    "observation process between windows.",
    )
    ap.add_argument("table", type=Path, help="CSV or Parquet with one row per unit/window/process")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--process", required=True, help="column labelling the observation process (two levels)")
    ap.add_argument("--value", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument("--transform", default="identity", choices=["identity", "logit", "rank"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-units", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooled-bootstrap", action="store_true",
                    help="joint whole-unit bootstrap for equal-pair mean contrasts")
    ap.add_argument("--json", type=Path, default=None, help="also write the full result here")
    a = ap.parse_args(argv)

    read = pd.read_parquet if a.table.suffix.lower() in {".parquet", ".pq"} else pd.read_csv
    df = read(a.table)
    res = swap_test(
        df, unit=a.unit, window=a.window, process=a.process, value=a.value,
        weight=a.weight, min_weight=a.min_weight, transform=a.transform,
        n_boot=a.n_boot, min_units=a.min_units, seed=a.seed,
        pooled_bootstrap=a.pooled_bootstrap,
    )
    print(res.report())
    if a.json:
        a.json.write_text(json.dumps(res.to_dict(), indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
