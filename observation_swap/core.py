"""Compare cross-window recurrence on a four-cell common cohort.

The gap contrasts same-process with swapped-process recurrence. Under an
equal-variance additive model with independent process-specific biases it
measures the persistent process-specific variance relative to total variance.
Shared observation bias survives the swap; unequal precision and correlated
process biases also affect the contrast. It is not a causal attributable
fraction without these model assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["SwapResult", "swap_test", "aggregate_records", "TRANSFORMS"]

CELLS = ("a_w1", "b_w1", "a_w2", "b_w2")


def _logit(x: np.ndarray, eps: float = 0.01) -> np.ndarray:
    q = np.clip(x, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def _ranks(m: np.ndarray) -> np.ndarray:
    out = np.empty_like(m, dtype=float)
    for c in range(m.shape[1]):
        v = m[:, c]
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i : j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        out[:, c] = r
    return out


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda m: m,
    "logit": lambda m: _logit(m),
    "rank": _ranks,
}


def _pearson(m: np.ndarray, i: int, j: int) -> float:
    a, b = m[:, i], m[:, j]
    if a.shape[0] < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if not (np.isfinite(sa) and np.isfinite(sb)) or sa == 0 or sb == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _estimates(m: np.ndarray) -> dict[str, float]:
    """Six correlations and the gaps, for a cohort matrix ordered as CELLS."""
    k = {name: idx for idx, name in enumerate(CELLS)}
    same_w1 = _pearson(m, k["a_w1"], k["b_w1"])
    same_w2 = _pearson(m, k["a_w2"], k["b_w2"])
    aa = _pearson(m, k["a_w1"], k["a_w2"])
    bb = _pearson(m, k["b_w1"], k["b_w2"])
    ab = _pearson(m, k["a_w1"], k["b_w2"])
    ba = _pearson(m, k["b_w1"], k["a_w2"])
    reliability = 0.5 * (same_w1 + same_w2)
    disatt = (
        0.5 * (ab + ba) / reliability
        if np.isfinite(reliability) and reliability > 0
        else np.nan
    )
    return {
        "r_same_window_w1": same_w1,
        "r_same_window_w2": same_w2,
        "r_same_process_A": aa,
        "r_same_process_B": bb,
        "r_swapped_A_to_B": ab,
        "r_swapped_B_to_A": ba,
        "gap_A": aa - ab,
        "gap_B": bb - ba,
        "gap": 0.5 * ((aa - ab) + (bb - ba)),
        "disattenuated_swapped": disatt,
    }


def _bootstrap(m: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, dict]:
    """Percentile bootstrap. The resampling unit is the row, i.e. the unit."""
    n = m.shape[0]
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        for key, val in _estimates(m[rng.integers(0, n, n), :]).items():
            draws.setdefault(key, []).append(val)
    out = {}
    for key, vals in draws.items():
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size < 2:
            out[key] = {"lo": None, "hi": None, "n_draws": int(a.size)}
        else:
            out[key] = {
                "lo": float(np.percentile(a, 2.5)),
                "hi": float(np.percentile(a, 97.5)),
                "n_draws": int(a.size),
            }
    return out


@dataclass
class SwapResult:
    """Result of an observation-swap test."""

    pairs: list[dict[str, Any]]
    summary: dict[str, Any]
    design: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        return self.summary.get("gap")

    @property
    def gap_ci(self) -> tuple[float | None, float | None]:
        ci = self.summary.get("gap_ci") or {}
        return ci.get("lo"), ci.get("hi")

    @property
    def gap_share(self) -> float | None:
        """Gap as a fraction of same-process recurrence."""
        return self.summary.get("gap_as_share_of_same_process")

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "summary": self.summary,
            "pairs": self.pairs,
            "warnings": self.warnings,
        }

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.pairs:
            if "estimates" not in p:
                continue
            row = {"window_1": p["window_1"], "window_2": p["window_2"],
                   "n_units": p["n_units"]}
            row.update(p["estimates"])
            ci = p.get("bootstrap", {}).get("gap") or {}
            row["gap_lo"], row["gap_hi"] = ci.get("lo"), ci.get("hi")
            rows.append(row)
        return pd.DataFrame(rows)

    def report(self) -> str:
        from .report import render

        return render(self)

    def __repr__(self) -> str:
        g, (lo, hi) = self.gap, self.gap_ci
        if g is None:
            return "<SwapResult: no window pair had enough units>"
        share = self.gap_share
        ci = f" [{lo:+.4f},{hi:+.4f}]" if lo is not None else ""
        pct = f", {share:.2%} of same-process recurrence" if share is not None else ""
        return f"<SwapResult gap={g:+.4f}{ci}{pct}, {len(self.pairs)} window pairs>"


def aggregate_records(
    records: pd.DataFrame,
    *,
    unit: str,
    window: str,
    process: str,
    outcome: str,
    how: str | Callable = "mean",
) -> pd.DataFrame:
    """Collapse raw observations to one value per unit x window x process.

    ``outcome`` is aggregated by ``how`` (any pandas aggregation, default the
    mean). A ``weight`` column counting the contributing records is added, which
    :func:`swap_test` uses for eligibility and for the sparsity diagnostic.
    """
    g = records.groupby([unit, window, process], observed=True, sort=False)
    out = g.agg(value=(outcome, how), weight=(outcome, "size")).reset_index()
    return out.rename(columns={"value": outcome})


def swap_test(
    data: pd.DataFrame,
    *,
    unit: str,
    window: str,
    process: str,
    value: str,
    weight: str | None = None,
    min_weight: float = 0.0,
    processes: Sequence[Any] | None = None,
    windows: Iterable[Any] | None = None,
    transform: str = "identity",
    n_boot: int = 2000,
    min_units: int = 30,
    sparsity_split: bool = True,
    seed: int = 0,
) -> SwapResult:
    """Run the observation-swap test.

    Parameters
    ----------
    data
        One row per ``unit`` x ``window`` x ``process`` with the measured
        ``value``. Use :func:`aggregate_records` to build this from raw records.
    unit, window, process, value
        Column names. ``process`` must have exactly two levels; it labels the
        observation process, and the test asks whether recurrence survives
        swapping it between windows.
    weight
        Optional column giving how much each cell rests on (a record count, say).
        Cells below ``min_weight`` are dropped, and it drives the sparsity
        diagnostic.
    min_weight
        Eligibility threshold applied to ``weight``.
    processes
        The two process levels, in order. Inferred and sorted if omitted.
    windows
        Window order. Inferred and sorted if omitted; pairs are taken in order.
    transform
        ``"identity"``, ``"logit"`` (for values on the unit interval), or
        ``"rank"``.
    n_boot
        Bootstrap replicates, resampling units.
    min_units
        Skip a window pair whose common cohort is smaller than this.
    sparsity_split
        Split each cohort at the median of its minimum cell weight and report the
        gap in each half. This is descriptive: cohort composition and precision
        can both change the contrast.
    seed
        Base seed. Each window pair uses a distinct derived seed.

    Returns
    -------
    SwapResult
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    fn = TRANSFORMS[transform]
    warnings: list[str] = []

    required = [unit, window, process, value] + ([weight] if weight else [])
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    df = data[required].copy()
    if processes is None:
        processes = sorted(df[process].dropna().unique())
    if len(processes) != 2:
        raise ValueError(
            f"process column {process!r} must have exactly two levels, got {list(processes)}"
        )
    pa, pb = processes
    df = df[df[process].isin([pa, pb])]

    if weight:
        before = len(df)
        df = df[df[weight] >= min_weight]
        if len(df) < before:
            warnings.append(
                f"dropped {before - len(df):,} of {before:,} cells below "
                f"min_weight={min_weight}"
            )
    else:
        df = df.assign(_w=1.0)
        weight = "_w"

    dup = df.duplicated([unit, window, process]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate unit/window/process rows; aggregate before calling"
        )

    if windows is None:
        windows = sorted(df[window].dropna().unique())
    windows = list(windows)
    if len(windows) < 2:
        raise ValueError("need at least two windows")

    wide = df.pivot_table(
        index=unit, columns=[window, process], values=[value, weight], aggfunc="first"
    )

    pairs: list[dict[str, Any]] = []
    for i, w1 in enumerate(windows):
        for w2 in windows[i + 1 :]:
            try:
                cols_v = [(value, w1, pa), (value, w1, pb), (value, w2, pa), (value, w2, pb)]
                cols_w = [(weight, w1, pa), (weight, w1, pb), (weight, w2, pa), (weight, w2, pb)]
                sub = wide.loc[:, cols_v + cols_w].dropna()
            except KeyError:
                pairs.append({"window_1": w1, "window_2": w2,
                              "skipped": "a process/window cell is absent"})
                continue
            n = len(sub)
            if n < min_units:
                pairs.append({"window_1": w1, "window_2": w2, "n_units": int(n),
                              "skipped": f"fewer than min_units={min_units}"})
                continue

            m = fn(sub.loc[:, cols_v].to_numpy(dtype=float))
            seed_bytes = repr((seed, str(w1), str(w2))).encode("utf-8")
            pair_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
            rng = np.random.default_rng(pair_seed)
            entry: dict[str, Any] = {
                "window_1": w1,
                "window_2": w2,
                "n_units": int(n),
                "estimates": _estimates(m),
                "bootstrap": _bootstrap(m, n_boot, rng),
                "median_weight_per_cell": {
                    c: float(sub.loc[:, cw].median())
                    for c, cw in zip(CELLS, cols_w)
                },
            }
            if sparsity_split:
                worst = sub.loc[:, cols_w].min(axis=1)
                cut = float(worst.median())
                halves = {}
                for name, mask in (("sparse", worst <= cut), ("dense", worst > cut)):
                    mm = fn(sub.loc[mask, cols_v].to_numpy(dtype=float))
                    if mm.shape[0] < min_units:
                        halves[name] = {"n_units": int(mm.shape[0]), "skipped": True}
                        continue
                    halves[name] = {
                        "n_units": int(mm.shape[0]),
                        "gap": _estimates(mm)["gap"],
                        "gap_ci": _bootstrap(mm, max(200, n_boot // 4), rng)["gap"],
                    }
                entry["sparsity"] = {"split_at": cut, **halves}
            pairs.append(entry)

    scored = [p for p in pairs if "estimates" in p]
    if not scored:
        warnings.append("no window pair had enough units")
        return SwapResult(pairs, {}, _design(locals()), warnings)

    def mean_of(key: str) -> float | None:
        v = [p["estimates"][key] for p in scored
             if np.isfinite(p["estimates"].get(key, np.nan))]
        return float(np.mean(v)) if v else None

    same = np.mean([
        0.5 * (p["estimates"]["r_same_process_A"] + p["estimates"]["r_same_process_B"])
        for p in scored
    ])
    gap_mean = mean_of("gap")
    gaps = np.array([p["estimates"]["gap"] for p in scored], dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    n_excl = sum(
        1 for p in scored
        if (c := p["bootstrap"].get("gap")) and c["lo"] is not None
        and (c["lo"] > 0 or c["hi"] < 0)
    )
    summary = {
        "n_window_pairs": len(scored),
        "n_units_median": float(np.median([p["n_units"] for p in scored])),
        **{k: mean_of(k) for k in (
            "r_same_window_w1", "r_same_process_A", "r_same_process_B",
            "r_swapped_A_to_B", "r_swapped_B_to_A", "gap_A", "gap_B", "gap",
            "disattenuated_swapped")},
        "gap_ci": {
            "lo": float(np.percentile(gaps, 2.5)) if gaps.size > 1 else None,
            "hi": float(np.percentile(gaps, 97.5)) if gaps.size > 1 else None,
            "basis": "spread of per-pair point estimates, not a pooled interval",
        },
        "gap_as_share_of_same_process": float(gap_mean / same) if same else None,
        "n_pairs_gap_interval_excludes_zero": n_excl,
    }
    if sparsity_split:
        sp = [p["sparsity"]["sparse"].get("gap") for p in scored
              if "sparsity" in p and "gap" in p["sparsity"].get("sparse", {})]
        dn = [p["sparsity"]["dense"].get("gap") for p in scored
              if "sparsity" in p and "gap" in p["sparsity"].get("dense", {})]
        summary["sparsity_gap_sparse_half"] = float(np.mean(sp)) if sp else None
        summary["sparsity_gap_dense_half"] = float(np.mean(dn)) if dn else None

    if summary["r_same_window_w1"] is not None and summary["r_same_window_w1"] < 0.5:
        warnings.append(
            f"same-window agreement between processes is low "
            f"({summary['r_same_window_w1']:.3f}); the two processes may not be "
            f"measuring the same quantity, which makes the swap hard to interpret"
        )
    return SwapResult(pairs, summary, _design(locals()), warnings)


def _design(ns: dict) -> dict[str, Any]:
    return {
        "unit": ns.get("unit"),
        "window": ns.get("window"),
        "process": ns.get("process"),
        "value": ns.get("value"),
        "processes": [str(x) for x in (ns.get("processes") or [])],
        "n_windows": len(ns.get("windows") or []),
        "min_weight": ns.get("min_weight"),
        "min_units": ns.get("min_units"),
        "transform": ns.get("transform"),
        "n_boot": ns.get("n_boot"),
        "seed": ns.get("seed"),
        "estimand": (
            "gap = r(same process across windows) - r(process swapped across "
            "windows), on a common cohort eligible in all four "
            "process-by-window cells"
        ),
    }
