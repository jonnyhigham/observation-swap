"""Human-readable rendering of a SwapResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SwapResult


def _f(x, spec="+.4f"):
    return "n/a" if x is None else format(x, spec)


def render(result: "SwapResult") -> str:
    d, s = result.design, result.summary
    pa, pb = (d.get("processes") or ["A", "B"])[:2]
    out: list[str] = []
    out.append("Observation-swap test")
    out.append("=" * 60)
    out.append(f"unit={d['unit']!r}  window={d['window']!r}  "
               f"process={d['process']!r} ({pa} / {pb})  value={d['value']!r}")
    out.append(f"transform={d['transform']}  min_weight={d['min_weight']}  "
               f"bootstrap={d['n_boot']} replicates resampling units")
    out.append("")

    if not s:
        out.append("No window pair had enough units; nothing estimated.")
        for w in result.warnings:
            out.append(f"  warning: {w}")
        return "\n".join(out)

    out.append(f"{s['n_window_pairs']} window pairs, "
               f"median common cohort {s['n_units_median']:.0f} units")
    out.append("")
    out.append("Agreement and recurrence (mean over window pairs)")
    out.append(f"  same window, {pa} vs {pb}      {_f(s['r_same_window_w1'], '.4f')}"
               "   (how closely the two processes agree)")
    out.append(f"  across windows, {pa} then {pa}   {_f(s['r_same_process_A'], '.4f')}")
    out.append(f"  across windows, {pb} then {pb}   {_f(s['r_same_process_B'], '.4f')}")
    out.append(f"  across windows, {pa} then {pb}   {_f(s['r_swapped_A_to_B'], '.4f')}"
               "   <- process swapped")
    out.append(f"  across windows, {pb} then {pa}   {_f(s['r_swapped_B_to_A'], '.4f')}"
               "   <- process swapped")
    out.append("")

    gap = s["gap"]
    lo, hi = result.gap_ci
    share = s.get("gap_as_share_of_same_process")
    out.append(f"  GAP  {_f(gap)}"
               + (f"   (joint 95% interval {_f(lo)} to {_f(hi)})" if lo is not None else ""))
    if share is not None:
        out.append(f"       {share:.2%} of same-process recurrence")
    out.append(f"       interval excludes zero in "
               f"{s['n_pairs_gap_interval_excludes_zero']}/{s['n_window_pairs']} pairs")
    if s.get("disattenuated_swapped") is not None:
        out.append(f"       swapped recurrence under independent-error correction: "
                   f"{s['disattenuated_swapped']:.4f}")
    out.append("")

    if s.get("covariance_gap") is not None:
        out.append(f"  Difference covariance / 2  {_f(s['covariance_gap'])}")
        out.append(f"  Difference recurrence      {_f(s.get('difference_recurrence'), '.4f')}")
        out.append("  Covariance uses squared outcome units; compare only on a meaningful common scale.")
        pooled = s.get("pooled_bootstrap") or {}
        if pooled:
            out.append("  Joint whole-unit bootstrap intervals for mean contrasts:")
            for key in ("gap", "covariance_gap", "difference_recurrence"):
                ci = pooled.get(key) or {}
                out.append(f"    {key}: {_f(ci.get('lo'))} to {_f(ci.get('hi'))} "
                           f"({ci.get('n_draws', 0)} valid draws)")
        out.append("")

    sp, dn = s.get("sparsity_gap_sparse_half"), s.get("sparsity_gap_dense_half")
    if sp is not None and dn is not None:
        out.append(f"Sparsity profile   sparse half {_f(sp)}   dense half {_f(dn)}")
        if sp > dn:
            out.append("  Larger in the sparse half; composition and precision can both "
                       "affect this contrast.")
        else:
            out.append("  Not larger in the sparse half. Coverage bias and selection "
                       "are not ruled out.")
        out.append("")

    out.append("Reading")
    out.append("  The gap describes how recurrence changes when the process is swapped.")
    out.append("  No equivalence margin is assumed. Report mean differences and differential")
    out.append("  recurrence as well as transfer; a small correlation contrast can coexist")
    out.append("  with a persistent calibration change.")
    out.append("  Shared observation bias can survive the swap. Unequal precision and")
    out.append("  correlated process biases can also change the gap. This is not generally")
    out.append("  a causal attributable fraction or an upper bound on observation bias.")
    out.append("")

    out.append("What this does not establish")
    out.append("  Only processes the data distinguish are separated. A condition "
               "common to both -- a unit that is absent from, or")
    out.append("  misreported by, every process alike -- is untouched by this test.")
    out.append("  The common cohort requires eligibility under both processes, so it "
               "is selected; the artefact may act most")
    out.append("  strongly on the units it excludes. A near-zero gap is therefore "
               "weaker evidence against an observation-process")
    out.append("  explanation than a large gap is for one.")

    for w in result.warnings:
        out.append(f"\n  warning: {w}")
    return "\n".join(out)
