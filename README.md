# observation-swap

**Is your cross-window recurrence in the units, or in how you observed them?**

You link records by an identifier, measure something per identifier in separated
time windows, and find the measure recurs. The usual reading is that the units
genuinely differ and those differences persist — individual repeatability,
behavioural type, a quenched trait.

There is a second reading that produces the same number. If each unit's
*observation conditions* persist — it sits in a well-covered cell, near a
receiver, in an easy detection geometry, on a route one sensor serves better —
then the same measurement bias is reproduced in both windows. Recurrence appears
with no contribution from the units at all.

Within one observation process the explanations can be confounded. This package
tests whether recurrence transfers across two labelled observation processes.

## The idea

Measure the statistic through one process in window 1 and through the **other**
process in window 2. Recurrence carried by independent process-specific biases can fall when the
process changes. Persistent unit differences and observation bias shared by
both processes can both survive the swap.

Holding window 1 fixed and changing only which process supplies window 2:

```
gap = r(same process across windows) − r(process swapped across windows)
```

- `gap ≈ 0` — recurrence transfers across these processes on this selected cohort.
- `gap` large — recurrence changes with the process; check bias, precision and
  comparability before attributing the difference to persistent observation effects.

Every correlation for a window pair is computed on **one common cohort**: units
eligible in all four process-by-window cells. Without that, the comparison
confounds sample composition with the effect under test.

## Install

```bash
pip install -e .          # numpy and pandas only
```

## Use

```python
import pandas as pd
from observation_swap import swap_test

# one row per unit x window x process
df = pd.read_csv("panel.csv")

result = swap_test(
    df,
    unit="animal_id",        # linked across windows
    window="season",         # the separated observation windows
    process="detector_array",# exactly two levels: the thing you swap
    value="activity_rate",   # the per-unit statistic
    weight="n_detections",   # optional: how much each cell rests on
    min_weight=10,
)
print(result.report())
print(result.gap, result.gap_share)
result.to_frame()            # per-window-pair table
```

From raw records rather than a prepared panel:

```python
from observation_swap import aggregate_records
panel = aggregate_records(records, unit="animal_id", window="season",
                          process="detector_array", outcome="was_active")
```

Command line:

```bash
observation-swap panel.csv --unit animal_id --window season \
    --process detector_array --value activity_rate --weight n --min-weight 10
```

## What counts as two observation processes

Anything the data label that records the same units through largely independent
machinery:

- two receiver networks (terrestrial and satellite; two volunteer networks)
- two detector arrays, camera-trap grids or acoustic sensors
- two survey protocols or observer teams
- two instruments with different detection geometry
- platform A vs platform B in an opportunistic data stream

What matters is that a unit's observation conditions under one are **not**
reproduced under the other. Two labels differing only cosmetically — the same
sensor writing two file formats — give a null gap for an uninteresting reason.
The report warns when within-window agreement between the processes is low,
which is the opposite failure: the two are not measuring the same quantity.

## Does it actually work?

Two simulated controls ship with the package and run in the test suite. Both
produce strong cross-window recurrence; they differ only in its source.

```python
from observation_swap import swap_test, simulate
swap_test(simulate.behavioural(), ...)    # recurrence from persistent unit traits
swap_test(simulate.observational(), ...)  # recurrence from persistent per-process conditions
```

| scenario | same-process r | swapped r | gap as share of recurrence |
|---|---|---|---|
| recurrence is behavioural | 0.889 | 0.890 | **−0.1%** |
| recurrence is observational | 0.890 | −0.006 | **100.7%** |

and it is monotone in between — holding the unit trait fixed and raising the
per-process condition variance gives shares of 0.0%, 4.7%, 17.6%, 45.9%, 75.2%.

These controls calibrate sensitivity to **independent process-specific** bias
under the additive simulation model. They do not cover shared observation bias
or unequal measurement precision. Purely shared observation bias gives a zero
gap even when all recurrence is observational.

## Reading the output

Alongside the gap you get:

- **Same-window agreement between processes.** How well the two measure the same
  thing. Used for a model-conditional independent-error correction
  (`disattenuated_swapped`), and a low value is flagged.
- **Per-pair bootstrap intervals**, resampling units.
- **A sparsity profile.** Each cohort is split at the median of its minimum cell
  weight. The split is descriptive: precision and cohort composition can both affect it.
  It does not test selection of units excluded from the common cohort.
- **`identity`, `logit` and `rank` transforms**, so you can check the result is
  not an artefact of scale.

## What this does not establish

Read these before quoting a gap.

- **Only processes the data distinguish are separated.** A condition common to
  both is untouched. In the originating study the two processes were two
  receiver networks. Shared reception effects and whether a vessel transmitted
  at all remain unresolved.
- **The common cohort is selected.** Units must be eligible under *both*
  processes. The artefact may act most strongly on the units this excludes.
  A near-zero gap is therefore weaker evidence against an observation-process
  explanation than a large gap is for one.
- **The relative gap is not generally a causal fraction.** Under an additive
  equal-variance model with independent process-specific biases, it measures the
  process-specific share of persistent covariance. Shared bias survives, and
  unequal precision or correlated biases can alter or cancel the contrast. A
  small combined gap cannot separately bound receiver and instrument effects.
- **Window pairs share units** and are not independent replicates.

## Origin

Built for an analysis of cross-window recurrence in coastal AIS vessel-tracking
records, where an intraclass correlation of 0.711 in a report-weighted mobility
measure could not be distinguished from persistent receiver coverage within a
single network. Applied to a public archive labelling terrestrial and satellite
reception, recurrence proved almost unchanged by the swap: a gap of 1.6% of the
same-network value, against about 100% in the independent process-specific
artefact control. This is a transfer result on the common cohort, not a bound
on all receiver-side effects or a calibration of another archive.

The problem is not specific to AIS. Repeatability and recurrence estimates from
identifier-linked tracking data are reported across movement ecology, human
mobility and sensor-network studies. The swap supplies a testable comparison
when suitable provenance labels and comparable measurements are available.

## Licence

MIT.
