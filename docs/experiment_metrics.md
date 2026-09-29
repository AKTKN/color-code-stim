# Per-shot experiment metrics

`ColorCode.decode(..., metrics=(...), full_output=False)` returns
`(predictions, metrics_dict)`. Each requested metric is a NumPy array of shape
`(shots,)`. Only requested fields are returned. Ordinary prediction-only calls
and the existing `full_output=True` diagnostic contract are unchanged.

```python
detectors, actual = code.sample(100, seed=73)
prediction, metrics = code.decode(
    detectors,
    metrics=("logical_error", "weights", "logical_gap"),
    actual_observables=actual,
)
```

This example requires `comparative_decoding=True` for `logical_gap`.

| Name | dtype | Meaning |
|---|---|---|
| `weights` | float64 | Selected weight in the configured comparison basis |
| `logical_error` | bool | Any predicted observable differs from the actual observables |
| `default_logical_error` | bool | The paired ordinary baseline fails |
| `better_weight_by_color_correlated_decoding` | uint8 | Selected weight is strictly smaller than the minimum over the first three ordinary candidates, across logical classes |
| `effect_by_color_correlated_decoding` | uint8 | The baseline fails and the final decision succeeds |
| `color_correlated_run` | uint8 | Selected class's ordinary correction multiplicity category, 0/1/2 |
| `relift_run` | uint8 | Corresponding relifting category, 0/1/2 |
| `logical_gap` | float64 | Existing comparative gap between the two smallest logical-class minima |
| `swim_distance` | float64 | Existing minimum SWIM score among generated candidates with the final hard decision's observable parity |

Error statistics require `actual_observables`, shaped `(shots,)` for one
observable or `(shots, observables)` otherwise. Actual outcomes enter only
post-decoding statistics, never candidate generation or selection.

Color-correlated baseline error statistics require `baseline_predictions`
from the paired ordinary decoder. This preserves the simulation's ordinary
selection policy, including configurations where its default comparison
basis differs from color-correlated selection. Perturbation and relifting
reuse their existing ordinary candidates and native generation-weight
baseline selection. Member 0, strict improvement, candidate ordering, exact
ties and comparative semantics are unchanged.

For spatial data-only SWIM, supply `compute_swim_distance=True` and request
`swim_distance`. For supported closed circuit-memory SWIM, the simulation
supplies `candidate_scorer(detectors, stage1_hypothesis, color)`, returning a
float64 array `(candidate_shots,)` from its unchanged-prior geometry backend.
The decoder reduces these scores by correction parity as candidates are
generated. The score retains its existing proxy interpretation; it is not
a full logical gap, posterior LLR or newly certified bound.

Candidate correction/hypothesis exports and generation-weight diagnostics
are not created for metrics mode. Winning corrections, ordinary corrections
needed for guides/relifts, and native stage-1 batch results remain working
data. Decoder working memory is therefore not limited to the returned scalar
arrays. `check_validity=True` checks the winning correction without retaining
all candidates for diagnostics.

`metrics=None` retains the old API; even `metrics=()` explicitly requests the
tuple form with an empty metric dictionary. Metrics cannot be combined with
`full_output=True`, `return_candidate_data=True`, BP/custom priors, erasure
predecoding or cultivation postselection. Unsupported names/modes and
invalid observable shapes fail explicitly. Empty batches return empty scalar
arrays without advancing perturbation state. Native absolute shot offsets
and advancing RNG/save/load behavior remain unchanged.
