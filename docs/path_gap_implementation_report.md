# Implementation and acceptance report

Implemented on `feature/final-correction-path-gap` from origin/main
`0eb35935c1e5ff30ba3db9def30a9d35bca2f16d`. Core/API/tests/documentation commit:
`05ccbe4a367a3e7fb91c9721b92cb270b314fd0c`. The smoke source includes the rate-plot
scale adjustment in `c147e60`; its full SHA is saved in
[`examples/path_gap_smoke/metadata.json`](../examples/path_gap_smoke/metadata.json).
The later artifact/report commit does not alter the implementation.

## Public API and integration

```python
from color_code_stim.metrics import ColorCodePathGap
metric = ColorCodePathGap(code)
predictions, extra = code.decode(detectors, full_output=True)
result = metric.evaluate(extra)
```

`MonochromaticPathGap`, `PathTopology`, `build_monochromatic_topologies`, and
`DemQubitMap` expose the independent physical API and separable mapping.
See [the complete usage/order/scope guide](monochromatic_path_gap.md).
No decoder entry point or candidate-selection code was edited. The final
`error_preds`, after class/color selection and any partial-correction assembly,
is mapped to ascending physical qids and used identically in every color.
W is recomputed from that physical correction and immutable original weights.
Raw DEM instruction positions, error-only columns, absolute detector IDs,
Tanner vertex indices and canonical physical positions remain distinct.

Changed files: `src/color_code_stim/metrics/{__init__,monochromatic_path_gap}.py`,
`tests/test_monochromatic_path_gap.py`, `tests/test_path_gap_evaluation.py`,
`examples/evaluate_path_gap.py`, the small `examples/path_gap_smoke/` output,
this report and the API guide, plus README/MkDocs navigation.
The implementation requires no modified PyMatching methods.

## Validation

Acceptance environment: a temporary `--system-site-packages` venv created with
the `color_code_so` Python, installing the upstream PyMatching 2.3.1 wheel
locally and this worktree editable. Existing Conda packages/checkouts were not
replaced. Python 3.12; NumPy 1.26.4, Stim 1.16.0, igraph 1.0.0, statsmodels
0.15.0. Full versions and source hashes are in the saved metadata.

Commands from the feature worktree:

```bash
/tmp/color-code-path-gap-env/bin/python -m pytest -q
# 99 passed, 2 skipped (28.86 s)
/tmp/color-code-path-gap-env/bin/python -m pytest \
  tests/test_monochromatic_path_gap.py tests/test_path_gap_evaluation.py -q
# 57 passed (5.13 s)
/tmp/color-code-path-gap-env/bin/python examples/evaluate_path_gap.py \
  --smoke --output examples/path_gap_smoke
git diff --check
```

All existing 42 tests pass; the two skips already exclude unsupported
comparative rectangular-stability decoding. No pre-existing failures appeared,
so an unmodified-base failure comparison was unnecessary. New checks include:

- Independent Steane face-incidence construction and DFS enumeration of all
  simple paths, every one of 128 masks, all colors, uniform and unequal/zero
  weights. The oracle calls neither production builder nor shortest paths.
- Independent d=5 integer-simplex incidence/path enumeration, plus d=3,5,7
  physical H/logical-parity checks, uniform analytic bounds, full and empty E.
- Pair/singleton costs, a corrected qubit outside the witness, negative scores,
  parallel edges, disconnected graphs, stable IDs under vertex permutations.
- Actual ordinary/comparative candidate disagreements and another-color final
  selections; final-only metadata invariance; a composed XOR fixture and the
  real comparative erasure/partial-predecoder path.
- Error-only DEM permutations with interspersed non-error instructions and
  detector shifts; independent physical syndrome/logical matrix checks;
  unchanged hard predictions, failure labels, state and caller arrays.
- Single/batch/empty/order/cache invariance, readonly weights, all unsupported
  noise fields including overrides, malformed weights/corrections/topology.
- Exact probability weighting, label-independent randomized ties, signed class
  versus correction gaps and within-class cost, zero-failure Wilson bounds.

Separate same-agent review (not external peer review) inspected the final
source/diff and the saved plots. It caught and corrected an exact-enumeration
tie-order bias: equal scores now receive equal randomized acceptance regardless
of which underlying physical pattern was enumerated first. MC ties use shot
IDs established independently of errors/labels. The final plot-only adjustment
was exercised by regenerating the bounded smoke output.

## Smoke observations and limitations

Seed 20260918, p=.03, ordinary/comparative as separate cohorts. Each d=3 cohort
enumerates 128 patterns with exact physical probabilities; each d=5 cohort
uses the same 256 sampled physical errors. The entire saved output is under
1 MiB, including four score/rate figures, two exact-diagnostic figures, CSV
tables and configuration/timing manifests.

Both d=3 cohorts have probability-weighted logical failure
`0.016418097822240005`. At expected retained fraction .9, randomized-score-tie
postselection gives `0.008830330913600021`; at .5 it gives
`0.0002070398532044594`. All three ablations (and the available comparative gap)
give the same values at these points for this small code. These exact values
are floating-point probability sums, not equal-weight pattern counts.

Both d=5 cohorts have 1/256 observed failures (0.00390625); the 95% Wilson
interval is approximately `[0.000690, 0.021791]`. At 230/256 retained shots,
each score has zero observed failures with upper limit about 0.01643.
This does not establish zero conditional risk or statistically resolved
superiority of any score. The smoke's decoder corrections produced no negative
phi values; negative scores remain covered by explicit fixtures and are never
clipped or filtered from evaluation.

All comparisons reuse identical final corrections and failure labels within a
cohort. d=3 diagnostics report signed Delta_class and Delta_corr, within-class
suboptimality, and nonnegative optimal-class gap separately. They are not
posterior odds. Larger CLI grids were not run. The saved timing manifest
separates physical code setup, topology setup, mapping/circuit audit, decoder
setup, decoding, final-correction mapping and metric evaluation; these tiny-run
timings are descriptive and do not establish asymptotic/production overhead.

The original SWIM worktrees and commits are unchanged. This feature does not
prove a universal full-gap bound or extend the adapter to circuit-level noise.
