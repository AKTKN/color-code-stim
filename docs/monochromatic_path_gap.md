# Final-correction monochromatic path gap

This is a decoder-independent heuristic for the final physical X correction.
It works with ordinary upstream PyMatching; it needs no SWIM backend or growth
data. The hard decoder and its existing output dictionary are unchanged.

## Definition and scope

For each color c in `(r,g,b)`, construct the monochromatic lattice of
[Lee, Li and Bartlett, Definition 2 and Appendix A.3](https://arxiv.org/html/2404.07482v2).
Keep the opposite corner `s_c` and the boundary missing c checks `t_c` distinct.
Suppress each degree-two primal c-edge vertex. A remaining edge has immutable
physical support `Q_c(a)` of size two; the corner edge has size one.
The topology is built from physical Tanner incidence and colored primal links,
not any decoder pairing. Stable qubit IDs survive vertex permutations.
Parallel edges keep distinct supports. Self-loops are retained; since all
residual costs are nonnegative, traversing a loop cannot improve a shortest path.

For the **same final physical correction E** on all three graphs:

```text
w_q = log((1-p_q)/p_q)
W(E) = sum_{q in E} w_q
omega_c^E(a) = sum_{q in Q_c(a) minus E} w_q
D_c(E) = shortest_distance(s_c, t_c; omega_c^E)
phi_c(E) = D_c(E) - W(E)
phi(E) = min_c phi_c(E)
```

A pair edge has cost `w1+w2`, `w1`, `w2`, or `0`. The entire original
correction weight is subtracted once. Negative scores are preserved.
For a path support L the identity is

```text
W(L minus E) - W(E)
  = [W(E symmetric_difference L) - W(E)] - W(E minus L).
```

Thus corrected qubits outside the path lower this score. The search uses only
the monochromatic path family. This is **not an established exact logical gap,
posterior log-odds, or universal bound on the full logical gap**. The paper
supplies the lattice construction, not this heuristic or its performance claim.

The `ColorCodePathGap` adapter supports only odd d >= 3 standard triangular
6.6.6 patches, `circuit_type="tri"`, `cnot_schedule="tri_optimal"`,
`superdense_circuit=False`, `rounds=1`, Z memory, and `NoiseModel(bitflip=p)`
with `0<p<1/2`. The stored resolved schedule must equal `tri_optimal`; its
identical `LLB` alias/explicit list also resolves to that schedule.
Every other noise parameter, including effective overrides, must be zero.
`perfect_first_syndrome_extraction=True` suppresses the needed error layer and
is rejected. The adapter audits the actual circuit for exactly one X-error
layer on every data qubit. Noiseless initialization/final flags are otherwise
permitted. Unsupported configurations raise `ValueError`.

## Minimal usage

```python
from color_code_stim import ColorCode, NoiseModel
from color_code_stim.metrics import ColorCodePathGap

code = ColorCode(d=5, rounds=1, noise_model=NoiseModel(bitflip=0.03))
metric = ColorCodePathGap(code)  # topology and mapping cached once
dets, actual_obs = code.circuit.compile_detector_sampler(seed=123).sample(
    100, separate_observables=True
)
predicted_obs, extra = code.decode(dets, full_output=True)
result = metric.evaluate(extra)
print(result.phi, result.distance_by_color, result.correction_weight)
failures = predicted_obs != actual_obs[:, 0]  # analysis only, not metric input
```

The same adapter works with `comparative_decoding=True`. Call it after decode
has returned, including when an enabled predecoder has composed a partial
correction. It reads only `extra["error_preds"]`; `best_colors`, matching
`weights`, stage-1 predictions, and candidate metadata do not select E.
Build a new adapter if the circuit/configuration changes.

For a supplied physical correction and explicit nonuniform nonnegative weights:

```python
import numpy as np
from color_code_stim.metrics import (
    MonochromaticPathGap, build_monochromatic_topologies,
)
qubit_ids, topologies = build_monochromatic_topologies(code.tanner_graph)
evaluator = MonochromaticPathGap(topologies, np.ones(len(qubit_ids)))
physical_E = np.zeros(len(qubit_ids), dtype=bool)
result = evaluator.evaluate(physical_E, diagnostics=True)
physical_paths = [tuple(qubit_ids[q] for q in path)
                  for path in result.paths_by_color]
```

This direct core is independent of the correction's decoder and accepts finite
nonnegative weights (including zero). It does not extend the adapter's noise
scope. `PathTopology` can also specify synthetic multigraph fixtures.

## Ordering, mapping and shapes

Canonical physical order is ascending Tanner `qid` for data vertices
(`pauli=None`), exposed as `adapter.qubit_ids`. A qid is a stable physical
identifier, not a position in a shortened/permuted igraph vertex sequence.
Graph supports/witnesses contain **positions** in this canonical order.

Decoder `error_preds` instead uses error-only columns of `dem_xz`. Under the
validated pure-X model, DemManager's depolarizing separation is the identity,
so the circuit DEM has precisely these columns. `DemQubitMap` flattens repeats
and shifts, ignores non-error instructions when incrementing columns, and
matches each full detector/logical signature against independently constructed
physical face incidence and the physical Z-logical support. First-round Z
detectors carry H, final noiseless difference detectors are zero, and the
comparative observable detector carries the same logical support as L0.
A missing, duplicated, or unrecognized signature raises `ValueError`.

The legacy `errors_to_qubits` enumerates raw instructions and happens to work
when all errors precede annotations. This implementation does not reuse that
implicit assumption. The generic mapping tests interspersed annotations,
detector shifts, logical parity and nontrivial column permutations. A direct
physical array must go to `evaluator.evaluate`, never be passed as DEM columns
to the adapter merely because its width matches.

Input shapes are `(n,)` and `(shots,n)`. Outputs for a single input are scalar
`phi`, `correction_weight`, `minimizing_color`, and `(3,)` `distance_by_color`
and `phi_by_color`. Batch outputs have shapes `(shots,)` and `(shots,3)`;
`(0,n)` consistently returns empty outputs. Decode's single-shot extra output
is itself a one-row batch, so its adapter result has a length-one batch axis.
`color_order` is always `("r","g","b")`; exact minimum ties choose r, then g,
then b. Shortest-path witness ties follow igraph and need not be unique.

Optional `diagnostics=True` returns `paths_by_color[color]` for a single input
or `[shot][color]` for a batch. Default batches store no witnesses.
Inputs, original weights, graph topology, circuit, and decoder state are not
modified. Original weights use read-only storage; residuals are fresh per shot.

igraph's nonnegative Dijkstra search is used with one source and one target,
not dense all-pairs distances. For these bounded-degree physical graphs the
core takes O(n log n) per shot and O(n) working memory, plus input and O(shots)
result storage. Binary validation is per row. Mapping a batch produces an
O(shots*n) physical array; callers can chunk large batches. Witness storage
can add O(shots*n). Physical topology setup sorts stable IDs, O(n log n),
with O(n) graph storage. Circuit/DEM construction and hard decoding have
separate costs; no negligible-overhead claim is made.

## Reproducible assessment

```bash
python examples/evaluate_path_gap.py --smoke --output /tmp/path-gap-smoke
python examples/evaluate_path_gap.py --distances 3 5 7 9 \
  --probabilities .01 .03 .05 --shots 10000 --seed 20260918 \
  --output /tmp/path-gap-larger
```

Only the smoke run is part of acceptance. d=3 enumerates all 128 physical
errors with probability `p^|F| (1-p)^(n-|F|)`; other distances sample independent
physical X errors using NumPy's recorded seed, then convert to decoder
detector columns. This is equivalent to the audited single data-error layer.
Both decoder modes reuse the same physical errors but are **separate cohorts**
with their own final corrections and failure labels. Each cohort compares
`phi`, `D_min`, and `minus_W`; comparative mode additionally reports the
**decoder-derived** comparative gap. Scores never receive true errors/labels.

For d=3, an independent enumeration over **all physical supports**, not just
paths, computes class minima `m_b(s)` and records:

- `delta_class = m_(1-b(E))(s) - m_b(E)(s)` (signed);
- `delta_corr = m_(1-b(E))(s) - W(E)` (signed);
- `within_class_suboptimality = W(E) - m_b(E)(s)`;
- `optimal_class_gap = abs(m_1(s)-m_0(s))` (nonnegative).

None is a posterior class log-odds. `shots.csv` stores these diagnostics and
per-color distances, W, phi, final corrections, hard predictions, failures,
and stable IDs; exact summaries weight patterns by physical probability.
`postselection.csv` retains whole ties at every raw threshold, score >= threshold.
`conditional.csv` gives failure versus score and the score probability masses.
`equal_retention.csv` compares common target retained fractions: MC uses
label-independent shot-ID ties (and records achieved fractions); exact analysis
randomly accepts an entire boundary score group with one common probability.
The latter is an expected-retention curve, never a true-error-dependent choice
among patterns sharing a score. Primary plotted curves use whole thresholds.

Monte Carlo rates have 95% Wilson intervals, including nonzero upper limits
when zero failures are observed. Exhaustive rates use probability sums without
binomial intervals. Plots use linear axes, retaining zero rates and negative
scores. `assessment.png` shows postselection, conditional rates and score
distributions; `exact_diagnostics.png` compares phi to all four exact quantities.
No confidence-interval band for a paired difference is inferred from separate
binomial intervals. Source hash, base/implementation commits, configuration,
versions and separated timings are saved in `metadata.json`/`summary.json`.

To compare legacy SWIM externally, export its score with the **same saved
physical_error rows**, `shot_id`, d, p, decoder mode and final correction.
Join one-to-one on cohort plus `shot_id`, verifying the physical-error hash,
final correction and failure label match before plotting. Matching only seed
or row count across different samplers is insufficient. No legacy backend is
required or imported by this implementation.
