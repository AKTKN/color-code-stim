# Global DEM BP predecoding

The public switches remain `bp_predecoding=True` and `bp_prms={...}`:

```python
from color_code_stim import ColorCode

code = ColorCode(d=3, rounds=3, temp_bdry_type="Z", p_circuit=.003)
detectors, actual = code.sample(20, seed=73)
prediction, output = code.decode(
    detectors, bp_predecoding=True, bp_prms={"max_iter": 10},
    metrics=("logical_error", "weights"), actual_observables=actual,
)
converged = output["bp_converged"]
```

Install `ldpc` with `pip install 'color-code-stim[bp]'`. BP runs on
`circuit.detector_error_model(decompose_errors=False, flatten_loops=True)`
before preparing the separated or color-decomposed matching DEM. For
comparative decoding, logical hypothesis detector rows are excluded from the
BP checks; the global mechanism columns and observable matrix are retained.
BP receives syndromes explicitly, including when the check matrix is square.

For a converged shot, the returned prediction is the global correction times
the global observable matrix modulo two. Its correction is checked against
the BP syndrome. Noise separation and concatenated matching are skipped.

For a nonconverged shot, the global mechanism prior becomes
`q[e] = min(expit(-posterior_llr[e]), 0.5)`. Mechanisms are projected by
detector Pauli metadata, directly from this global mechanism space. Logical
labels go to the X detector sector for `temp_bdry_type="X"`, and the Z
detector sector for `"Z"`. Y memory raises `NotImplementedError` before any
shots. Cultivation postselection is also unsupported.

Global mechanisms with identical projected detector **and observable**
targets are combined using their parity probability:

`p = (1 - product(1 - 2*q[e])) / 2`.

The implementation evaluates this law with `log1p`/`expm1` for small priors.
This gives exact sector marginals of the independent mechanism surrogate;
it does not reconstruct the joint BP posterior or retain X/Z correlations.
The global and projected spaces have separate source IDs and matrices.
Logical labels that differ never collide in the projection.

Projected matcher probabilities are regularized to `[1e-14, 0.5]` after
contraction. Matching uses `log((1-p)/p)`, including exactly zero weight at
p=0.5. Color decomposition is rebuilt for this shot. Probability-dependent
stage-2 column sorting and correction source maps are rebuilt together;
original-DEM perturbations also retain their existing symbolic ordering maps.
The shared original DEM, priors and cached ordinary decoder are unchanged.

Ordinary, comparative, color-correlated, relifting, original-DEM perturbation
and native stage-1 perturbation use this posterior DEM as their base. Final
candidate scoring retains the configured comparison basis within that DEM.
Color-correlated ordinary baselines are computed in the same posterior DEM.
Guide-raised and perturbed BP priors are capped at 0.5. Native stage-1 uses
the PyMatching fork's explicit `clip_perturbed_probabilities=True` mode;
without BP, its existing rejection of possible negative weights is unchanged.
The usual native topology restrictions and incompatible strategy combinations
still apply. Supported SWIM uses the shot's posterior base priors; its proxy
interpretation is unchanged.

Hard predictions are returned for every shot. When metrics or full output
are requested, `bp_converged` is a bool array `(shots,)`. Every concatenated
metric, including `logical_error`, is a masked array whose mask equals
`bp_converged`. With full output, `concat_outputs[i]` is the ordinary
diagnostic dictionary for a fallback shot and `None` for a converged shot;
common fields such as `weights` and `error_preds` also have aligned masks.
Metrics may be combined with full output in BP mode for verification.
Corrections in these diagnostic records use the projected X/Z mechanism
space, while standalone `decode_bp` returns global-space corrections.

`bp_shot_offset` optionally supplies an absolute physical-shot index. Skipped
shots retain their positions. A supplied native `perturbation_shot_offset`
must agree with it. Without an offset the BP cursor advances across calls;
save/load preserves the resolved uint64 seed and cursor. Native draws retain
their existing color streams and absolute shot seed; capped weights identify
scheme version 2. BP original-DEM perturbation uses NumPy generators seeded
with `SeedSequence([resolved_seed, absolute_shot])`, shared across colors and
logical classes. Its law/version is stored as global-BP version 2, permitting
replay across split batches, worker reconstruction and converged-shot skips.
No-BP original-DEM sampling is unchanged.

The first implementation rebuilds the matching DEM and decoder for each
nonconverged shot. It makes no runtime or logical-error improvement claim.
