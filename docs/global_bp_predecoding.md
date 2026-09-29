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

For a nonconverged shot, the global mechanism posterior is
`q[e] = expit(-posterior_llr[e])`, without a 0.5 cap. Mechanisms are projected by
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

**After this X/Z aggregation**, each projected DEM mechanism (possibly a
hyperedge) receives weight `w = -log(p)`. The existing probability-based
color decomposition receives the effective prior
`p_eff = 1/(1+exp(w)) = p/(1+p)`, so its log odds reproduce `w`.
This algebraic evaluation avoids infinities at p=0; effective priors are then
regularized to `[1e-14, 0.5]`, giving a finite maximum weight. p=1 gives zero
weight; p=0.5 gives log(2). Posteriors above 0.5 are no longer flattened.

The transformation occurs **before color/stage decomposition**, not on global
mechanisms and not independently on each final stage-1/stage-2 matching edge.
Only **stage 1** uses the resulting effective priors (version 4). This does
not reproduce official belief matching exactly:
our X/Z aggregation retains independent XOR, whereas its edge aggregation
uses addition. Effective priors are not physical posterior probabilities.

The original pre-BP X/Z DEM's physical probabilities are matched to the
projected source labels, including observables. Stage 2 is rebuilt from those
physical probabilities; its column sorting and source maps are rebuilt
together. Stage-1 graph ordering and posterior-derived weights are retained
exactly. Original physical probabilities are not capped or transformed.
Source-label or stage-1 alignment failures raise an explicit error.
The shared original DEM, priors and cached ordinary decoder are unchanged.

Ordinary, comparative, color-correlated, relifting, original-DEM perturbation
and native stage-1 perturbation follow this separation. **Final selection uses
original physical X/Z log odds** for every color/member/logical class. In BP
mode the effective options are `use_original_prior_for_stage2=True` and
`color_correlated_weight_basis="original_dem"`, even if ordinary-decoder
settings request otherwise. These overrides do not mutate the ColorCode's
ordinary decoder or its configured no-BP behavior.

Guide reweighting and original-DEM perturbation start from the posterior
stage-1 source priors, not physical source priors. Native perturbation starts
from the same decomposed posterior stage-1 edge probabilities as version 3.
Perturbed/guide priors never enter stage 2 or final selection. Color-correlated
ordinary baselines use BP stage 1 and physical stage 2 / original-DEM selection.
Guide-raised and perturbed BP priors are capped at 0.5. Native stage-1 uses
the PyMatching fork's explicit `clip_perturbed_probabilities=True` mode;
without BP, its existing rejection of possible negative weights is unchanged.
The usual native topology restrictions and incompatible strategy combinations
still apply. Supported SWIM uses physical stage-2 priors; its proxy
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
logical classes. Global-BP state and run logs now store version 4 and
`weight_rule="negative_log_xz_probability"`. The native perturbation law
remains scheme version 2; stage-1 probabilities and random draws are unchanged
from version 3. State/log metadata identifies `stage2_prior` and
`selection_prior` as `original_physical`.
Replay across split batches, worker reconstruction and converged-shot skips
is retained. Version-2 and version-3 decoder states are rejected explicitly:
both used posterior-derived stage-2 and selection priors, and version 2 also
capped the global posterior before projection. Historical
saved results remain historical results; they are not converted by this change.
No-BP original-DEM sampling is unchanged.

The first implementation rebuilds the matching DEM and decoder for each
nonconverged shot. It makes no runtime or logical-error improvement claim.
