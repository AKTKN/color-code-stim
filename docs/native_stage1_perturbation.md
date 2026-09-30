# Native stage-1 perturbation

`stage1_perturbation=False` retains the original-X/Z-DEM perturbation workflow,
including its shared cross-colour draws and both stage-2 prior options.
`stage1_perturbation=True` selects a different candidate generation rule:
perturb the original decomposed stage-1 edge probabilities independently by
shot, member and colour, inside PyMatching.

```python
from color_code_stim import ColorCode, NoiseModel

code = ColorCode(
    d=5, rounds=1, noise_model=NoiseModel(bitflip=0.03),
    stage1_perturbation=True,
    perturbation_ensemble_size=12, perturbation_alpha=1.0,
    perturbation_seed=20260929,
    color_correlated_weight_basis="original_dem",
)
shots, actual = code.sample(10, seed=17)
predictions, extra = code.decode(shots, full_output=True)
```

M includes unchanged member 0. Nonbaseline priors use
`clip(p1 * (1 + alpha * Uniform(-1, 1)), 1e-14, 1-1e-14)`.
Alpha=0 preserves the exact original probabilities. M=1 and alpha=0 reproduce
the existing perturbation workflow's outputs with original stage-2 priors,
including its configured final selection basis and batched floating arithmetic.

Native mode automatically sets the effective `enable_prior_perturbation=True`
and `use_original_prior_for_stage2=True`. All candidates use the original
stage-2 decomposition/weights and cached stage-2 Matching. Final correction
mapping, common-prior scoring, candidate order (`m0:r/g/b`, then `m1:r/g/b`, ...),
ties, baseline predictions and supported comparative/SWIM output retain the
existing contract. Stage-1 candidates are generated in native batches; stage 2
and final selection remain in this package.

Colour streams are r=0, g=1, b=2. Each physical shot's colour-specific draws are
shared across comparative logical classes. An explicit seed plus absolute shot
position determines the native stream (scheme version 1). This is not the old
NumPy stream, and native stochastic candidates need not equal original-DEM
candidates. The two modes differ in both prior construction and cross-colour
correlations; runtime comparisons do not establish a logical-error improvement.

Repeated `decode` calls advance a physical-shot cursor. To replay or schedule
shots independently, pass `perturbation_shot_offset=<absolute shot index>`.
Empty batches consume no positions. `ColorCode.save/load` retains the effective
seed, stream version and cursor, rebuilding native objects lazily. Legacy saves
without the new option load with `stage1_perturbation=False`.

Install/build the native PyMatching fork on
`codex/native-stage1-perturbation-20260929`; ordinary PyMatching remains usable
for `stage1_perturbation=False`. Missing native support produces an actionable
error. Native stage 1 supports only simple graphlike columns with finite,
nonnegative log odds; potential negative perturbations (`p1*(1+alpha)>0.5`)
and parallel columns are rejected. Perturbation remains incompatible with
guide/relifting and erasure predecoding modes.

Global BP predecoding is supported on `codex/global-bp-predecoding-20260929`.
For a nonconverged BP shot, this shot's posterior CSS DEM is the base prior;
stage 2 and final scoring use that prior. Native nonbaseline stage-1 priors
are capped at 0.5 after perturbation, using PyMatching's explicit
`clip_perturbed_probabilities=True` mode (scheme version 2). Member 0 and
random draws are unchanged. Absolute shot IDs include converged BP shots.
See [global BP](global_bp_predecoding.md) for its output and memory scope.
