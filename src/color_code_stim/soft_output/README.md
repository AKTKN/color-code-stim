# Phase-2A fixed-branch swim output

Requires the modified PyMatching fork exposing `SoftOutputConfig` and
`Matching.decode_batch_with_soft_output`. Ordinary decoding remains available
with upstream PyMatching; SO is opt-in.

```python
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel

code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=0.05))
detectors, actual = code.sample(64, seed=20260912)
predicted, extra = code.decode(
    detectors, full_output=True, compute_swim_distance=True)
# extra['color_order'] == ('r', 'g', 'b')
# stage2_weights_by_color and swim_distances_by_color: (64, 3)
# selected_swim_distance: (64,)
```

Only single-round triangular Z-memory with data-only bit-flip noise is
accepted. Comparative decoding, BP/custom DEMs, and circuit-level swim
are explicitly refused. Source metadata preserves detector coordinates
including time and virtual-row provenance for future work.

The backend uses final defect-center Sparse Blossom radii to define metric
balls on the resolved labelled graph. The exact residual geometry and
physical logical witnesses have independent small-code tests. These radii
are not an exported nonnegative optimal odd-cut certificate, so the
certified representative-gap bound is not asserted. Selected swim is an
exploratory per-branch proxy, not a full-decoder logical gap or LLR.

The original H2 contains zero padding rows from other colors. Metadata marks
these `INACTIVE_PADDING`; all active rows are physical-c or stage-1 virtual
constraints. Only the analysis graph omits inactive rows. Hard decoding
retains the original H2, weights, and stage-2 input unchanged.
