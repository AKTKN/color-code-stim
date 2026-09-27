"""Original-DEM conditioning for color-guided concatenated matching.

Guide corrections identify mechanisms of the pre-decomposition X/Z DEM. For
candidate generation, condition those Bernoulli mechanisms on being active,
rebuild that DEM, then decompose it for the target color. Candidate selection
always uses an unmodified base prior.
"""

from dataclasses import dataclass

import numpy as np
import stim

from ..dem_utils.dem_decomp import DemDecomp


COLORS = ("r", "g", "b")
MATCHING_EPS = 1e-14


@dataclass(frozen=True)
class CandidateSpec:
    target_color: str
    guide_colors: tuple[str, ...]

    @property
    def label(self) -> str:
        if not self.guide_colors:
            return self.target_color
        return f"{self.target_color}<-{' OR '.join(self.guide_colors)}"


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return three baselines, then three guide choices per target color."""
    baseline = tuple(CandidateSpec(c, ()) for c in COLORS)
    correlated = []
    for color in COLORS:
        others = tuple(c for c in COLORS if c != color)
        correlated.extend(CandidateSpec(color, guide) for guide in
                          ((others[0],), (others[1],), others))
    return baseline + tuple(correlated)


def guide_union(baseline_corrections: dict[str, np.ndarray],
                guide_colors: tuple[str, ...]) -> np.ndarray:
    """Combine original-DEM guide mechanisms with OR, never parity/XOR."""
    return np.logical_or.reduce([baseline_corrections[c] for c in guide_colors])


def align_stage2_to_base(native: np.ndarray, temporary: DemDecomp,
                         base: DemDecomp) -> tuple[np.ndarray, np.ndarray]:
    """Map a rebuilt stage-2 correction to original and base-stage-2 order."""
    temporary_map = temporary.error_map_matrices[1].tocsr()
    base_map = base.error_map_matrices[1].tocsr()
    if (not np.all(np.diff(temporary_map.indptr) == 1)
            or not np.all(np.diff(base_map.indptr) == 1)
            or set(temporary_map.indices) != set(base_map.indices)):
        raise NotImplementedError("Reweighted stage-2 source set differs from base DEM")
    mapped = np.asarray(temporary.map_errors_to_org_dem(native, stage=2), dtype=bool)
    base_native = mapped[..., base_map.indices]
    return mapped, base_native


class ColorCorrelatedPriorReweighter:
    """Rebuild target-color decompositions after conditioning original sources.

    A selected guide source has conditional Bernoulli probability one. It is
    clipped just below one so the downstream matching weights remain finite.
    The unselected original source probabilities are unchanged.
    """

    def __init__(self, dem_manager):
        self._manager = dem_manager
        self._base_dem = dem_manager.dem_xz.flattened()
        self._base_q = np.asarray(dem_manager.probs_xz, dtype=float).copy()
        if sum(inst.type == "error" for inst in self._base_dem) != len(self._base_q):
            raise ValueError("original DEM source probabilities are misaligned")
        for color in COLORS:
            decomp = dem_manager.dems_decomposed[color]
            if not np.array_equal(decomp.org_prob, self._base_q):
                raise ValueError("color decomposition has different original DEM source ordering")

    def source_probabilities(self, guide_sources: np.ndarray) -> np.ndarray:
        """Return the conditioned original-DEM source probabilities."""
        guide_sources = np.asarray(guide_sources, dtype=bool)
        if guide_sources.shape != self._base_q.shape:
            raise ValueError("Guide must be in original DEM mechanism ordering")
        updated = self._base_q.copy()
        updated[guide_sources] = 1 - MATCHING_EPS
        return updated

    def reweighted_dem(self, guide_sources: np.ndarray) -> stim.DetectorErrorModel:
        """Replace only error probabilities, preserving source and detector order."""
        updated = self.source_probabilities(guide_sources)
        dem = stim.DetectorErrorModel()
        source = 0
        for inst in self._base_dem:
            if inst.type == "error":
                dem.append("error", float(updated[source]), inst.targets_copy())
                source += 1
            else:
                dem.append(inst)
        return dem

    def decomposition(self, color: str, guide_sources: np.ndarray) -> DemDecomp:
        """Re-decompose the reweighted original DEM for one target color."""
        if color not in COLORS:
            raise ValueError(f"Unknown target color: {color}")
        return DemDecomp(
            org_dem=self.reweighted_dem(guide_sources),
            color=color,
            remove_non_edge_like_errors=self._manager.remove_non_edge_like_errors,
        )
