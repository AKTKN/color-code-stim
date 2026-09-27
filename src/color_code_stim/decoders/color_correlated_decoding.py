"""Original-DEM guide reweighting for color-guided concatenated matching.

Guide corrections identify mechanisms of the pre-decomposition X/Z DEM. For
candidate generation, raise their priors to the power 1/b for stage 1.
Stage 2 and candidate selection use the unmodified original prior.
"""

from dataclasses import dataclass
from collections import OrderedDict
import math

import numpy as np
import stim

from ..dem_utils.dem_decomp import DemDecomp
from .baseline_equivalence import classify_baselines


COLORS = ("r", "g", "b")
MATCHING_EPS = 1e-14


class OriginalDemProbabilityBuilder:
    """Replace probabilities in the common X/Z DEM without changing its sources."""

    def __init__(self, dem_manager):
        self.base_dem = dem_manager.dem_xz.flattened()
        self.base_q = np.asarray(dem_manager.probs_xz, dtype=float).copy()
        if sum(inst.type == "error" for inst in self.base_dem) != len(self.base_q):
            raise ValueError("original DEM source probabilities are misaligned")
        for color in COLORS:
            if not np.array_equal(dem_manager.dems_decomposed[color].org_prob, self.base_q):
                raise ValueError("color decomposition has different original DEM source ordering")

    def build(self, probabilities: np.ndarray) -> stim.DetectorErrorModel:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != self.base_q.shape:
            raise ValueError("probabilities must use original DEM mechanism ordering")
        dem = stim.DetectorErrorModel()
        source = 0
        for inst in self.base_dem:
            if inst.type == "error":
                dem.append("error", float(probabilities[source]), inst.targets_copy())
                source += 1
            else:
                dem.append(inst)
        return dem


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


def candidate_schedule(baseline: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Classify three mapped corrections and choose only necessary reruns.

    The first matching color in r/g/b order represents a duplicated guide.
    Returned indices address candidate_specs(), including its first three
    ordinary candidates. Categories are 0: all equal, 1: one equal pair,
    and 2: all distinct.
    """
    category, groups = classify_baselines(baseline)
    if category == 0:
        return 0, ()
    if category == 2:
        return 2, tuple(range(3, 12))
    duplicate = next(group for group in groups if len(group) == 2)
    singleton = COLORS.index(next(group[0] for group in groups if len(group) == 1))
    representative = COLORS.index(duplicate[0])
    specs = candidate_specs()
    chosen = []
    for target in range(3):
        guide = representative if target == singleton else singleton
        chosen.append(next(i for i, spec in enumerate(specs)
                           if spec.target_color == COLORS[target]
                           and spec.guide_colors == (COLORS[guide],)))
    return 1, tuple(sorted(chosen))


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


class CandidateEvaluator:
    """Map and score candidates using one unmodified base-prior basis."""

    def __init__(self, dem_manager, weight_basis: str):
        if weight_basis not in ("stage2", "original_dem"):
            raise ValueError("weight_basis must be 'stage2' or 'original_dem'")
        self._manager = dem_manager
        self.weight_basis = weight_basis
        self._source_count = len(dem_manager.probs_xz)
        self._base_maps = {}
        self._stage2_llr = {}
        for color in COLORS:
            base = dem_manager.dems_decomposed[color]
            mapping = base.error_map_matrices[1].tocsr()
            if np.all(np.diff(mapping.indptr) == 1) and len(set(mapping.indices)) == len(mapping.indices):
                self._base_maps[color] = mapping.indices.copy()
            else:
                self._base_maps[color] = None
            self._stage2_llr[color] = np.log((1 - base.probs[1]) / base.probs[1])
        self._original_llr = None
        if weight_basis == "original_dem":
            q = np.asarray(dem_manager.probs_xz, dtype=float)
            if (np.any(q <= 0) or np.any(q >= 1) or not np.isfinite(q).all()):
                raise ValueError("original DEM probabilities must be finite and in (0, 1)")
            self._original_llr = np.log((1 - q) / q)

    def evaluate(self, color: str, native: np.ndarray,
                 generation_weight: np.ndarray | float,
                 temporary: DemDecomp | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray | float,
                            np.ndarray | float]:
        """Return mapped, base-native, selection and diagnostic weights."""
        base = self._manager.dems_decomposed[color]
        if temporary is None:
            base_native = np.asarray(native, dtype=bool).copy()
            indices = self._base_maps[color]
            if indices is None:
                mapped = base.map_errors_to_org_dem(base_native, stage=2)
            else:
                mapped = np.zeros(base_native.shape[:-1] + (self._source_count,), dtype=bool)
                mapped[..., indices] = base_native
        else:
            mapped, base_native = align_stage2_to_base(native, temporary, base)
        if self._original_llr is None:
            selection_weight = base_native.astype(float) @ self._stage2_llr[color]
        else:
            selection_weight = mapped.astype(float) @ self._original_llr
        return mapped, base_native, selection_weight, generation_weight


class ColorCorrelatedPriorReweighter:
    """Cache stage-1 priors from fixed symbolic source maps; keep stage 2 fixed."""

    def __init__(self, dem_manager, b: float, cache_size: int = 128):
        if type(b) not in (int, float) or not math.isfinite(b) or b <= 0:
            raise ValueError("color_correlated_b must be positive and finite")
        self._manager = dem_manager
        self._builder = OriginalDemProbabilityBuilder(dem_manager)
        self._base_q = self._builder.base_q
        self.b = float(b)
        self._raised_q = np.minimum(self._base_q ** (1 / self.b), 1 - MATCHING_EPS)
        self._cache_size = cache_size
        self._cache = OrderedDict()
        self._stage1 = {}
        for color in COLORS:
            base = dem_manager.dems_decomposed[color]
            ems = base.dems_symbolic[0]._ems
            rows_by_source = [[] for _ in self._base_q]
            for row, em in enumerate(ems):
                for source in np.unique(em.prob_vars):
                    rows_by_source[int(source)].append(row)
            self._stage1[color] = (ems, rows_by_source, base.probs[0])

    def source_probabilities(self, guide_sources: np.ndarray) -> np.ndarray:
        """Return the guide-reweighted original-DEM source probabilities."""
        guide_sources = np.asarray(guide_sources, dtype=bool)
        if guide_sources.shape != self._base_q.shape:
            raise ValueError("Guide must be in original DEM mechanism ordering")
        updated = self._base_q.copy()
        updated[guide_sources] = self._raised_q[guide_sources]
        return updated

    def stage1_probabilities(self, color: str, guide_sources: np.ndarray) -> np.ndarray:
        """Recompute only stage-1 columns touched by guide sources."""
        if color not in self._stage1:
            raise ValueError(f"Unknown target color: {color}")
        guide_sources = np.asarray(guide_sources, dtype=bool)
        if guide_sources.shape != self._base_q.shape:
            raise ValueError("Guide must be in original DEM mechanism ordering")
        key = (color, guide_sources.tobytes())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ems, rows_by_source, base_p = self._stage1[color]
        updated = base_p.copy()
        if np.any(guide_sources):
            q = self.source_probabilities(guide_sources)
            affected = set()
            for source in np.flatnonzero(guide_sources):
                affected.update(rows_by_source[int(source)])
            for row in affected:
                em = ems[row]
                updated[row] = (1 - np.prod(1 - 2 * em.prob_muls * q[em.prob_vars])) / 2
        self._cache[key] = updated
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return updated

    def reweighted_dem(self, guide_sources: np.ndarray) -> stim.DetectorErrorModel:
        """Replace only error probabilities, preserving source and detector order."""
        updated = self.source_probabilities(guide_sources)
        return self._builder.build(updated)
