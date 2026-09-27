"""Source-provenance priors for color-correlated concatenated matching.

Guide corrections are Boolean sets of *original* DEM mechanisms. Reweighting
changes only candidate generation; candidates must be compared using the
unmodified stage-2 prior of the decode call.
"""

from dataclasses import dataclass

import numpy as np


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


class ColorCorrelatedPriorReweighter:
    """Cache aligned source sets and construct temporary stage-1/2 priors.

    The conditional rule requires unit symbolic multipliers. A decomposition
    that splits a source among multiple terms is rejected explicitly.
    """

    def __init__(self, dem_manager):
        self._sources = {}
        self._base = {}
        self._org_prob = {}
        for color in COLORS:
            decomp = dem_manager.dems_decomposed[color]
            q = np.asarray(decomp.org_prob, dtype=float)
            self._org_prob[color] = q
            for stage in (0, 1):
                symbolic = decomp.dems_symbolic[stage]
                for em in symbolic:
                    if not np.all(np.asarray(em.prob_muls) == 1):
                        raise NotImplementedError(
                            "Color-correlated decoding requires unit prob_muls "
                            f"(color={color}, stage={stage + 1})"
                        )
                mapping = decomp.error_map_matrices[stage].tocsr()
                base = np.asarray(decomp.probs[stage], dtype=float)
                if mapping.shape != (len(base), len(q)):
                    raise ValueError("Decomposition provenance is not aligned with H columns")
                sources = tuple(mapping.indices[mapping.indptr[j]:mapping.indptr[j + 1]]
                                for j in range(mapping.shape[0]))
                marginal = np.array([
                    (1 - np.prod(1 - 2 * q[indices])) / 2 for indices in sources
                ])
                if not np.allclose(marginal, base, rtol=1e-9, atol=1e-12):
                    raise NotImplementedError(
                        "Color-correlated decoding requires unit-multiplicity "
                        f"provenance consistent with the base prior ({color}, stage {stage + 1})"
                    )
                self._sources[color, stage] = sources
                self._base[color, stage] = base

    def probabilities(self, color: str, guide_sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return independent temporary priors for both stages for one guide set."""
        guide_sources = np.asarray(guide_sources, dtype=bool)
        if guide_sources.shape != self._org_prob[color].shape:
            raise ValueError("Guide must be in original DEM mechanism ordering")
        q = self._org_prob[color]
        result = []
        for stage in (0, 1):
            updated = self._base[color, stage].copy()
            for column, sources in enumerate(self._sources[color, stage]):
                for source in sources:
                    if guide_sources[source]:
                        other = sources[sources != source]
                        conditional = (1 + np.prod(1 - 2 * q[other])) / 2
                        updated[column] = max(updated[column], conditional)
            result.append(np.clip(updated, MATCHING_EPS, 1 - MATCHING_EPS))
        return result[0], result[1]
