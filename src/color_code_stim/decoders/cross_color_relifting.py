"""Candidate geometry and execution metadata for cross-color relifting."""

from dataclasses import dataclass

import numpy as np
from .baseline_equivalence import classify_baselines


COLORS = ("r", "g", "b")


@dataclass(frozen=True)
class ReliftSpec:
    target_color: str
    source_colors: tuple[str, ...] = ()

    @property
    def kind(self):
        return ("baseline" if not self.source_colors else
                "single_relift" if len(self.source_colors) == 1 else
                "all_color_relift")

    @property
    def label(self):
        if not self.source_colors:
            return self.target_color
        suffix = "[anchored]" if len(self.source_colors) == 2 else ""
        return f"{self.target_color}<-{','.join(self.source_colors)}{suffix}"


def candidate_specs():
    baseline = [ReliftSpec(c) for c in COLORS]
    singles = [ReliftSpec(c, (d,)) for c in COLORS for d in COLORS if d != c]
    all_color = [ReliftSpec(c, tuple(d for d in COLORS if d != c)) for c in COLORS]
    return tuple(baseline + singles + all_color)


def classify(baselines):
    """Classify exact mapped original-DEM corrections."""
    return classify_baselines(baselines)[0]


def projection(original, stage1_map):
    """Project an original correction through stage-1 provenance with parity."""
    return np.asarray((np.asarray(original, dtype=np.uint8) @ stage1_map.T) % 2,
                      dtype=bool)


def anchored(anchor, first, second):
    return np.asarray(anchor, dtype=bool) ^ np.asarray(first, dtype=bool) ^ np.asarray(second, dtype=bool)


def stage2_syndrome(detectors, stage1, color, detector_ids_by_color):
    physical = np.zeros_like(detectors, dtype=bool)
    physical[detector_ids_by_color[color]] = detectors[detector_ids_by_color[color]]
    return np.concatenate((physical, np.asarray(stage1, dtype=bool)))


def stage1_valid(stage1, H, detectors):
    predicted = np.asarray((np.asarray(stage1, dtype=np.uint8) @ H.T) % 2, dtype=bool)
    kept_checks = np.asarray(H.tocsr().getnnz(axis=1) > 0)
    return np.array_equal(predicted[kept_checks], np.asarray(detectors, dtype=bool)[kept_checks])


def alias_for_class_one(spec, baselines, specs):
    """Return canonical earlier slot for a redundant class-one hypothesis."""
    target = spec.target_color
    if not spec.source_colors:
        return None
    _, groups = classify_baselines(baselines)
    representative = {color: group[0] for group in groups for color in group}
    distinct_sources = []
    for source in spec.source_colors:
        if representative[source] == representative[target]:
            continue
        if representative[source] not in distinct_sources:
            distinct_sources.append(representative[source])
    if spec.kind == "all_color_relift":
        if len(distinct_sources) == 0 or len(spec.source_colors) == 2 and np.array_equal(
            baselines[spec.source_colors[0]], baselines[spec.source_colors[1]]
        ):
            return COLORS.index(target)
    if not distinct_sources:
        return COLORS.index(target)
    canonical = next(i for i, candidate in enumerate(specs)
                     if candidate.target_color == target and
                     candidate.source_colors == (distinct_sources[0],))
    if spec.kind == "all_color_relift" or spec.source_colors != (distinct_sources[0],):
        return canonical
    return None
