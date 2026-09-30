"""Requested per-shot experiment statistics, without candidate exports.

This layer observes existing candidate generation; it never changes priors,
candidate order, matching or the meaning of the comparative/SWIM scores.
"""

import numpy as np


METRIC_NAMES = frozenset({
    "weights", "logical_error", "default_logical_error",
    "better_weight_by_color_correlated_decoding",
    "effect_by_color_correlated_decoding", "color_correlated_run",
    "relift_run", "logical_gap", "swim_distance",
})
_BASELINE_ERRORS = {"default_logical_error", "effect_by_color_correlated_decoding"}
_ERRORS = _BASELINE_ERRORS | {"logical_error"}


def _observables(values, shots, count, name):
    values = np.asarray(values, dtype=bool)
    if count == 1 and values.shape == (shots,):
        values = values[:, None]
    if values.shape != (shots, count):
        raise ValueError(f"{name} must have shape ({shots}, {count}) or ({shots},) for one observable")
    return values


class ExperimentMetrics:
    """Retain winning corrections and, when requested, two SWIM minima/shot."""

    def __init__(self, names, manager, shots, comparative, *,
                 actual_observables=None, baseline_predictions=None,
                 candidate_scorer=None, check_validity=False):
        if isinstance(names, str):
            raise ValueError("metrics must be a sequence of metric names")
        self.names = tuple(names)
        if (any(not isinstance(name, str) for name in self.names) or
                len(set(self.names)) != len(self.names) or set(self.names) - METRIC_NAMES):
            raise ValueError(f"Unknown or duplicate metrics: {self.names}")
        self.requested = set(self.names)
        self.manager, self.shots = manager, shots
        self.num_obs = manager.obs_matrix.shape[0]
        self.comparative = comparative
        self.actual = (_observables(actual_observables, shots, self.num_obs, "actual_observables")
                       if self.requested & _ERRORS else None)
        self.baseline = (_observables(baseline_predictions, shots, self.num_obs, "baseline_predictions")
                         if baseline_predictions is not None else None)
        self.need_baseline = bool(self.requested & _BASELINE_ERRORS) and self.baseline is None
        self.indices = np.arange(shots)
        self.selected = (np.zeros((shots, manager.H.shape[1]), dtype=bool)
                         if not comparative or check_validity else None)
        if self.selected is not None:
            self.best_weight = np.full(shots, np.inf)
            self.best_rank = np.full(shots, np.iinfo(np.int64).max, dtype=np.int64)
        if self.need_baseline:
            self.baseline_weight = np.full(shots, np.inf)
            self.baseline_rank = np.full(shots, np.iinfo(np.int64).max, dtype=np.int64)
            self.baseline_selected = (np.zeros((shots, manager.H.shape[1]), dtype=bool)
                                      if not comparative else None)
        self.candidate_scorer = candidate_scorer
        self.swim = None
        if "swim_distance" in self.requested:
            if self.num_obs != 1 or comparative or candidate_scorer is None:
                raise ValueError("swim_distance metrics require a supported one-observable non-comparative scorer")
            self.swim = np.full((2, shots), np.inf)

    def wants(self, name):
        return name in self.requested

    def observe(self, logical_class, slot, candidate_count, shot_slice,
                correction, weight, generation_weight, *, detectors=None,
                hypothesis=None, color=None, swim_scores=None):
        need_baseline = self.need_baseline and slot < 3
        if self.selected is None and not need_baseline and self.swim is None:
            return
        indices = self.indices[shot_slice]
        if self.selected is not None or self.swim is not None or (
                need_baseline and self.baseline_selected is not None):
            correction = np.asarray(correction, dtype=bool).reshape(len(indices), -1)
        # Stochastic ensembles visit one shot at a time. Scalar comparisons
        # avoid allocating several indexing/mask arrays for every candidate.
        # Comparative output already follows the final class argmin and does
        # not need to retain any winning correction unless validity is checked.
        if len(indices) == 1:
            shot = indices[0]
            if self.selected is not None:
                score = float(np.asarray(weight).flat[0])
                rank = logical_class * candidate_count + slot
                if (score < self.best_weight[shot] or
                        (score == self.best_weight[shot] and rank < self.best_rank[shot])):
                    self.best_weight[shot], self.best_rank[shot] = score, rank
                    self.selected[shot] = correction[0]
            if need_baseline:
                generation = float(np.asarray(generation_weight).flat[0])
                rank = logical_class * 3 + slot
                if (generation < self.baseline_weight[shot] or
                        (generation == self.baseline_weight[shot] and rank < self.baseline_rank[shot])):
                    self.baseline_weight[shot], self.baseline_rank[shot] = generation, rank
                    if self.baseline_selected is not None:
                        self.baseline_selected[shot] = correction[0]
        else:
            if self.selected is not None:
                weight = np.asarray(weight, dtype=float).reshape(len(indices))
                rank = logical_class * candidate_count + slot
                update = ((weight < self.best_weight[indices]) |
                          ((weight == self.best_weight[indices]) & (rank < self.best_rank[indices])))
                if np.any(update):
                    active = indices[update]
                    self.best_weight[active] = weight[update]
                    self.best_rank[active] = rank
                    self.selected[active] = correction[update]
            if need_baseline:
                generation = np.asarray(generation_weight, dtype=float).reshape(len(indices))
                rank = logical_class * 3 + slot
                update = ((generation < self.baseline_weight[indices]) |
                          ((generation == self.baseline_weight[indices]) &
                           (rank < self.baseline_rank[indices])))
                if np.any(update):
                    active = indices[update]
                    self.baseline_weight[active] = generation[update]
                    self.baseline_rank[active] = rank
                    if self.baseline_selected is not None:
                        self.baseline_selected[active] = correction[update]
        if self.swim is not None:
            scores = np.asarray(self.candidate_scorer(detectors, hypothesis, color)
                                if swim_scores is None else swim_scores, dtype=float)
            if scores.shape != (len(indices),) or not np.isfinite(scores).all() or np.any(scores < 0):
                raise ValueError("Invalid per-candidate SWIM scores")
            parity = np.asarray((correction.astype(np.uint8) @ self.manager.obs_matrix.T) % 2,
                                dtype=np.uint8).ravel()
            self.swim[parity, indices] = np.minimum(self.swim[parity, indices], scores)
            return scores

    def finish(self, prediction, weights, candidate_weights, gaps,
               logical_classes, *, run_category=None):
        predicted = _observables(prediction, self.shots, self.num_obs, "prediction")
        result = {}
        if self.requested & _ERRORS:
            failure = np.any(predicted != self.actual, axis=1)
        if "logical_error" in self.requested:
            result["logical_error"] = failure
        if self.requested & _BASELINE_ERRORS:
            baseline = self.baseline
            if baseline is None:
                if self.comparative:
                    baseline = np.asarray(logical_classes, dtype=bool)[self.baseline_rank // 3]
                else:
                    baseline = np.asarray((self.baseline_selected.astype(np.uint8) @
                                           self.manager.obs_matrix.T) % 2, dtype=bool)
            default_failure = np.any(baseline != self.actual, axis=1)
            if "default_logical_error" in self.requested:
                result["default_logical_error"] = default_failure
            if "effect_by_color_correlated_decoding" in self.requested:
                result["effect_by_color_correlated_decoding"] = (default_failure & ~failure).astype(np.uint8)
        if "weights" in self.requested:
            result["weights"] = weights
        if "better_weight_by_color_correlated_decoding" in self.requested:
            if (not np.isfinite(candidate_weights[:, :3, :]).all() or
                    np.isnan(candidate_weights).any() or np.isneginf(candidate_weights).any() or
                    not np.isfinite(weights).all()):
                raise ValueError("Expected finite ordinary/selected weights and finite/+inf candidate weights")
            ordinary = np.min(candidate_weights[:, :3, :], axis=(0, 1))
            result["better_weight_by_color_correlated_decoding"] = (weights < ordinary).astype(np.uint8)
        if "logical_gap" in self.requested:
            if gaps is None and self.shots:
                raise ValueError("logical_gap metrics require all comparative logical classes")
            result["logical_gap"] = gaps if gaps is not None else np.empty(0, dtype=float)
            if not np.isfinite(result["logical_gap"]).all() or np.any(result["logical_gap"] < 0):
                raise ValueError("Invalid comparative logical gap")
        for name in ("color_correlated_run", "relift_run"):
            if name in self.requested:
                category = np.asarray(run_category)
                if self.shots and (not np.issubdtype(category.dtype, np.integer) or
                                   np.any((category < 0) | (category > 2))):
                    raise ValueError(f"{name} must contain 0, 1 or 2 per shot")
                result[name] = category.astype(np.uint8)
        if "swim_distance" in self.requested:
            result["swim_distance"] = self.swim[predicted[:, 0].astype(np.uint8), self.indices]
            if not np.isfinite(result["swim_distance"]).all():
                raise RuntimeError("No finite same-logical-class SWIM candidate")
        if any(values.shape != (self.shots,) for values in result.values()):
            raise RuntimeError("Experiment metric must have one scalar per shot")
        return {name: result[name] for name in self.names}
