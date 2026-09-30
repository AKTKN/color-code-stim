"""
Concatenated Matching Decoder for Color Code

This module implements the concatenated minimum-weight perfect matching (MWPM) decoder
for color codes. It performs two-stage decoding with color-based decomposition,
supporting comparative decoding and advanced pre-decoding strategies.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pymatching

from .base import BaseDecoder
from .color_correlated_decoding import (
    CandidateEvaluator, ColorCorrelatedPriorReweighter, candidate_schedule,
    candidate_specs, guide_union,
)
from . import cross_color_relifting as relift
from .prior_perturbation import PriorPerturbationEnsemble
from .native_stage1_perturbation import NativeStage1Ensemble, resolve_native_seed
from .experiment_metrics import ExperimentMetrics
from .matching_cache import MatchingCache, StagePrior, StageStructure
from ..config import COLOR_LABEL, color_to_color_val
from ..dem_utils.dem_manager import DemManager
from ..utils import _get_final_predictions


def _ordinary_baseline_predictions(mapped, generation_weights, manager,
                                   logical_classes, comparative_decoding, num_obs):
    """Select the already solved ordinary r/g/b candidates by native MWPM weight."""
    shots = mapped.shape[2]
    if not shots:
        return np.zeros((0,) if num_obs == 1 else (0, num_obs), dtype=bool)
    baseline_class, baseline_color, _, _ = _get_final_predictions(
        generation_weights[:, :3, :]
    )
    if comparative_decoding:
        observed = np.asarray([logical_classes[i] for i in baseline_class], dtype=bool)
    else:
        selected = mapped[baseline_class, baseline_color, np.arange(shots)]
        observed = np.asarray(
            (selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool
        )
    return observed.ravel() if num_obs == 1 else observed


class ConcatMatchingDecoder(BaseDecoder):
    """
    Concatenated minimum-weight perfect matching decoder for color codes.

    This decoder implements the sophisticated concatenated decoding strategy where
    each color is decoded in two stages, and the results are combined to find the
    minimum-weight error correction. It supports comparative decoding for magic
    state distillation and various pre-decoding strategies.

    Key Features:
    - Two-stage MWPM decoding per color (stage 1: local errors, stage 2: global errors)
    - Comparative decoding: test all logical classes, return minimum weight
    - Logical gap calculation for post-selection in magic state distillation
    - Erasure matcher pre-decoding for improved performance
    - BP pre-decoding integration (when available)
    - Color-specific decoding with flexible color selection

    Attributes
    ----------
    dem_manager : DEMManager
        Manager for detector error models and decompositions
    circuit_type : str
        Type of circuit being decoded
    num_obs : int
        Number of observables
    comparative_decoding : bool
        Whether comparative decoding is enabled
    enable_colorcorrelated_decoding : bool
        Whether to generate nine additional source-guided candidates. Each
        candidate is selected using the unmodified stage-2 prior.
    """

    def __init__(
        self,
        dem_manager: DemManager,
        enable_colorcorrelated_decoding: bool = False,
        color_correlated_weight_basis: str | None = None,
        color_correlated_b: float = 1.0,
        enable_cross_color_relifting: bool = False,
        enable_prior_perturbation: bool = False,
        perturbation_ensemble_size: int = 1,
        perturbation_alpha: float = 0.0,
        perturbation_seed: int | None = None,
        use_original_prior_for_stage2: bool = False,
        stage1_perturbation: bool = False,
    ):
        """
        Initialize the concatenated matching decoder.

        Parameters
        ----------
        dem_manager : DEMManager
            Manager providing access to decomposed DEMs and matrices
        enable_colorcorrelated_decoding : bool, default False
            Run three baseline and nine color-correlated candidates per logical
            class. BP/custom DEM and matching-growth swim combinations are not
            supported.
        """
        self._swim_backends = {}
        self.dem_manager = dem_manager
        self.circuit_type = dem_manager.circuit_type
        self.num_obs = dem_manager.circuit.num_observables
        self.comparative_decoding = dem_manager.comparative_decoding
        self.enable_colorcorrelated_decoding = enable_colorcorrelated_decoding
        if type(color_correlated_b) not in (int, float) or not np.isfinite(color_correlated_b) or color_correlated_b <= 0:
            raise ValueError("color_correlated_b must be positive and finite")
        self.color_correlated_b = float(color_correlated_b)
        self._color_correlated_reweighter = None
        self._matching_cache = MatchingCache()
        self._candidate_evaluators = {}
        self.enable_cross_color_relifting = enable_cross_color_relifting
        if type(stage1_perturbation) is not bool:
            raise ValueError("stage1_perturbation must be boolean")
        self.stage1_perturbation = stage1_perturbation
        enable_prior_perturbation = enable_prior_perturbation or stage1_perturbation
        self.enable_prior_perturbation = enable_prior_perturbation
        if not isinstance(perturbation_ensemble_size, int) or perturbation_ensemble_size < 1:
            raise ValueError("perturbation_ensemble_size must be >= 1")
        if not np.isfinite(perturbation_alpha) or not 0 <= perturbation_alpha <= 1:
            raise ValueError("perturbation_alpha must be between 0 and 1")
        self.perturbation_ensemble_size = perturbation_ensemble_size
        self.perturbation_alpha = perturbation_alpha
        self.perturbation_seed = resolve_native_seed(perturbation_seed) if stage1_perturbation else perturbation_seed
        if type(use_original_prior_for_stage2) is not bool:
            raise ValueError("use_original_prior_for_stage2 must be boolean")
        self.use_original_prior_for_stage2 = use_original_prior_for_stage2 or stage1_perturbation
        if enable_prior_perturbation and (enable_cross_color_relifting or enable_colorcorrelated_decoding):
            raise NotImplementedError("Prior perturbation cannot be combined with cross-color relifting or color-correlated decoding")
        self._perturbation_ensemble = None
        if enable_cross_color_relifting and enable_colorcorrelated_decoding:
            raise NotImplementedError("Cross-color relifting and color-correlated decoding cannot be combined")
        if enable_cross_color_relifting and dem_manager.remove_non_edge_like_errors:
            raise NotImplementedError("Cross-color relifting requires remove_non_edge_like_errors=False")
        if color_correlated_weight_basis is None:
            color_correlated_weight_basis = "original_dem" if enable_colorcorrelated_decoding else "stage2"
        if color_correlated_weight_basis not in ("stage2", "original_dem"):
            raise ValueError("color_correlated_weight_basis must be 'stage2' or 'original_dem'")
        if enable_colorcorrelated_decoding and color_correlated_weight_basis != "original_dem":
            raise ValueError("color-correlated decoding requires original_dem selection basis")
        self.color_correlated_weight_basis = color_correlated_weight_basis

    def supports_comparative_decoding(self) -> bool:
        """Return True - this decoder supports comparative decoding."""
        return True

    def supports_predecoding(self) -> bool:
        """Return True - this decoder supports pre-decoding strategies."""
        return True

    def decode(
        self,
        detector_outcomes: np.ndarray,
        colors: Union[str, List[str]] = "all",
        logical_value: Union[bool, Sequence[bool], None] = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        full_output: bool = False,
        check_validity: bool = False,
        verbose: bool = False,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
        compute_swim_distance: bool = False,
        return_candidate_data: bool = False,
        perturbation_shot_offset: int | None = None,
        metrics: Sequence[str] | None = None,
        actual_observables: np.ndarray | None = None,
        baseline_predictions: np.ndarray | None = None,
        candidate_scorer=None,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Decode detector outcomes using concatenated MWPM decoding.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            1D or 2D array of detector measurement outcomes.
            If 1D, interpreted as a single sample.
            If 2D, each row is a sample, each column a detector.
        colors : str or list of str, default 'all'
            Colors to use for decoding. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        logical_value : bool or sequence of bool, optional
            Logical value(s) to use for decoding. If None and comparative_decoding
            is True, all possible logical value combinations will be tested.
        erasure_matcher_predecoding : bool, default False
            Whether to use erasure matcher as a pre-decoding step.
        partial_correction_by_predecoding : bool, default False
            Whether to apply partial correction from erasure matcher predecoding.
        full_output : bool, default False
            Whether to return extra information about the decoding process.
        check_validity : bool, default False
            Whether to check the validity of predicted error patterns.
        verbose : bool, default False
            Whether to print additional information during decoding.
        custom_dem_data : dict, optional
            Custom DEM matrices and probabilities for BP predecoding.
            Format: {color: ((H1, p1), (H2, p2))} where H1,H2 are parity check
            matrices and p1,p2 are probability arrays for stages 1 and 2.
        compute_swim_distance : bool, default False
            Spatial matching-growth soft output for single-round triangular
            data-only X noise. Ensemble hypotheses are scored on the
            unmodified base-prior stage-2 graph.
        return_candidate_data : bool, default False
            Export generated candidate hypotheses and corrections for an
            external scorer without changing hard decoding.
        metrics : sequence of str, optional
            Return (predictions, requested_metrics) with one scalar per shot
            and metric, without full candidate exports. Requires full_output
            and return_candidate_data to be False. Error metrics require
            actual_observables. See docs/experiment_metrics.md.
        **kwargs
            Additional parameters (for compatibility).

        Returns
        -------
        np.ndarray or tuple
            If metrics is specified: (predictions, requested_metrics_dict).
            Otherwise full_output=False returns predicted observables and
            full_output=True returns (predictions, extra_outputs_dict).
        """
        if perturbation_shot_offset is not None and not self.stage1_perturbation:
            raise ValueError("perturbation_shot_offset requires stage1_perturbation=True")
        if compute_swim_distance:
            if custom_dem_data is not None:
                raise NotImplementedError("Swim output with custom_dem_data is not supported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding or self.comparative_decoding:
                raise NotImplementedError("Swim output is not validated for predecoding or comparative DEMs")
            if not self.dem_manager.swim_data_only:
                raise NotImplementedError("Swim requires single-round triangular data-only X noise; unresolved boundaries are UNCLASSIFIED")

        metric_state = None
        if metrics is not None:
            if full_output or return_candidate_data:
                raise ValueError("metrics requires full_output=False and return_candidate_data=False")
            if custom_dem_data is not None or erasure_matcher_predecoding or partial_correction_by_predecoding:
                raise NotImplementedError("metrics does not support custom priors or erasure predecoding")
            if self.circuit_type == "cult+growing":
                raise NotImplementedError("metrics does not support cultivation postselection")
            shots = np.asarray(detector_outcomes)
            shot_count = 1 if shots.ndim == 1 else len(shots)
            if perturbation_shot_offset is not None and (
                    type(perturbation_shot_offset) is not int or
                    not 0 <= perturbation_shot_offset < 2**64 or
                    shot_count > 2**64 - 1 - perturbation_shot_offset):
                raise ValueError("perturbation_shot_offset and batch length must fit uint64")
            if compute_swim_distance and candidate_scorer is None:
                candidate_scorer = lambda det, hyp, color: self._decode_stage2(
                    det, hyp, color, compute_swim_distance=True).swim_distances
            metric_state = ExperimentMetrics(
                metrics, self.dem_manager, shot_count, self.comparative_decoding,
                actual_observables=actual_observables, baseline_predictions=baseline_predictions,
                candidate_scorer=candidate_scorer, check_validity=check_validity,
            )
            if metric_state.wants("logical_gap") and (not self.comparative_decoding or logical_value is not None):
                raise ValueError("logical_gap metrics require all comparative logical classes")
            if metric_state.wants("color_correlated_run") and not self.enable_colorcorrelated_decoding:
                raise ValueError("color_correlated_run requires color-correlated decoding")
            if metric_state.wants("relift_run") and not self.enable_cross_color_relifting:
                raise ValueError("relift_run requires cross-color relifting")
            if self.enable_colorcorrelated_decoding and metric_state.need_baseline:
                raise ValueError("color-correlated error metrics require ordinary baseline_predictions")
            # SWIM is scored as candidates are generated, on the unchanged
            # prior. No all-candidate correction/hypothesis export is needed.
            compute_swim_distance = False
            if not shot_count:
                output = np.empty((0,) if self.num_obs == 1 else (0, self.num_obs), dtype=bool)
                return output, metric_state.finish(
                    output, np.empty(0), np.empty((1, 3, 0)), None,
                    np.empty((0, self.num_obs), dtype=bool), run_category=np.empty(0),
                )

        if self.enable_prior_perturbation:
            if custom_dem_data is not None:
                raise NotImplementedError("Prior perturbation with BP/custom DEM priors is unsupported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding:
                raise NotImplementedError("Prior perturbation with erasure predecoding is unsupported")
            return self._decode_prior_perturbation(
                detector_outcomes, colors, logical_value, full_output, check_validity,
                compute_swim_distance, perturbation_shot_offset, metric_state,
            )
        if self.enable_cross_color_relifting:
            if self.dem_manager.remove_non_edge_like_errors:
                raise NotImplementedError("Cross-color relifting requires remove_non_edge_like_errors=False")
            if self.enable_colorcorrelated_decoding:
                raise NotImplementedError("Cross-color relifting and color-correlated decoding cannot be combined")
            if custom_dem_data is not None:
                raise NotImplementedError("Cross-color relifting with BP/custom DEM priors is unsupported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding:
                raise NotImplementedError("Cross-color relifting with predecoding ensembles is unsupported")
            return self._decode_cross_color_relifting(
                detector_outcomes, colors, logical_value, full_output, check_validity,
                compute_swim_distance, return_candidate_data, metric_state,
            )
        if self.enable_colorcorrelated_decoding:
            if custom_dem_data is not None:
                raise NotImplementedError(
                    "Color-correlated decoding with custom_dem_data/BP is not supported"
                )

        if erasure_matcher_predecoding:
            if not self.comparative_decoding:
                raise ValueError(
                    "Erasure matcher predecoding requires comparative_decoding=True"
                )

        # Ensure detector_outcomes is 2D
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim == 1:
            detector_outcomes = detector_outcomes.reshape(1, -1)

        # Process color selection
        if colors == "all":
            colors = ["r", "g", "b"]
        elif colors in ["r", "g", "b"]:
            colors = [colors]

        if self.enable_colorcorrelated_decoding and (
            len(colors) != 3 or set(colors) != {"r", "g", "b"}
        ):
            raise ValueError("Color-correlated decoding requires all three colors r, g, b")
        if self.enable_colorcorrelated_decoding:
            colors = ["r", "g", "b"]
            specs = candidate_specs()
            if self._color_correlated_reweighter is None:
                self._color_correlated_reweighter = ColorCorrelatedPriorReweighter(
                    self.dem_manager, self.color_correlated_b)
            reweighter = self._color_correlated_reweighter
        if self.enable_colorcorrelated_decoding or self.color_correlated_weight_basis == "original_dem":
            evaluator = self._candidate_evaluators.get(self.color_correlated_weight_basis)
            if evaluator is None:
                evaluator = CandidateEvaluator(self.dem_manager, self.color_correlated_weight_basis)
                self._candidate_evaluators[self.color_correlated_weight_basis] = evaluator

        if compute_swim_distance and detector_outcomes.shape[0] == 0:
            result = np.empty((0,) if self.num_obs == 1 else (0,self.num_obs), dtype=bool)
            extra = {"best_colors": np.empty(0,dtype=np.uint8), "weights": np.empty(0),
                     "error_preds": np.empty((0,self.dem_manager.H.shape[1]),dtype=bool),
                     "color_order": tuple(colors),
                     "stage2_weights_by_color": np.empty((0,len(colors))),
                     "swim_distances_by_color": np.empty((0,len(colors))),
                     "selected_swim_distance": np.empty(0), "swim_bound_certified": False}
            if check_validity: extra["validity"] = np.empty(0,dtype=bool)
            return (result, extra) if full_output else result

        # Generate all logical value combinations for comparative decoding
        all_logical_values = np.array(
            list(itertools.product([False, True], repeat=self.num_obs))
        )

        if logical_value is not None:
            logical_value = np.asarray(logical_value, dtype=bool).ravel()
            if len(logical_value) != self.num_obs:
                raise ValueError(f"logical_value must have length {self.num_obs}")

        # Handle cultivation circuit post-selection
        if self.circuit_type == "cult+growing":
            cult_interface_det_ids = (
                self.dem_manager.cult_detector_ids
                + self.dem_manager.interface_detector_ids
            )
            cult_success = ~np.any(detector_outcomes[:, cult_interface_det_ids], axis=1)
            detector_outcomes = detector_outcomes[cult_success, :]

        # Determine number of logical classes to test
        num_logical_classes = (
            len(all_logical_values)
            if self.comparative_decoding and logical_value is None
            else 1
        )

        # Stage 1 decoding for all logical classes and colors
        error_preds_stage1_all = []
        if verbose:
            print("First-round decoding:")

        for i in range(num_logical_classes):
            error_preds_stage1_all.append({})
            for c in colors:
                if verbose:
                    print(f"    > logical class {i}, color {c}...")

                if self.comparative_decoding:
                    detector_outcomes_copy = detector_outcomes.copy()
                    if logical_value is not None:
                        detector_outcomes_copy[:, -self.num_obs :] = logical_value
                    else:
                        detector_outcomes_copy[:, -self.num_obs :] = all_logical_values[
                            i
                        ]
                    error_preds_stage1_all[i][c] = self._decode_stage1(
                        detector_outcomes_copy, c, custom_dem_data
                    )
                else:
                    error_preds_stage1_all[i][c] = self._decode_stage1(
                        detector_outcomes, c, custom_dem_data
                    )

        # Erasure matcher predecoding
        if erasure_matcher_predecoding:
            if len(error_preds_stage1_all) <= 1:
                raise ValueError(
                    "Erasure matcher predecoding requires multiple logical classes"
                )

            if verbose:
                print("Erasure matcher predecoding:")

            (
                predecoding_obs_preds,
                predecoding_error_preds,
                predecoding_weights,
                predecoding_success,
            ) = self._erasure_matcher_predecoding(
                error_preds_stage1_all, detector_outcomes
            )

            predecoding_failure = ~predecoding_success
            detector_outcomes_left = detector_outcomes[predecoding_failure, :]
            error_preds_stage1_left = [
                {
                    c: arr[predecoding_failure, :]
                    for c, arr in error_preds_stage1_all[i].items()
                }
                for i in range(len(error_preds_stage1_all))
            ]

            if verbose:
                print(
                    f"    > # of samples with successful predecoding: {predecoding_success.sum()}"
                )
        else:
            detector_outcomes_left = detector_outcomes
            error_preds_stage1_left = error_preds_stage1_all

        # Stage 2 decoding
        if verbose:
            print("Second-round decoding:")

        num_left_samples = detector_outcomes_left.shape[0]
        need_all_corrections = (metric_state is None and
                               (full_output or check_validity or compute_swim_distance
                                or return_candidate_data))
        if self.enable_colorcorrelated_decoding:
            candidate_shape = (num_logical_classes, 12, num_left_samples)
            candidate_weights = np.full(candidate_shape, np.nan) if full_output else None
            generation_weights = np.full(candidate_shape, np.nan) if full_output else None
            candidate_executed = (np.zeros(candidate_shape, dtype=bool)
                                  if full_output or compute_swim_distance or return_candidate_data else None)
            run_by_class = (np.full((num_logical_classes, num_left_samples), -1, dtype=np.int8)
                            if full_output or (metric_state is not None and metric_state.wants("color_correlated_run")) else None)
            color_correlated_run = np.full(num_left_samples, -1, dtype=np.int8) if full_output else None
            best_candidate_indices = np.full(num_left_samples, -1, dtype=int) if full_output else None
            native_candidate_preds = ([[None] * 12 for _ in range(num_logical_classes)]
                                      if full_output else None)
            if compute_swim_distance or return_candidate_data:
                candidate_stage1_hypotheses = [[None] * 12 for _ in range(num_logical_classes)]

        if num_left_samples > 0 and not (
            erasure_matcher_predecoding and partial_correction_by_predecoding
        ):
            num_errors = self.dem_manager.H.shape[1]

            num_candidates = 12 if self.enable_colorcorrelated_decoding else len(colors)
            if self.enable_colorcorrelated_decoding:
                error_preds = np.zeros(
                    (num_logical_classes, num_candidates if need_all_corrections else 3,
                     num_left_samples, num_errors), dtype=bool,
                )
                weights = np.full(
                    (num_logical_classes, num_candidates, num_left_samples), np.inf
                )
                if full_output:
                    generation_weights[:, 3:, :] = np.inf
            else:
                error_preds = (np.empty(
                    (num_logical_classes, num_candidates, num_left_samples, num_errors),
                    dtype=bool,
                ) if need_all_corrections else None)
                weights = np.empty(
                    (num_logical_classes, num_candidates, num_left_samples), dtype=float
                )

            candidate_observables = (np.zeros(weights.shape + (self.num_obs,), dtype=bool)
                                     if metric_state is None and not need_all_corrections and not self.comparative_decoding
                                     else None)
            if compute_swim_distance:
                swim_by_color = np.empty((num_left_samples,len(colors)))
                swim_stage2_weights = np.empty((num_left_samples,len(colors)))
            for i in range(len(error_preds_stage1_left)):
                for i_c, c in enumerate(colors):
                    if verbose:
                        print(f"    > logical class {i}, color {c}...")

                    if self.comparative_decoding:
                        detector_outcomes_copy = detector_outcomes_left.copy()
                        if logical_value is not None:
                            detector_outcomes_copy[:, -self.num_obs :] = logical_value
                        else:
                            detector_outcomes_copy[:, -self.num_obs :] = (
                                all_logical_values[i]
                            )
                        error_preds_new, weights_new = self._decode_stage2(
                            detector_outcomes_copy,
                            error_preds_stage1_left[i][c],
                            c,
                            custom_dem_data,
                        )
                    else:
                        stage2_result = self._decode_stage2(
                            detector_outcomes_left,
                            error_preds_stage1_left[i][c],
                            c,
                            custom_dem_data,
                            compute_swim_distance=compute_swim_distance,
                        )
                        if compute_swim_distance:
                            error_preds_new = stage2_result.predictions
                            weights_new = stage2_result.solution_weights
                            swim_by_color[:,i_c] = stage2_result.swim_distances
                            swim_stage2_weights[:,i_c] = weights_new
                        else:
                            error_preds_new, weights_new = stage2_result

                    if self.enable_colorcorrelated_decoding:
                        (error_preds_new, base_native, weights_new,
                         generation_weight) = evaluator.evaluate(
                            c, error_preds_new, weights_new
                        )
                        if compute_swim_distance or return_candidate_data:
                            candidate_stage1_hypotheses[i][i_c] = error_preds_stage1_left[i][c]
                        if full_output:
                            native_candidate_preds[i][i_c] = base_native
                            generation_weights[i, i_c] = generation_weight
                    elif self.color_correlated_weight_basis == "original_dem":
                        # Generate with the ordinary stage-2 matching prior, but
                        # compare the mapped corrections on the original DEM.
                        (error_preds_new, _, weights_new, _) = evaluator.evaluate(
                            c, error_preds_new, weights_new
                        )
                    else:
                        # Preserve the ordinary decoder's historical weights.
                        error_preds_new = self.dem_manager.dems_decomposed[
                            c
                        ].map_errors_to_org_dem(error_preds_new, stage=2)

                    if metric_state is not None:
                        metric_state.observe(
                            i, i_c, num_candidates, slice(None), error_preds_new,
                            weights_new, weights_new, detectors=detector_outcomes_left,
                            hypothesis=error_preds_stage1_left[i][c], color=c,
                        )
                    if need_all_corrections or self.enable_colorcorrelated_decoding:
                        error_preds[i, i_c, :, :] = error_preds_new
                    if candidate_observables is not None:
                        candidate_observables[i, i_c] = np.asarray(
                            (error_preds_new.astype(np.uint8) @ self.dem_manager.obs_matrix.T) % 2,
                            dtype=bool)
                    weights[i, i_c, :] = weights_new
                    if self.enable_colorcorrelated_decoding and candidate_executed is not None:
                        candidate_executed[i, i_c, :] = True

                if self.enable_colorcorrelated_decoding:
                    # Guides are the three baseline original-DEM corrections of
                    # this logical class. The two-guide case is Boolean union.
                    guides = {c: error_preds[i, j] for j, c in enumerate(colors)}
                    schedules = []
                    for shot in range(num_left_samples):
                        category, indices = candidate_schedule(error_preds[i, :3, shot])
                        if run_by_class is not None:
                            run_by_class[i, shot] = category
                        schedules.append(indices)
                    for i_spec, spec in enumerate(specs[3:], start=3):
                        c = spec.target_color
                        for shot in range(num_left_samples):
                            if i_spec not in schedules[shot]:
                                continue
                            guide = guide_union(
                                {name: correction[shot] for name, correction in guides.items()},
                                spec.guide_colors,
                            )
                            base_decomp = self.dem_manager.dems_decomposed[c]
                            temporary_dem = {c: (
                                StagePrior(self._matching_cache.base_structure(c, 1, base_decomp.Hs[0]),
                                           reweighter.stage1_probabilities(c, guide)),
                                (base_decomp.Hs[1], base_decomp.probs[1]),
                            )}
                            det = detector_outcomes_left[shot:shot + 1].copy()
                            if self.comparative_decoding:
                                det[:, -self.num_obs:] = (
                                    logical_value if logical_value is not None
                                    else all_logical_values[i]
                                )
                            stage1 = self._decode_stage1(det, c, temporary_dem)
                            if compute_swim_distance or return_candidate_data:
                                if candidate_stage1_hypotheses[i][i_spec] is None:
                                    candidate_stage1_hypotheses[i][i_spec] = np.zeros(
                                        (num_left_samples, stage1.shape[1]), dtype=bool)
                                candidate_stage1_hypotheses[i][i_spec][shot] = stage1[0]
                            native, generation_weight = self._decode_stage2(
                                det, stage1, c
                            )
                            (mapped, base_native, selection_weight,
                             diagnostic_weight) = evaluator.evaluate(
                                c, native[0], generation_weight[0]
                            )
                            if metric_state is not None:
                                metric_state.observe(
                                    i, i_spec, num_candidates, slice(shot, shot + 1),
                                    mapped, selection_weight, diagnostic_weight,
                                    detectors=det, hypothesis=stage1, color=c,
                                )
                            if full_output:
                                if native_candidate_preds[i][i_spec] is None:
                                    native_candidate_preds[i][i_spec] = np.zeros(
                                        (num_left_samples, base_native.shape[0]), dtype=bool)
                                native_candidate_preds[i][i_spec][shot] = base_native
                                generation_weights[i, i_spec, shot] = diagnostic_weight
                            if need_all_corrections:
                                error_preds[i, i_spec, shot] = mapped
                            if candidate_observables is not None:
                                candidate_observables[i, i_spec, shot] = np.asarray(
                                    (mapped.astype(np.uint8) @ self.dem_manager.obs_matrix.T) % 2,
                                    dtype=bool)
                            if candidate_executed is not None:
                                candidate_executed[i, i_spec, shot] = True
                            weights[i, i_spec, shot] = selection_weight

            if self.enable_colorcorrelated_decoding and full_output:
                candidate_weights = weights.copy()

            # Find best predictions across logical classes and colors
            best_logical_classes, best_color_inds, weights_final, logical_gaps = (
                _get_final_predictions(weights)
            )
            if self.enable_colorcorrelated_decoding and full_output:
                best_candidate_indices = best_color_inds.copy()
                color_correlated_run = run_by_class[
                    best_logical_classes, np.arange(num_left_samples)
                ]

            error_preds_final = (error_preds[
                best_logical_classes, best_color_inds, np.arange(num_left_samples), :
            ] if need_all_corrections else metric_state.selected if metric_state is not None else None)

            # Calculate observable predictions
            if self.comparative_decoding:
                if logical_value is None:
                    obs_preds_final = all_logical_values[best_logical_classes]
                    if obs_preds_final.shape != (num_left_samples, self.num_obs):
                        raise RuntimeError("Observable prediction shape mismatch")
                else:
                    obs_preds_final = np.tile(logical_value, (num_left_samples, 1))
            elif candidate_observables is not None:
                obs_preds_final = candidate_observables[
                    best_logical_classes, best_color_inds, np.arange(num_left_samples)]
            else:
                obs_preds_final = np.empty((num_left_samples, self.num_obs), dtype=bool)
                for i_c, c in enumerate(colors):
                    obs_matrix = self.dem_manager.obs_matrix
                    if self.enable_colorcorrelated_decoding:
                        mask = np.array([spec.target_color == c for spec in specs])[best_color_inds]
                    else:
                        mask = best_color_inds == i_c
                    obs_preds_final[mask, :] = (
                        (error_preds_final[mask, :].astype("uint8") @ obs_matrix.T) % 2
                    ).astype(bool)

            if compute_swim_distance:
                if self.enable_colorcorrelated_decoding:
                    candidate_swim, class_min_swim = self._score_candidate_swim(
                        detector_outcomes_left, error_preds,
                        candidate_stage1_hypotheses,
                        tuple(spec.target_color for spec in specs),
                        obs_preds_final, candidate_executed,
                    )
                else:
                    # All ordinary branches use the unchanged stage-2 prior.
                    flat = error_preds[0].reshape(-1, error_preds.shape[-1])
                    candidate_observables = np.asarray(
                        (flat.astype(np.uint8) @ self.dem_manager.obs_matrix.T) % 2,
                        dtype=bool,
                    ).reshape(len(colors), num_left_samples, self.num_obs)
                    same_class = np.all(candidate_observables == obs_preds_final[None, :, :], axis=-1)
                    class_min_swim = np.min(np.where(same_class.T, swim_by_color, np.inf), axis=1)
                if self.enable_colorcorrelated_decoding:
                    selected_swim = candidate_swim[
                        best_logical_classes, best_color_inds, np.arange(num_left_samples)]
                else:
                    selected_swim = swim_by_color[np.arange(num_left_samples), best_color_inds]

            # Adjust color indices for non-standard color selections
            if self.enable_colorcorrelated_decoding:
                best_colors = np.array([
                    color_to_color_val(spec.target_color) for spec in specs
                ], dtype=np.uint8)[best_color_inds]
            elif colors == ["r", "g", "b"]:
                best_colors = best_color_inds
            else:
                best_colors = np.array([color_to_color_val(c) for c in colors])[
                    best_color_inds
                ]

        elif (
            num_left_samples > 0
            and erasure_matcher_predecoding
            and partial_correction_by_predecoding
        ):
            # Partial correction strategy
            predecoding_error_preds_failed = predecoding_error_preds[
                predecoding_failure, :
            ].astype("uint8")

            def get_partial_corr(matrix):
                corr = (predecoding_error_preds_failed @ matrix.T) % 2
                return corr.astype(bool)

            obs_partial_corr = get_partial_corr(self.dem_manager.obs_matrix)
            det_partial_corr = get_partial_corr(self.dem_manager.H)
            detector_outcomes_left ^= det_partial_corr

            # Recursive call with partial correction
            obs_preds_final = self.decode(
                detector_outcomes_left,
                colors=colors,
                full_output=full_output,
            )
            if full_output:
                obs_preds_final, extra_outputs = obs_preds_final
            else:
                extra_outputs = {}

            if obs_preds_final.ndim == 1:
                obs_preds_final = obs_preds_final[:, np.newaxis]

            if full_output:
                error_preds_final = extra_outputs["error_preds"]
                best_colors = extra_outputs["best_colors"]
                weights_final = extra_outputs["weights"]
                logical_gaps = extra_outputs["logical_gaps"]
                if self.enable_colorcorrelated_decoding:
                    best_candidate_indices = extra_outputs["best_candidate_indices"]
                    candidate_weights = extra_outputs["candidate_weights"]
                    generation_weights = extra_outputs["candidate_generation_weights"]
                    candidate_executed = extra_outputs["candidate_executed"]
                    color_correlated_run = extra_outputs["color_correlated_run"]
                    native_candidate_preds = extra_outputs["candidate_native_stage2_preds"]

        else:
            # No samples to decode
            error_preds_final = np.array([[]], dtype=bool)
            obs_preds_final = np.array([[]], dtype=bool)
            best_colors = np.array([], dtype=np.uint8)
            weights_final = np.array([], dtype=float)
            logical_gaps = np.array([], dtype=float)

        # Merge predecoding and second-round results
        if erasure_matcher_predecoding and np.any(predecoding_success):
            if verbose:
                print("Merging predecoding & second-round decoding outcomes")

            full_obs_preds_final = predecoding_obs_preds.copy()
            if full_output:
                full_best_colors = np.full(detector_outcomes.shape[0], "P")
                full_weights_final = predecoding_weights.copy()
                full_logical_gaps = np.full(detector_outcomes.shape[0], -1)
                full_error_preds_final = predecoding_error_preds.copy()

            if detector_outcomes_left.shape[0] > 0:
                if partial_correction_by_predecoding:
                    obs_preds_final ^= obs_partial_corr
                    if full_output:
                        error_preds_final ^= predecoding_error_preds_failed.astype(bool)

                full_obs_preds_final[predecoding_failure, :] = obs_preds_final

                if full_output:
                    full_best_colors[predecoding_failure] = best_colors
                    full_weights_final[predecoding_failure] = weights_final
                    full_logical_gaps[predecoding_failure] = logical_gaps
                    full_error_preds_final[predecoding_failure, :] = error_preds_final

            obs_preds_final = full_obs_preds_final
            if full_output:
                best_colors = full_best_colors
                weights_final = full_weights_final
                logical_gaps = full_logical_gaps
                error_preds_final = full_error_preds_final
                if self.enable_colorcorrelated_decoding:
                    full_best_candidate_indices = np.full(detector_outcomes.shape[0], -1, dtype=int)
                    full_best_candidate_indices[predecoding_failure] = best_candidate_indices
                    best_candidate_indices = full_best_candidate_indices
                    full_candidate_weights = np.full(
                        (num_logical_classes, 12, detector_outcomes.shape[0]), np.nan
                    )
                    full_generation_weights = full_candidate_weights.copy()
                    full_candidate_weights[:, :, predecoding_failure] = candidate_weights
                    full_generation_weights[:, :, predecoding_failure] = generation_weights
                    candidate_weights = full_candidate_weights
                    generation_weights = full_generation_weights
                    full_candidate_executed = np.zeros(
                        (num_logical_classes, 12, detector_outcomes.shape[0]), dtype=bool
                    )
                    full_candidate_executed[:, :, predecoding_failure] = candidate_executed
                    candidate_executed = full_candidate_executed
                    full_color_correlated_run = np.full(
                        detector_outcomes.shape[0], -1, dtype=np.int8
                    )
                    full_color_correlated_run[predecoding_failure] = color_correlated_run
                    color_correlated_run = full_color_correlated_run

        # Validity checking
        if check_validity:
            det_preds = (
                error_preds_final.astype("uint8") @ self.dem_manager.H.T % 2
            ).astype(bool)
            validity = np.all(det_preds == detector_outcomes, axis=1)
            if verbose:
                if np.all(validity):
                    print("All predictions are valid")
                else:
                    print(f"{np.sum(~validity)} invalid predictions found!")

        # Format output
        if obs_preds_final.shape[1] == 1:
            obs_preds_final = obs_preds_final.ravel()

        if metric_state is not None:
            category = (run_by_class[best_logical_classes, np.arange(num_left_samples)]
                        if self.enable_colorcorrelated_decoding and run_by_class is not None else None)
            return obs_preds_final, metric_state.finish(
                obs_preds_final, weights_final, weights, logical_gaps,
                all_logical_values if self.comparative_decoding and logical_value is None else [logical_value],
                run_category=category,
            )

        if full_output:
            extra_outputs = {
                "best_colors": best_colors,
                "weights": weights_final,
                "error_preds": error_preds_final,
            }
            if self.enable_colorcorrelated_decoding:
                extra_outputs.update(
                    candidate_labels=tuple(spec.label for spec in specs),
                    candidate_target_colors=tuple(spec.target_color for spec in specs),
                    best_candidate_indices=best_candidate_indices,
                    candidate_weights=candidate_weights,
                    candidate_executed=candidate_executed,
                    color_correlated_run=color_correlated_run,
                    candidate_weight_basis=self.color_correlated_weight_basis,
                    candidate_generation_weights=generation_weights,
                    candidate_native_stage2_preds=native_candidate_preds,
                )
                if return_candidate_data:
                    extra_outputs.update(
                        candidate_stage1_hypotheses=candidate_stage1_hypotheses,
                        candidate_original_corrections=error_preds,
                        candidate_valid=candidate_executed,
                    )
            elif return_candidate_data:
                extra_outputs.update(
                    candidate_target_colors=tuple(colors),
                    candidate_stage1_hypotheses=[
                        [error_preds_stage1_left[i][c] for c in colors]
                        for i in range(num_logical_classes)],
                    candidate_original_corrections=error_preds,
                    candidate_valid=np.ones(error_preds.shape[:3], dtype=bool),
                )

            if compute_swim_distance:
                from ..soft_output.results import GROWTH_CONVENTION
                extra_outputs.update(selected_swim_distance=selected_swim,
                                     class_min_swim_distance=class_min_swim,
                                     swim_growth_convention=GROWTH_CONVENTION,
                                     swim_bound_certified=False)
                if self.enable_colorcorrelated_decoding:
                    extra_outputs["candidate_swim_distances"] = candidate_swim
                else:
                    extra_outputs.update(color_order=tuple(colors),
                        stage2_weights_by_color=swim_stage2_weights.copy(),
                        swim_distances_by_color=swim_by_color)

            if len(error_preds_stage1_all) > 1:
                extra_outputs["logical_gaps"] = logical_gaps
                extra_outputs["logical_values"] = all_logical_values
                if erasure_matcher_predecoding:
                    extra_outputs["erasure_matcher_success"] = predecoding_success
                    extra_outputs["predecoding_error_preds"] = predecoding_error_preds
                    extra_outputs["predecoding_obs_preds"] = predecoding_obs_preds

            if self.circuit_type == "cult+growing":
                extra_outputs["cult_success"] = cult_success

            if check_validity:
                extra_outputs["validity"] = validity

            return obs_preds_final, extra_outputs
        else:
            return obs_preds_final

    def _score_candidate_swim(self, detectors, mapped, hypotheses, targets,
                              selected_observables, valid=None):
        """Score each generated stage-2 syndrome on the unchanged base graph.

        Perturbed priors may generate a candidate correction, but they never
        enter this metric calculation. The reported score is the minimum among
        generated candidates with the final hard decision's logical parity.
        """
        shots = len(detectors)
        scores = np.full(mapped.shape[:3], np.inf, dtype=float)
        if valid is None:
            valid = np.ones(scores.shape, dtype=bool)
        for logical_class in range(mapped.shape[0]):
            for slot, color in enumerate(targets):
                hypothesis = hypotheses[logical_class][slot]
                active = valid[logical_class, slot]
                if hypothesis is None or not np.any(active):
                    continue
                result = self._decode_stage2(
                    detectors[active], np.asarray(hypothesis, dtype=bool)[active],
                    color, compute_swim_distance=True,
                )
                scores[logical_class, slot, active] = result.swim_distances
        flat = mapped.reshape(-1, mapped.shape[-1])
        observable = np.asarray(
            (flat.astype(np.uint8) @ self.dem_manager.obs_matrix.T) % 2,
            dtype=bool,
        ).reshape(mapped.shape[:3] + (self.num_obs,))
        same = np.all(observable == selected_observables[None, None, :, :], axis=-1)
        selected = np.min(np.where(same & valid, scores, np.inf), axis=(0, 1))
        if selected.shape != (shots,) or not np.isfinite(selected).all():
            raise RuntimeError("No finite same-logical-class swim candidate")
        return scores, selected

    def _decode_prior_perturbation(
        self, detector_outcomes, colors, logical_value, full_output, check_validity,
        compute_swim_distance=False, perturbation_shot_offset=None, metric_state=None,
    ):
        """Generate a per-shot ensemble and retain the existing candidate contract."""
        if colors != "all" and colors != ["r", "g", "b"]:
            raise ValueError("Prior perturbation requires all three colors r, g, b")
        if self._perturbation_ensemble is None:
            if self.stage1_perturbation:
                self._perturbation_ensemble = NativeStage1Ensemble(
                    self.dem_manager, self.perturbation_ensemble_size,
                    self.perturbation_alpha, self.perturbation_seed, self._matching_cache)
            else:
                self._perturbation_ensemble = PriorPerturbationEnsemble(
                    self.dem_manager, self.perturbation_ensemble_size,
                    self.perturbation_alpha, self.perturbation_seed,
                )
        ensemble = self._perturbation_ensemble
        manager = self.dem_manager
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim == 1:
            detector_outcomes = detector_outcomes[None, :]
        if logical_value is not None:
            logical_value = np.asarray(logical_value, dtype=bool).ravel()
            if logical_value.size != self.num_obs:
                raise ValueError(f"logical_value must have length {self.num_obs}")
        logical_classes = (list(itertools.product((False, True), repeat=self.num_obs))
                           if self.comparative_decoding and logical_value is None
                           else [logical_value])
        colors = ("r", "g", "b")
        members = (tuple(m for m in range(self.perturbation_ensemble_size) for _ in colors)
                   if full_output else None)
        targets = colors * self.perturbation_ensemble_size
        labels = (tuple(f"m{m}:{c}" for m in range(self.perturbation_ensemble_size) for c in colors)
                  if full_output else None)
        n_classes, n_shots, n_errors = len(logical_classes), len(detector_outcomes), manager.H.shape[1]
        n_candidates = 3 * self.perturbation_ensemble_size
        native_stage1 = None
        if self.stage1_perturbation:
            offset = ensemble.start(perturbation_shot_offset, n_shots)
            native_stage1 = []
            for logical in logical_classes:
                det = detector_outcomes.copy()
                if self.comparative_decoding:
                    det[:, -self.num_obs:] = logical
                native_stage1.append({color: ensemble.decode_stage1(det, color, offset) for color in colors})
        native_swim = None
        if self.stage1_perturbation and metric_state is not None and metric_state.swim is not None:
            # Reuse the native batch hypotheses instead of copying them into
            # per-candidate exports or making one SWIM API call per shot.
            native_swim = {}
            for color in colors:
                scores = np.empty((n_shots, ensemble.size), dtype=float)
                for member in range(ensemble.size):
                    scores[:, member] = metric_state.candidate_scorer(
                        detector_outcomes, native_stage1[0][color][:, member], color)
                native_swim[color] = scores
        need_mapped = metric_state is None and (full_output or compute_swim_distance or check_validity)
        need_hypotheses = full_output or compute_swim_distance
        mapped = (np.zeros((n_classes, n_candidates, n_shots, n_errors), dtype=bool)
                  if need_mapped else None)
        stage1_hypotheses = ([[None] * n_candidates for _ in range(n_classes)]
                             if need_hypotheses else None)
        native = [[None] * n_candidates for _ in range(n_classes)] if full_output else None
        weights = np.empty((n_classes, n_candidates, n_shots), dtype=float)
        generations = np.empty_like(weights) if full_output else None
        observables = (np.empty(weights.shape + (self.num_obs,), dtype=bool)
                       if metric_state is None and not need_mapped and not self.comparative_decoding else None)
        evaluator = self._candidate_evaluators.get(self.color_correlated_weight_basis)
        if evaluator is None:
            evaluator = CandidateEvaluator(manager, self.color_correlated_weight_basis)
            self._candidate_evaluators[self.color_correlated_weight_basis] = evaluator
        # Preserve the historical batched floating-point scoring operations
        # when there is no stochastic perturbation. Only genuinely per-shot
        # priors require single-shot matching/scoring problems.
        fixed_priors = self.perturbation_ensemble_size == 1 or self.perturbation_alpha == 0
        shot_groups = ([slice(0, n_shots)] if fixed_priors and n_shots else
                       (slice(shot, shot + 1) for shot in range(n_shots)))
        for shot_slice in shot_groups:
            if self.stage1_perturbation:
                decompositions_by_member = [manager.dems_decomposed] * ensemble.size
            else:
                decompositions_by_member = ensemble.next_shot()
                if fixed_priors:
                    ensemble.shot_position += n_shots - 1
            for class_index, logical in enumerate(logical_classes):
                det = detector_outcomes[shot_slice].copy()
                if self.comparative_decoding:
                    det[:, -self.num_obs:] = logical
                for member, decompositions in enumerate(decompositions_by_member):
                    for color_index, color in enumerate(colors):
                        slot = 3 * member + color_index
                        temporary = decompositions[color]
                        custom = None if member == 0 or self.stage1_perturbation else {
                            color: (temporary.stage_priors if hasattr(temporary, 'stage_priors')
                                    else tuple(StagePrior(
                                        self._matching_cache.base_structure(color, stage, H), p)
                                        for stage, (H, p) in enumerate(zip(temporary.Hs, temporary.probs), start=1)))
                        }
                        stage1 = (native_stage1[class_index][color][shot_slice, member]
                                  if self.stage1_perturbation else self._decode_stage1(det, color, custom))
                        if need_hypotheses:
                            if stage1_hypotheses[class_index][slot] is None:
                                stage1_hypotheses[class_index][slot] = np.zeros(
                                    (n_shots, stage1.shape[1]), dtype=bool)
                            stage1_hypotheses[class_index][slot][shot_slice] = stage1
                        stage2_custom = None if self.use_original_prior_for_stage2 else custom
                        stage2, generation = self._decode_stage2(det, stage1, color, stage2_custom)
                        correction, aligned, score, diagnostic = evaluator.evaluate(
                            color, stage2, generation,
                            temporary=None if stage2_custom is None else temporary,
                        )
                        if metric_state is not None:
                            metric_state.observe(
                                class_index, slot, n_candidates, shot_slice,
                                correction, score, diagnostic,
                                detectors=det, hypothesis=stage1, color=color,
                                swim_scores=native_swim[color][shot_slice, member] if native_swim is not None else None,
                            )
                        if need_mapped:
                            mapped[class_index, slot, shot_slice] = correction
                        if observables is not None:
                            observables[class_index, slot, shot_slice] = np.asarray(
                                (correction.astype(np.uint8) @ manager.obs_matrix.T) % 2,
                                dtype=bool)
                        if full_output:
                            if native[class_index][slot] is None:
                                native[class_index][slot] = np.zeros(
                                    (n_shots, aligned.shape[1]), dtype=bool)
                            native[class_index][slot][shot_slice] = aligned
                            generations[class_index, slot, shot_slice] = diagnostic
                        weights[class_index, slot, shot_slice] = score
        if self.stage1_perturbation:
            ensemble.advance(offset, n_shots)
        if n_shots:
            best_class, best_slot, selected_weights, gaps = _get_final_predictions(weights)
            selected = (mapped[best_class, best_slot, np.arange(n_shots)]
                        if need_mapped else metric_state.selected if metric_state is not None else None)
            if self.comparative_decoding:
                observed = np.asarray([logical_classes[i] for i in best_class], dtype=bool)
                if logical_value is not None:
                    observed[:] = logical_value
            elif selected is not None:
                observed = np.asarray((selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool)
            else:
                observed = observables[best_class, best_slot, np.arange(n_shots)]
            best_colors = np.asarray([color_to_color_val(targets[i]) for i in best_slot], dtype=np.uint8)
        else:
            selected = np.zeros((0, n_errors), dtype=bool)
            observed = np.zeros((0, self.num_obs), dtype=bool)
            selected_weights = np.zeros(0)
            best_slot = np.zeros(0, dtype=int)
            best_class = np.zeros(0, dtype=int)
            best_colors = np.zeros(0, dtype=np.uint8)
            gaps = None
        output = observed.ravel() if self.num_obs == 1 else observed
        if check_validity and metric_state is not None:
            valid = np.all(np.asarray((selected.astype(np.uint8) @ manager.H.T) % 2,
                                      dtype=bool) == detector_outcomes, axis=1)
        if metric_state is not None:
            return output, metric_state.finish(output, selected_weights, weights, gaps, logical_classes)
        if compute_swim_distance:
            from ..soft_output.results import GROWTH_CONVENTION
            candidate_swim, class_min_swim = self._score_candidate_swim(
                detector_outcomes, mapped, stage1_hypotheses, targets, observed)
            selected_swim = candidate_swim[
                best_class, best_slot, np.arange(n_shots)]
        if not full_output:
            return output
        extra = dict(
            best_colors=best_colors, weights=selected_weights, error_preds=selected,
            baseline_predictions=_ordinary_baseline_predictions(
                mapped, generations, manager, logical_classes,
                self.comparative_decoding, self.num_obs),
            candidate_labels=labels, candidate_target_colors=targets,
            candidate_ensemble_members=members, candidate_weights=weights,
            candidate_weight_basis=self.color_correlated_weight_basis,
            candidate_generation_weights=generations,
            candidate_native_stage2_preds=native, best_candidate_indices=best_slot,
            candidate_stage1_hypotheses=stage1_hypotheses,
            candidate_original_corrections=mapped,
        )
        if compute_swim_distance:
            extra.update(candidate_swim_distances=candidate_swim,
                         selected_swim_distance=selected_swim,
                         class_min_swim_distance=class_min_swim,
                         swim_growth_convention=GROWTH_CONVENTION,
                         swim_bound_certified=False)
        if gaps is not None:
            extra.update(logical_gaps=gaps, logical_values=np.asarray(
                list(itertools.product((False, True), repeat=self.num_obs))))
        if check_validity:
            extra["validity"] = np.all(
                np.asarray((selected.astype(np.uint8) @ manager.H.T) % 2, dtype=bool)
                == detector_outcomes, axis=1,
            )
        return output, extra

    def _decode_cross_color_relifting(
        self, detector_outcomes, colors, logical_value, full_output, check_validity,
        compute_swim_distance=False, return_candidate_data=False, metric_state=None,
    ):
        """Decode canonical relift slots with class-local syndrome caching."""
        if colors != "all" and colors != ["r", "g", "b"]:
            raise ValueError("Cross-color relifting requires all three colors r, g, b")
        colors = relift.COLORS
        manager = self.dem_manager
        for color in colors:
            decomp = manager.dems_decomposed[color]
            for stage, H in enumerate(decomp.Hs, start=1):
                if np.any(np.diff(H.tocsc().indptr) > 2):
                    raise NotImplementedError(
                        f"Cross-color relifting unsupported: full {color} stage-{stage} decomposition is not graphlike"
                    )
            if np.any(np.diff(decomp.error_map_matrices[1].tocsc().indptr) > 1):
                raise NotImplementedError(
                    f"Cross-color relifting unsupported: {color} stage-2 original mapping is not one-to-one"
                )
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim == 1:
            detector_outcomes = detector_outcomes[None, :]
        if logical_value is not None:
            logical_value = np.asarray(logical_value, dtype=bool).ravel()
            if logical_value.size != self.num_obs:
                raise ValueError(f"logical_value must have length {self.num_obs}")
        logical_classes = (list(itertools.product((False, True), repeat=self.num_obs))
                           if self.comparative_decoding and logical_value is None
                           else [logical_value])
        specs = relift.candidate_specs()
        n_classes, n_shots, n_errors = len(logical_classes), len(detector_outcomes), manager.H.shape[1]
        need_mapped = metric_state is None and (full_output or compute_swim_distance or return_candidate_data or check_validity)
        mapped = np.zeros((n_classes, 12 if need_mapped else 3, n_shots, n_errors), dtype=bool)
        native = [[None] * 12 for _ in range(n_classes)] if full_output else None
        if compute_swim_distance or return_candidate_data:
            stage1_hypotheses = [[None] * 12 for _ in range(n_classes)]
        weights = np.full((n_classes, 12, n_shots), np.inf)
        generations = np.full_like(weights, np.nan) if full_output else None
        validity = np.zeros_like(weights, dtype=bool) if full_output or compute_swim_distance else None
        executed = np.zeros_like(weights, dtype=bool) if full_output else None
        alias = np.full(weights.shape, -1, dtype=int) if full_output else None
        run_class = (np.full((n_classes, n_shots), -1, dtype=np.int8)
                     if full_output or (metric_state is not None and metric_state.wants("relift_run")) else None)
        extra_calls = np.zeros((n_classes, n_shots), dtype=np.int8) if full_output else None
        pairwise_baseline_syndrome_equal = np.zeros((n_classes, n_shots), dtype=np.int8) if full_output else None
        cache_skips = np.zeros((n_classes, n_shots), dtype=np.int8) if full_output else None
        observables = (np.zeros(weights.shape + (self.num_obs,), dtype=bool)
                       if metric_state is None and not need_mapped and not self.comparative_decoding else None)
        evaluator = self._candidate_evaluators.get(self.color_correlated_weight_basis)
        if evaluator is None:
            evaluator = CandidateEvaluator(manager, self.color_correlated_weight_basis)
            self._candidate_evaluators[self.color_correlated_weight_basis] = evaluator
        current_results = {}
        current_swim = {}

        def store(class_index, slot, shot, result, did_execute=False, alias_slot=-1,
                  hypothesis=None):
            correction, base_native, weight, generation = result
            current_results[slot] = result
            if metric_state is not None:
                cached_swim = None
                if metric_state.swim is not None:
                    cached_swim = ([current_swim[alias_slot]] if alias_slot >= 0
                                   else baseline_swim[colors[slot]][shot:shot + 1] if slot < 3 else None)
                scores = metric_state.observe(
                    class_index, slot, 12, slice(shot, shot + 1),
                    correction, weight, generation,
                    detectors=detector_outcomes[shot:shot + 1],
                    hypothesis=None if hypothesis is None else hypothesis[None, :],
                    color=specs[slot].target_color,
                    swim_scores=cached_swim,
                )
                if scores is not None:
                    current_swim[slot] = scores[0]
            if need_mapped or slot < 3:
                mapped[class_index, slot, shot] = correction
            if observables is not None:
                observables[class_index, slot, shot] = np.asarray(
                    (correction.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool)
            if full_output:
                if native[class_index][slot] is None:
                    native[class_index][slot] = np.zeros((n_shots, len(base_native)), dtype=bool)
                native[class_index][slot][shot] = base_native
                generations[class_index, slot, shot] = generation
                executed[class_index, slot, shot] = did_execute
                alias[class_index, slot, shot] = alias_slot
            weights[class_index, slot, shot] = weight
            if validity is not None:
                validity[class_index, slot, shot] = True
            if compute_swim_distance or return_candidate_data:
                if hypothesis is None and alias_slot >= 0:
                    hypothesis = stage1_hypotheses[class_index][alias_slot][shot]
                if hypothesis is None:
                    raise RuntimeError("Missing stage-1 hypothesis for swim candidate")
                if stage1_hypotheses[class_index][slot] is None:
                    stage1_hypotheses[class_index][slot] = np.zeros(
                        (n_shots, len(hypothesis)), dtype=bool)
                stage1_hypotheses[class_index][slot][shot] = hypothesis

        for class_index, logical in enumerate(logical_classes):
            det = detector_outcomes.copy()
            if self.comparative_decoding:
                det[:, -self.num_obs:] = logical
            stage1 = {c: self._decode_stage1(det, c) for c in colors}
            stage1 = {c: np.asarray(v, dtype=bool) for c, v in stage1.items()}
            baseline_swim = ({c: metric_state.candidate_scorer(det, stage1[c], c) for c in colors}
                             if metric_state is not None and metric_state.swim is not None else None)
            baseline = {}
            for color_index, color in enumerate(colors):
                native_batch, generation_batch = self._decode_stage2(det, stage1[color], color)
                for shot in range(n_shots):
                    evaluated = evaluator.evaluate(color, native_batch[shot], generation_batch[shot])
                    result = tuple(np.asarray(v).copy() for v in evaluated)
                    baseline[color] = baseline.get(color, []) + [result]
                    store(class_index, color_index, shot, result, did_execute=True,
                          hypothesis=stage1[color][shot])
            for shot in range(n_shots):
                sources = {c: baseline[c][shot][0] for c in colors}
                category = relift.classify(sources)
                if run_class is not None:
                    run_class[class_index, shot] = category
                current_results = {j: baseline[c][shot] for j, c in enumerate(colors)}
                if metric_state is not None and metric_state.swim is not None:
                    current_swim = {j: baseline_swim[c][shot] for j, c in enumerate(colors)}
                cache = {}
                for color_index, color in enumerate(colors):
                    syndrome = relift.stage2_syndrome(
                        det[shot], stage1[color][shot], color, manager.detector_ids_by_color
                    )
                    cache[(color, syndrome.tobytes())] = (color_index, baseline[color][shot])
                # Count all six pairwise target problems, including candidates
                # pruned by the baseline multiplicity schedule.
                for spec in (specs[3:9] if full_output else ()):
                    target = spec.target_color
                    projected = relift.projection(
                        sources[spec.source_colors[0]],
                        manager.dems_decomposed[target].error_map_matrices[0],
                    )
                    if np.array_equal(projected, stage1[target][shot]):
                        pairwise_baseline_syndrome_equal[class_index, shot] += 1
                for slot, spec in enumerate(specs[3:], start=3):
                    target = spec.target_color
                    target_slot = colors.index(target)
                    if category == 0:
                        store(class_index, slot, shot, baseline[target][shot], alias_slot=target_slot)
                        continue
                    if category == 1:
                        alias_slot = relift.alias_for_class_one(spec, sources, specs)
                        if alias_slot is not None:
                            prior = current_results[alias_slot]
                            store(class_index, slot, shot, prior, alias_slot=alias_slot)
                            continue
                    H1 = manager.dems_decomposed[target].Hs[0]
                    A1 = manager.dems_decomposed[target].error_map_matrices[0]
                    projected = [relift.projection(sources[source], A1)
                                 for source in spec.source_colors]
                    for hypothesis in projected:
                        if not relift.stage1_valid(hypothesis, H1, det[shot]):
                            raise RuntimeError("Cross-color relift violates target stage-1 syndrome")
                    hypothesis = (projected[0] if len(projected) == 1 else
                                  relift.anchored(stage1[target][shot], *projected))
                    if not relift.stage1_valid(hypothesis, H1, det[shot]):
                        raise RuntimeError("Anchored cross-color relift violates target stage-1 syndrome")
                    syndrome = relift.stage2_syndrome(
                        det[shot], hypothesis, target, manager.detector_ids_by_color
                    )
                    key = (target, syndrome.tobytes())
                    if key in cache:
                        prior_slot, prior = cache[key]
                        store(class_index, slot, shot, prior, alias_slot=prior_slot)
                        if full_output:
                            cache_skips[class_index, shot] += 1
                        continue
                    new_native, new_generation = self._decode_stage2(
                        det[shot:shot + 1], hypothesis[None, :], target
                    )
                    evaluated = evaluator.evaluate(target, new_native[0], new_generation[0])
                    result = tuple(np.asarray(v).copy() for v in evaluated)
                    cache[key] = (slot, result)
                    store(class_index, slot, shot, result, did_execute=True,
                          hypothesis=hypothesis)
                    if full_output:
                        extra_calls[class_index, shot] += 1

        if n_shots:
            best_class, best_slot, selected_weights, gaps = _get_final_predictions(weights)
            selected = (mapped[best_class, best_slot, np.arange(n_shots)] if need_mapped
                        else metric_state.selected if metric_state is not None else None)
            if self.comparative_decoding:
                observed = np.asarray([logical_classes[i] for i in best_class], dtype=bool)
                if logical_value is not None:
                    observed[:] = logical_value
            elif selected is not None:
                observed = np.asarray((selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool)
            else:
                observed = observables[best_class, best_slot, np.arange(n_shots)]
            best_colors = np.asarray([color_to_color_val(specs[i].target_color) for i in best_slot], dtype=np.uint8)
            if full_output:
                selected_run = run_class[best_class, np.arange(n_shots)]
                selected_calls = extra_calls[best_class, np.arange(n_shots)]
                selected_pairwise_equal = pairwise_baseline_syndrome_equal[best_class, np.arange(n_shots)]
                selected_cache_skips = cache_skips[best_class, np.arange(n_shots)]
        else:
            selected = np.zeros((0, n_errors), dtype=bool)
            observed = np.zeros((0, self.num_obs), dtype=bool)
            best_colors = np.zeros(0, dtype=np.uint8)
            selected_weights = np.zeros(0)
            best_slot = np.zeros(0, dtype=int)
            selected_run = np.zeros(0, dtype=np.int8)
            selected_calls = np.zeros(0, dtype=np.int8)
            selected_pairwise_equal = np.zeros(0, dtype=np.int8)
            selected_cache_skips = np.zeros(0, dtype=np.int8)
            gaps = None
        output = observed.ravel() if self.num_obs == 1 else observed
        if check_validity and metric_state is not None:
            valid = np.all(np.asarray((selected.astype(np.uint8) @ manager.H.T) % 2,
                                      dtype=bool) == detector_outcomes, axis=1)
        if metric_state is not None:
            category = run_class[best_class, np.arange(n_shots)] if run_class is not None else None
            return output, metric_state.finish(output, selected_weights, weights, gaps,
                                              logical_classes, run_category=category)
        if compute_swim_distance:
            from ..soft_output.results import GROWTH_CONVENTION
            candidate_swim, class_min_swim = self._score_candidate_swim(
                detector_outcomes, mapped, stage1_hypotheses,
                tuple(spec.target_color for spec in specs), observed, validity)
            selected_swim = candidate_swim[
                best_class, best_slot, np.arange(n_shots)]
        if not full_output:
            return output
        extra = dict(best_colors=best_colors, weights=selected_weights, error_preds=selected,
                     baseline_predictions=_ordinary_baseline_predictions(
                         mapped, generations, manager, logical_classes,
                         self.comparative_decoding, self.num_obs),
                     candidate_labels=tuple(s.label for s in specs),
                     candidate_target_colors=tuple(s.target_color for s in specs),
                     candidate_source_colors=tuple(s.source_colors for s in specs),
                     candidate_anchor_colors=tuple(s.target_color if s.kind == "all_color_relift" else None for s in specs),
                     candidate_kinds=tuple(s.kind for s in specs),
                     candidate_weights=weights, candidate_weight_basis=self.color_correlated_weight_basis,
                     candidate_generation_weights=generations, candidate_stage1_validity=validity,
                     candidate_executed=executed, candidate_alias_of=alias,
                     candidate_native_stage2_preds=native, best_candidate_indices=best_slot,
                     relift_run_class=selected_run, relift_extra_stage2_calls=selected_calls,
                     relift_pairwise_baseline_syndrome_equal=selected_pairwise_equal,
                     relift_cache_skips=selected_cache_skips,
                     relift_pairwise_baseline_syndrome_equal_by_logical_class=pairwise_baseline_syndrome_equal,
                     relift_cache_skips_by_logical_class=cache_skips,
                     relift_run_class_by_logical_class=run_class,
                     relift_extra_stage2_calls_by_logical_class=extra_calls)
        if compute_swim_distance:
            extra.update(candidate_swim_distances=candidate_swim,
                         selected_swim_distance=selected_swim,
                         class_min_swim_distance=class_min_swim,
                         swim_growth_convention=GROWTH_CONVENTION,
                         swim_bound_certified=False)
        if return_candidate_data:
            extra.update(candidate_stage1_hypotheses=stage1_hypotheses,
                         candidate_original_corrections=mapped,
                         candidate_valid=validity)
        if gaps is not None:
            extra.update(logical_gaps=gaps, logical_values=np.asarray(list(itertools.product((False, True), repeat=self.num_obs))))
        if check_validity:
            extra["validity"] = np.all(
                np.asarray((selected.astype(np.uint8) @ manager.H.T) % 2, dtype=bool) == detector_outcomes,
                axis=1,
            )
        return output, extra

    @property
    def _color_correlated_stage2_matchings(self):
        # Compatibility view; these are the shared ordinary fixed matchings.
        return {color: compiled.matching
                for (color, stage), compiled in self._matching_cache.fixed.items()
                if stage == 2}

    def _compiled_stage(self, color, stage, custom_dem_data):
        base = self.dem_manager.dems_decomposed[color]
        if custom_dem_data and color in custom_dem_data:
            prior = custom_dem_data[color][stage - 1]
            if isinstance(prior, StagePrior):
                if prior.compiled is None:
                    prior.compiled = self._matching_cache.get(
                        color, stage, prior.structure, prior.probabilities)
                return prior.compiled
            # Public custom inputs may alias mutable base arrays. Fingerprint
            # their current contents, not just the Python object identity.
            H, p = prior
            return self._matching_cache.get(
                color, stage, StageStructure.prepare(H, stage), p)
        H, p = base.Hs[stage - 1], base.probs[stage - 1]
        structure = self._matching_cache.base_structure(color, stage, H)
        return self._matching_cache.get(color, stage, structure, p, fixed=True)

    def _decode_stage1(
        self,
        detector_outcomes: np.ndarray,
        color: str,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
    ) -> np.ndarray:
        """
        Perform stage 1 decoding for a specific color.

        Stage 1 focuses on local error patterns within each color's subspace.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            2D array of detector outcomes
        color : str
            Color to decode ('r', 'g', or 'b')

        Returns
        -------
        np.ndarray
            Stage 1 error predictions
        """
        compiled = self._compiled_stage(color, 1, custom_dem_data)
        return compiled.matching.decode_batch(
            detector_outcomes[:, compiled.structure.checks_to_keep])

    def _decode_stage2(
        self,
        detector_outcomes: np.ndarray,
        preds_dem1: np.ndarray,
        color: COLOR_LABEL,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
        *, compute_swim_distance: bool = False,
    ):
        """
        Perform stage 2 decoding for a specific color.

        Stage 2 combines stage 1 predictions with remaining detector information
        to find global error patterns.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            2D array of detector outcomes
        preds_dem1 : np.ndarray
            Stage 1 error predictions
        color : COLOR_LABEL
            Color to decode ('r', 'g', or 'b')

        Returns
        -------
        tuple
            (error_predictions, weights) from stage 2 decoding
        """
        if compute_swim_distance and custom_dem_data is not None:
            raise NotImplementedError("Swim output with custom_dem_data is not supported")
        det_outcome_dem2 = detector_outcomes.copy()

        # Mask out detectors not belonging to this color
        mask = self._matching_cache.stage2_detector_mask(
            color, det_outcome_dem2.shape[1], self.dem_manager.detector_ids_by_color[color])
        det_outcome_dem2[:, mask] = False

        # Combine with stage 1 predictions
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1], axis=1)

        if compute_swim_distance:
            from ..soft_output.pymatching_backend import Stage2Backend
            backend = self._swim_backends.get(color)
            if backend is None or not backend.matches(self.dem_manager.dems_decomposed[color]):
                backend = Stage2Backend(self.dem_manager,color)
                self._swim_backends[color] = backend
            return backend.decode(det_outcome_dem2)

        compiled = self._compiled_stage(color, 2, custom_dem_data)
        preds, weights_new = compiled.matching.decode_batch(
            det_outcome_dem2, return_weights=True
        )

        return preds, weights_new

    def _find_error_set_intersection(
        self,
        preds_dem1: Dict[COLOR_LABEL, np.ndarray],
    ) -> np.ndarray:
        """
        Find the intersection of error sets from different colors.

        This method identifies errors that are consistent across all color
        predictions, used in erasure matcher predecoding.

        Parameters
        ----------
        preds_dem1 : dict
            Stage 1 predictions for each color

        Returns
        -------
        np.ndarray
            Boolean array indicating error set intersection
        """
        possible_errors = []
        for c in ["r", "g", "b"]:
            preds_dem1_c = preds_dem1[c]
            error_map_matrix = (
                self.dem_manager.dems_decomposed[c].dems_symbolic[0].error_map_matrix
            )
            possible_errors_c = (preds_dem1_c.astype("uint8") @ error_map_matrix) > 0
            possible_errors.append(possible_errors_c)

        possible_errors = np.stack(possible_errors, axis=-1)
        error_set_intersection = np.all(possible_errors, axis=-1).astype(bool)

        return error_set_intersection

    def _erasure_matcher_predecoding(
        self,
        preds_dem1_all: List[Dict[COLOR_LABEL, np.ndarray]],
        detector_outcomes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform erasure matcher predecoding.

        This advanced predecoding strategy finds error predictions that are
        consistent across all colors and logical classes, providing high-confidence
        corrections before the main decoding stage.

        Parameters
        ----------
        preds_dem1_all : list
            Stage 1 predictions for all logical classes and colors
        detector_outcomes : np.ndarray
            Original detector outcomes

        Returns
        -------
        tuple
            (obs_preds, error_preds, weights, validity) from predecoding
        """
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)

        # Generate all logical value combinations
        all_logical_values = list(itertools.product([False, True], repeat=self.num_obs))
        all_logical_values = np.array(all_logical_values)

        # Calculate error set intersection and weights for each logical class
        error_preds_all = []
        weights_all = []
        for preds_dem1 in preds_dem1_all:
            error_preds = self._find_error_set_intersection(preds_dem1)
            llrs_all = np.log(
                (1 - self.dem_manager.probs_xz) / self.dem_manager.probs_xz
            )
            llrs = np.zeros_like(error_preds, dtype=float)
            llrs[error_preds] = llrs_all[np.where(error_preds)[1]]
            weights = llrs.sum(axis=1)
            error_preds_all.append(error_preds)
            weights_all.append(weights)

        # Stack results
        error_preds_all = np.stack(error_preds_all, axis=1)
        weights_all = np.stack(weights_all, axis=1)
        num_samples = error_preds_all.shape[0]

        # Sort logical classes by prediction weight
        inds_logical_class_sorted = np.argsort(weights_all, axis=1)

        # Sort error predictions and weights by weight
        error_preds_all_sorted = error_preds_all[
            np.arange(num_samples)[:, np.newaxis], inds_logical_class_sorted
        ].astype("uint8")

        weights_all_sorted = np.take_along_axis(
            weights_all, inds_logical_class_sorted, axis=1
        )

        # Check validity (match with detectors and observables)
        match_with_dets = np.all(
            ((error_preds_all_sorted @ self.dem_manager.H.T.toarray()) % 2).astype(bool)
            == detector_outcomes[:, np.newaxis, :],
            axis=-1,
        )

        logical_classes_sorted = all_logical_values[inds_logical_class_sorted]
        match_with_obss = np.all(
            (
                (error_preds_all_sorted @ self.dem_manager.obs_matrix.T.toarray()) % 2
            ).astype(bool)
            == logical_classes_sorted,
            axis=-1,
        )

        validity_full = match_with_dets & match_with_obss

        # Find first valid prediction for each sample
        inds_first_valid_logical_classes = np.argmax(validity_full, axis=1)
        obs_preds = logical_classes_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]
        validity = np.any(validity_full, axis=1)

        # Extract weights and error predictions
        weights = weights_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes
        ]
        weights[~validity] = np.inf

        error_preds = error_preds_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]

        return obs_preds, error_preds, weights, validity
