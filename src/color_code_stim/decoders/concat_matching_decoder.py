"""
Concatenated Matching Decoder for Color Code

This module implements the concatenated minimum-weight perfect matching (MWPM) decoder
for color codes. It performs two-stage decoding with color-based decomposition,
supporting comparative decoding and advanced pre-decoding strategies.
"""

import itertools
from collections import OrderedDict
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
        self._color_correlated_stage1_matchings = OrderedDict()
        self._color_correlated_stage2_matchings = {}
        self._candidate_evaluators = {}
        self.enable_cross_color_relifting = enable_cross_color_relifting
        self.enable_prior_perturbation = enable_prior_perturbation
        if not isinstance(perturbation_ensemble_size, int) or perturbation_ensemble_size < 1:
            raise ValueError("perturbation_ensemble_size must be >= 1")
        if not np.isfinite(perturbation_alpha) or not 0 <= perturbation_alpha <= 1:
            raise ValueError("perturbation_alpha must be between 0 and 1")
        self.perturbation_ensemble_size = perturbation_ensemble_size
        self.perturbation_alpha = perturbation_alpha
        self.perturbation_seed = perturbation_seed
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
            Matching-growth soft output; incompatible with color-correlated
            decoding because its matching state would describe another graph.
        **kwargs
            Additional parameters (for compatibility).

        Returns
        -------
        np.ndarray or tuple
            If full_output is False: predicted observables as bool array.
            If full_output is True: tuple of (predictions, extra_outputs_dict).
        """
        if self.enable_prior_perturbation:
            if custom_dem_data is not None:
                raise NotImplementedError("Prior perturbation with BP/custom DEM priors is unsupported")
            if compute_swim_distance:
                raise NotImplementedError("Prior perturbation with matching-growth swim output is unsupported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding:
                raise NotImplementedError("Prior perturbation with erasure predecoding is unsupported")
            return self._decode_prior_perturbation(
                detector_outcomes, colors, logical_value, full_output, check_validity
            )
        if self.enable_cross_color_relifting:
            if self.dem_manager.remove_non_edge_like_errors:
                raise NotImplementedError("Cross-color relifting requires remove_non_edge_like_errors=False")
            if self.enable_colorcorrelated_decoding:
                raise NotImplementedError("Cross-color relifting and color-correlated decoding cannot be combined")
            if custom_dem_data is not None:
                raise NotImplementedError("Cross-color relifting with BP/custom DEM priors is unsupported")
            if compute_swim_distance:
                raise NotImplementedError("Cross-color relifting with matching-growth swim output is unsupported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding:
                raise NotImplementedError("Cross-color relifting with predecoding ensembles is unsupported")
            return self._decode_cross_color_relifting(
                detector_outcomes, colors, logical_value, full_output, check_validity
            )
        if self.enable_colorcorrelated_decoding:
            if custom_dem_data is not None:
                raise NotImplementedError(
                    "Color-correlated decoding with custom_dem_data/BP is not supported"
                )
            if compute_swim_distance:
                raise NotImplementedError(
                    "Color-correlated decoding with matching-growth swim output is not supported"
                )
        if compute_swim_distance:
            if custom_dem_data is not None:
                raise NotImplementedError("Swim output with custom_dem_data is not supported")
            if erasure_matcher_predecoding or partial_correction_by_predecoding or self.comparative_decoding:
                raise NotImplementedError("Swim output is not validated for predecoding or comparative DEMs")
            if not self.dem_manager.swim_data_only:
                raise NotImplementedError("Swim requires single-round triangular data-only X noise; unresolved boundaries are UNCLASSIFIED")

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
        if self.enable_colorcorrelated_decoding:
            candidate_weights = np.full((num_logical_classes, 12, num_left_samples), np.nan)
            generation_weights = np.full_like(candidate_weights, np.nan)
            candidate_executed = np.zeros(candidate_weights.shape, dtype=bool)
            run_by_class = np.full((num_logical_classes, num_left_samples), -1, dtype=np.int8)
            color_correlated_run = np.full(num_left_samples, -1, dtype=np.int8)
            best_candidate_indices = np.full(num_left_samples, -1, dtype=int)
            native_candidate_preds = [[None] * 12 for _ in range(num_logical_classes)]

        if num_left_samples > 0 and not (
            erasure_matcher_predecoding and partial_correction_by_predecoding
        ):
            num_errors = self.dem_manager.H.shape[1]

            num_candidates = 12 if self.enable_colorcorrelated_decoding else len(colors)
            if self.enable_colorcorrelated_decoding:
                error_preds = np.zeros(
                    (num_logical_classes, num_candidates, num_left_samples, num_errors),
                    dtype=bool,
                )
                weights = np.full(
                    (num_logical_classes, num_candidates, num_left_samples), np.inf
                )
                generation_weights[:, 3:, :] = np.inf
            else:
                error_preds = np.empty(
                    (num_logical_classes, num_candidates, num_left_samples, num_errors),
                    dtype=bool,
                )
                weights = np.empty(
                    (num_logical_classes, num_candidates, num_left_samples), dtype=float
                )

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

                    error_preds[i, i_c, :, :] = error_preds_new
                    weights[i, i_c, :] = weights_new
                    if self.enable_colorcorrelated_decoding:
                        candidate_executed[i, i_c, :] = True

                if self.enable_colorcorrelated_decoding:
                    # Guides are the three baseline original-DEM corrections of
                    # this logical class. The two-guide case is Boolean union.
                    guides = {c: error_preds[i, j] for j, c in enumerate(colors)}
                    schedules = []
                    for shot in range(num_left_samples):
                        category, indices = candidate_schedule(error_preds[i, :3, shot])
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
                                (base_decomp.Hs[0], reweighter.stage1_probabilities(c, guide)),
                                (base_decomp.Hs[1], base_decomp.probs[1]),
                            )}
                            det = detector_outcomes_left[shot:shot + 1].copy()
                            if self.comparative_decoding:
                                det[:, -self.num_obs:] = (
                                    logical_value if logical_value is not None
                                    else all_logical_values[i]
                                )
                            stage1 = self._decode_stage1(det, c, temporary_dem)
                            native, generation_weight = self._decode_stage2(
                                det, stage1, c
                            )
                            (mapped, base_native, selection_weight,
                             diagnostic_weight) = evaluator.evaluate(
                                c, native[0], generation_weight[0]
                            )
                            if native_candidate_preds[i][i_spec] is None:
                                native_candidate_preds[i][i_spec] = np.zeros(
                                    (num_left_samples, base_native.shape[0]), dtype=bool
                                )
                            native_candidate_preds[i][i_spec][shot] = base_native
                            error_preds[i, i_spec, shot] = mapped
                            generation_weights[i, i_spec, shot] = diagnostic_weight
                            candidate_executed[i, i_spec, shot] = True
                            weights[i, i_spec, shot] = selection_weight

            if self.enable_colorcorrelated_decoding:
                candidate_weights = weights.copy()

            # Find best predictions across logical classes and colors
            best_logical_classes, best_color_inds, weights_final, logical_gaps = (
                _get_final_predictions(weights)
            )
            if self.enable_colorcorrelated_decoding:
                best_candidate_indices = best_color_inds.copy()
                color_correlated_run = run_by_class[
                    best_logical_classes, np.arange(num_left_samples)
                ]

            error_preds_final = error_preds[
                best_logical_classes, best_color_inds, np.arange(num_left_samples), :
            ]

            # Calculate observable predictions
            if self.comparative_decoding:
                if logical_value is None:
                    obs_preds_final = all_logical_values[best_logical_classes]
                    if obs_preds_final.shape != (num_left_samples, self.num_obs):
                        raise RuntimeError("Observable prediction shape mismatch")
                else:
                    obs_preds_final = np.tile(logical_value, (num_left_samples, 1))
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

            if compute_swim_distance:
                from ..soft_output.results import GROWTH_CONVENTION
                extra_outputs.update(
                    color_order=tuple(colors),
                    stage2_weights_by_color=swim_stage2_weights.copy(),
                    swim_distances_by_color=swim_by_color,
                    selected_swim_distance=swim_by_color[np.arange(num_left_samples),best_color_inds],
                    swim_growth_convention=GROWTH_CONVENTION,
                    swim_bound_certified=False,
                )

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

    def _decode_prior_perturbation(
        self, detector_outcomes, colors, logical_value, full_output, check_validity
    ):
        """Run each fixed common-DEM ensemble member for all three colors."""
        if colors != "all" and colors != ["r", "g", "b"]:
            raise ValueError("Prior perturbation requires all three colors r, g, b")
        if self._perturbation_ensemble is None:
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
        members = tuple(m for m in range(self.perturbation_ensemble_size) for _ in colors)
        targets = colors * self.perturbation_ensemble_size
        labels = tuple(f"m{m}:{c}" for m in range(self.perturbation_ensemble_size) for c in colors)
        n_classes, n_shots, n_errors = len(logical_classes), len(detector_outcomes), manager.H.shape[1]
        n_candidates = len(labels)
        mapped = np.zeros((n_classes, n_candidates, n_shots, n_errors), dtype=bool)
        stage1_hypotheses = [[None] * n_candidates for _ in range(n_classes)]
        native = [[None] * n_candidates for _ in range(n_classes)]
        weights = np.empty((n_classes, n_candidates, n_shots), dtype=float)
        generations = np.empty_like(weights)
        evaluator = CandidateEvaluator(manager, self.color_correlated_weight_basis)
        for class_index, logical in enumerate(logical_classes):
            det = detector_outcomes.copy()
            if self.comparative_decoding:
                det[:, -self.num_obs:] = logical
            for member in range(self.perturbation_ensemble_size):
                decompositions = ensemble.decompositions[member]
                for color_index, color in enumerate(colors):
                    slot = 3 * member + color_index
                    temporary = decompositions[color]
                    custom = None if member == 0 else {
                        color: tuple(zip(temporary.Hs, temporary.probs))
                    }
                    stage1 = self._decode_stage1(det, color, custom)
                    stage1_hypotheses[class_index][slot] = np.asarray(stage1, dtype=bool).copy()
                    stage2, generation = self._decode_stage2(det, stage1, color, custom)
                    correction, aligned, score, diagnostic = evaluator.evaluate(
                        color, stage2, generation,
                        temporary=None if member == 0 else temporary,
                    )
                    mapped[class_index, slot] = correction
                    native[class_index][slot] = aligned
                    weights[class_index, slot] = score
                    generations[class_index, slot] = diagnostic
        if n_shots:
            best_class, best_slot, selected_weights, gaps = _get_final_predictions(weights)
            selected = mapped[best_class, best_slot, np.arange(n_shots)]
            if self.comparative_decoding:
                observed = np.asarray([logical_classes[i] for i in best_class], dtype=bool)
                if logical_value is not None:
                    observed[:] = logical_value
            else:
                observed = np.asarray((selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool)
            best_colors = np.asarray([color_to_color_val(targets[i]) for i in best_slot], dtype=np.uint8)
        else:
            selected = np.zeros((0, n_errors), dtype=bool)
            observed = np.zeros((0, self.num_obs), dtype=bool)
            selected_weights = np.zeros(0)
            best_slot = np.zeros(0, dtype=int)
            best_colors = np.zeros(0, dtype=np.uint8)
            gaps = None
        output = observed.ravel() if self.num_obs == 1 else observed
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
        self, detector_outcomes, colors, logical_value, full_output, check_validity
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
        mapped = np.zeros((n_classes, 12, n_shots, n_errors), dtype=bool)
        native = [[None] * 12 for _ in range(n_classes)]
        weights = np.full((n_classes, 12, n_shots), np.inf)
        generations = np.full_like(weights, np.nan)
        validity = np.zeros_like(weights, dtype=bool)
        executed = np.zeros_like(weights, dtype=bool)
        alias = np.full(weights.shape, -1, dtype=int)
        run_class = np.full((n_classes, n_shots), -1, dtype=np.int8)
        extra_calls = np.zeros((n_classes, n_shots), dtype=np.int8)
        pairwise_baseline_syndrome_equal = np.zeros((n_classes, n_shots), dtype=np.int8)
        cache_skips = np.zeros((n_classes, n_shots), dtype=np.int8)
        evaluator = CandidateEvaluator(manager, self.color_correlated_weight_basis)

        def store(class_index, slot, shot, result, did_execute=False, alias_slot=-1):
            correction, base_native, weight, generation = result
            mapped[class_index, slot, shot] = correction
            if native[class_index][slot] is None:
                native[class_index][slot] = np.zeros((n_shots, len(base_native)), dtype=bool)
            native[class_index][slot][shot] = base_native
            weights[class_index, slot, shot] = weight
            generations[class_index, slot, shot] = generation
            validity[class_index, slot, shot] = True
            executed[class_index, slot, shot] = did_execute
            alias[class_index, slot, shot] = alias_slot

        for class_index, logical in enumerate(logical_classes):
            det = detector_outcomes.copy()
            if self.comparative_decoding:
                det[:, -self.num_obs:] = logical
            stage1 = {c: self._decode_stage1(det, c) for c in colors}
            stage1 = {c: np.asarray(v, dtype=bool) for c, v in stage1.items()}
            baseline = {}
            for color_index, color in enumerate(colors):
                native_batch, generation_batch = self._decode_stage2(det, stage1[color], color)
                for shot in range(n_shots):
                    evaluated = evaluator.evaluate(color, native_batch[shot], generation_batch[shot])
                    result = tuple(np.asarray(v).copy() for v in evaluated)
                    baseline[color] = baseline.get(color, []) + [result]
                    store(class_index, color_index, shot, result, did_execute=True)
            for shot in range(n_shots):
                sources = {c: baseline[c][shot][0] for c in colors}
                category = relift.classify(sources)
                run_class[class_index, shot] = category
                cache = {}
                for color_index, color in enumerate(colors):
                    syndrome = relift.stage2_syndrome(
                        det[shot], stage1[color][shot], color, manager.detector_ids_by_color
                    )
                    cache[(color, syndrome.tobytes())] = (color_index, baseline[color][shot])
                # Count all six pairwise target problems, including candidates
                # pruned by the baseline multiplicity schedule.
                for spec in specs[3:9]:
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
                            prior = (mapped[class_index, alias_slot, shot],
                                     native[class_index][alias_slot][shot],
                                     weights[class_index, alias_slot, shot],
                                     generations[class_index, alias_slot, shot])
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
                        cache_skips[class_index, shot] += 1
                        continue
                    new_native, new_generation = self._decode_stage2(
                        det[shot:shot + 1], hypothesis[None, :], target
                    )
                    evaluated = evaluator.evaluate(target, new_native[0], new_generation[0])
                    result = tuple(np.asarray(v).copy() for v in evaluated)
                    cache[key] = (slot, result)
                    store(class_index, slot, shot, result, did_execute=True)
                    extra_calls[class_index, shot] += 1

        if n_shots:
            best_class, best_slot, selected_weights, gaps = _get_final_predictions(weights)
            selected = mapped[best_class, best_slot, np.arange(n_shots)]
            if self.comparative_decoding:
                observed = np.asarray([logical_classes[i] for i in best_class], dtype=bool)
                if logical_value is not None:
                    observed[:] = logical_value
            else:
                observed = np.asarray((selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool)
            best_colors = np.asarray([color_to_color_val(specs[i].target_color) for i in best_slot], dtype=np.uint8)
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
        if gaps is not None:
            extra.update(logical_gaps=gaps, logical_values=np.asarray(list(itertools.product((False, True), repeat=self.num_obs))))
        if check_validity:
            extra["validity"] = np.all(
                np.asarray((selected.astype(np.uint8) @ manager.H.T) % 2, dtype=bool) == detector_outcomes,
                axis=1,
            )
        return output, extra

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
        det_outcomes_dem1 = detector_outcomes.copy()

        # Use custom DEM data if provided, otherwise use DEM manager data
        if custom_dem_data and color in custom_dem_data:
            H, p = custom_dem_data[color][0]  # Stage 1 data (H1, p1)
        else:
            H, p = (
                self.dem_manager.dems_decomposed[color].Hs[0],
                self.dem_manager.dems_decomposed[color].probs[0],
            )

        cache_key = None
        if self.enable_colorcorrelated_decoding and custom_dem_data and color in custom_dem_data:
            cache_key = (color, np.asarray(p).tobytes())
            if cache_key in self._color_correlated_stage1_matchings:
                self._color_correlated_stage1_matchings.move_to_end(cache_key)
                checks_to_keep, matching = self._color_correlated_stage1_matchings[cache_key]
                return matching.decode_batch(det_outcomes_dem1[:, checks_to_keep])

        # Remove empty checks
        checks_to_keep = H.tocsr().getnnz(axis=1) > 0
        det_outcomes_dem1 = det_outcomes_dem1[:, checks_to_keep]
        H = H[checks_to_keep, :]

        # MWPM decoding
        weights = np.log((1 - p) / p)
        matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        if cache_key is not None:
            self._color_correlated_stage1_matchings[cache_key] = (checks_to_keep, matching)
            if len(self._color_correlated_stage1_matchings) > 32:
                self._color_correlated_stage1_matchings.popitem(last=False)
        preds_dem1 = matching.decode_batch(det_outcomes_dem1)

        del det_outcomes_dem1, matching
        return preds_dem1

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
        mask = np.full_like(det_outcome_dem2, True)
        mask[:, self.dem_manager.detector_ids_by_color[color]] = False
        det_outcome_dem2[mask] = False
        del mask

        # Combine with stage 1 predictions
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1], axis=1)

        # Stage 2 MWPM decoding
        # Use custom DEM data if provided, otherwise use DEM manager data
        if custom_dem_data and color in custom_dem_data:
            H, p = custom_dem_data[color][1]  # Stage 2 data (H2, p2)
        else:
            H, p = (
                self.dem_manager.dems_decomposed[color].Hs[1],
                self.dem_manager.dems_decomposed[color].probs[1],
            )
        if compute_swim_distance:
            from ..soft_output.pymatching_backend import Stage2Backend
            backend = self._swim_backends.get(color)
            if backend is None or not backend.matches(self.dem_manager.dems_decomposed[color]):
                backend = Stage2Backend(self.dem_manager,color)
                self._swim_backends[color] = backend
            return backend.decode(det_outcome_dem2)

        if self.enable_colorcorrelated_decoding and custom_dem_data is None:
            matching = self._color_correlated_stage2_matchings.get(color)
            if matching is None:
                weights = np.log((1 - p) / p)
                matching = pymatching.Matching.from_check_matrix(H, weights=weights)
                self._color_correlated_stage2_matchings[color] = matching
        else:
            weights = np.log((1 - p) / p)
            matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        preds, weights_new = matching.decode_batch(
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
