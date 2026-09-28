"""Slow full-pipeline oracle: fresh check-matrix matchings for each shot/member."""
import itertools
import numpy as np
import pymatching
from color_code_stim.decoders.color_correlated_decoding import CandidateEvaluator, COLORS
from color_code_stim.utils import _get_final_predictions


def fresh_stage1_reference(cc, shots, offset=0):
    """Rebuild matching stages without the stage decoder or matching cache."""
    manager = cc.dem_manager
    # The sampler is independently verified in PyMatching's Python MT64 tests.
    # This oracle rebuilds both matching stages and independently selects outputs.
    ensemble = cc.concat_matching_decoder._perturbation_ensemble
    if ensemble is None:
        from color_code_stim.decoders.native_stage1_perturbation import NativeStage1Ensemble
        decoder = cc.concat_matching_decoder
        ensemble = NativeStage1Ensemble(manager, cc.perturbation_ensemble_size,
                                       cc.perturbation_alpha, cc.perturbation_seed, decoder._matching_cache)
    classes = list(itertools.product((False, True), repeat=cc.num_obs)) if cc.comparative_decoding else [None]
    nc, ns, nm = len(classes), len(shots), cc.perturbation_ensemble_size
    shape = (nc, 3 * nm, ns)
    mapped = np.zeros(shape + (manager.H.shape[1],), dtype=bool)
    weights, generations = np.empty(shape), np.empty(shape)
    native = [[None] * (3 * nm) for _ in classes]
    hypotheses = [[None] * (3 * nm) for _ in classes]
    evaluator = CandidateEvaluator(manager, cc.color_correlated_weight_basis)
    for shot in range(ns):
        draws = {c: ensemble.matchings[c]._perturbation_weights_for_shot(offset+shot) for c in COLORS}
        members = [manager.dems_decomposed] * nm
        for ic, logical in enumerate(classes):
            detectors = shots[shot:shot + 1].copy()
            if cc.comparative_decoding:
                detectors[:, -cc.num_obs:] = logical
            for member, decompositions in enumerate(members):
                for color_id, color in enumerate(COLORS):
                    slot = 3 * member + color_id
                    temporary = decompositions[color]
                    H1, p1 = temporary.Hs[0], temporary.probs[0]
                    keep = H1.tocsr().getnnz(axis=1) > 0
                    m1 = pymatching.Matching.from_check_matrix(H1[keep], weights=draws[color][member])
                    stage1 = m1.decode_batch(detectors[:, keep])
                    stage2_decomp = manager.dems_decomposed[color]
                    H2, p2 = stage2_decomp.Hs[1], stage2_decomp.probs[1]
                    syndrome = np.zeros_like(detectors)
                    ids = manager.detector_ids_by_color[color]
                    syndrome[:, ids] = detectors[:, ids]
                    syndrome = np.concatenate((syndrome, stage1), axis=1)
                    m2 = pymatching.Matching.from_check_matrix(H2, weights=np.log((1-p2)/p2))
                    correction, generation = m2.decode_batch(syndrome, return_weights=True)
                    org, aligned, score, _ = evaluator.evaluate(color, correction, generation,
                        temporary=None)
                    mapped[ic, slot, shot] = org[0]
                    weights[ic, slot, shot], generations[ic, slot, shot] = score[0], generation[0]
                    for container, value in ((native, aligned), (hypotheses, stage1)):
                        if container[ic][slot] is None:
                            container[ic][slot] = np.zeros((ns, value.shape[1]), dtype=bool)
                        container[ic][slot][shot] = value[0]
    bc, bs, selected_weights, gaps = _get_final_predictions(weights)
    selected = mapped[bc, bs, np.arange(ns)]
    observable = (np.asarray([classes[i] for i in bc], dtype=bool) if cc.comparative_decoding else
                  np.asarray((selected.astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool))
    baseline_class, baseline_slot, _, _ = _get_final_predictions(generations[:, :3])
    baseline = (np.asarray([classes[i] for i in baseline_class], dtype=bool) if cc.comparative_decoding else
                np.asarray((mapped[baseline_class, baseline_slot, np.arange(ns)].astype(np.uint8) @ manager.obs_matrix.T) % 2, dtype=bool))
    extra = dict(weights=selected_weights, error_preds=selected,
        best_colors=np.asarray(bs % 3, dtype=np.uint8),
        baseline_predictions=baseline.ravel() if cc.num_obs == 1 else baseline,
        candidate_labels=tuple(f'm{m}:{c}' for m in range(nm) for c in COLORS),
        candidate_target_colors=COLORS * nm,
        candidate_ensemble_members=tuple(m for m in range(nm) for _ in COLORS),
        candidate_weights=weights, candidate_weight_basis=cc.color_correlated_weight_basis,
        candidate_generation_weights=generations, candidate_native_stage2_preds=native,
        best_candidate_indices=bs, candidate_stage1_hypotheses=hypotheses,
        candidate_original_corrections=mapped)
    if gaps is not None:
        extra.update(logical_gaps=gaps, logical_values=np.asarray(classes))
    return observable.ravel() if cc.num_obs == 1 else observable, extra
