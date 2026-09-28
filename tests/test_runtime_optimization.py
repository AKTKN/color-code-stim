"""Independent fresh-build oracles for shot randomness, caches, and retention."""

import itertools
import pickle

import numpy as np
import pymatching
import pytest
import stim

from color_code_stim import ColorCode
from color_code_stim.decoders.color_correlated_decoding import CandidateEvaluator, COLORS
from color_code_stim.decoders.prior_perturbation import PriorPerturbationEnsemble
from color_code_stim.dem_utils.dem_decomp import DemDecomp
from color_code_stim.noise_model import NoiseModel
from color_code_stim.utils import _get_final_predictions
from test_prior_perturbation import code


def assert_exact(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    elif actual is None or isinstance(actual, str):
        assert actual == expected
    else:
        np.testing.assert_array_equal(actual, expected)


def fresh_build_reference(cc, shots):
    """Deliberately rebuild DEMs, decompositions, and matchings for each shot.

    Does not call the optimized sampler, symbolic plans, stage decoder, or cache.
    Keeps all corrections before performing the historical class-major argmin.
    """
    manager = cc.dem_manager
    from color_code_stim.decoders.color_correlated_decoding import OriginalDemProbabilityBuilder, MATCHING_EPS
    builder = OriginalDemProbabilityBuilder(manager)
    rng = np.random.default_rng(cc.perturbation_seed)
    classes = list(itertools.product((False, True), repeat=cc.num_obs)) if cc.comparative_decoding else [None]
    nc, ns, nm = len(classes), len(shots), cc.perturbation_ensemble_size
    shape = (nc, 3 * nm, ns)
    mapped = np.zeros(shape + (manager.H.shape[1],), dtype=bool)
    weights, generations = np.empty(shape), np.empty(shape)
    native = [[None] * (3 * nm) for _ in classes]
    hypotheses = [[None] * (3 * nm) for _ in classes]
    evaluator = CandidateEvaluator(manager, cc.color_correlated_weight_basis)
    for shot in range(ns):
        members = [manager.dems_decomposed]
        for _ in range(1, nm):
            if cc.perturbation_alpha == 0:
                members.append(manager.dems_decomposed)
            else:
                xi = rng.uniform(-1, 1, len(builder.base_q))
                q = np.clip(builder.base_q * (1 + cc.perturbation_alpha * xi), MATCHING_EPS, 1 - MATCHING_EPS)
                dem = builder.build(q)
                members.append({c: DemDecomp(org_dem=dem, color=c,
                    remove_non_edge_like_errors=manager.remove_non_edge_like_errors) for c in COLORS})
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
                    m1 = pymatching.Matching.from_check_matrix(H1[keep], weights=np.log((1-p1)/p1))
                    stage1 = m1.decode_batch(detectors[:, keep])
                    stage2_decomp = manager.dems_decomposed[color] if cc.use_original_prior_for_stage2 else temporary
                    H2, p2 = stage2_decomp.Hs[1], stage2_decomp.probs[1]
                    syndrome = np.zeros_like(detectors)
                    ids = manager.detector_ids_by_color[color]
                    syndrome[:, ids] = detectors[:, ids]
                    syndrome = np.concatenate((syndrome, stage1), axis=1)
                    m2 = pymatching.Matching.from_check_matrix(H2, weights=np.log((1-p2)/p2))
                    correction, generation = m2.decode_batch(syndrome, return_weights=True)
                    org, aligned, score, _ = evaluator.evaluate(color, correction, generation,
                        temporary=temporary if member and not cc.use_original_prior_for_stage2 else None)
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


@pytest.mark.parametrize('comparative', [False, True])
@pytest.mark.parametrize('basis', ['stage2', 'original_dem'])
@pytest.mark.parametrize('original_stage2', [False, True])
def test_all_outputs_equal_fresh_build_reference(comparative, basis, original_stage2):
    cc = code(size=3, alpha=1, comparative=comparative, basis=basis,
              use_original_prior_for_stage2=original_stage2)
    shots, _ = cc.sample(3, seed=53)
    assert_exact(cc.decode(shots, full_output=True), fresh_build_reference(cc, shots))
    assert_exact(code(size=3, alpha=1, comparative=comparative, basis=basis,
        use_original_prior_for_stage2=original_stage2).decode(shots), fresh_build_reference(cc, shots)[0])


@pytest.mark.parametrize('noise', ['bitflip', 'depol', 'uniform'])
@pytest.mark.parametrize('comparative', [False, True])
@pytest.mark.parametrize('remove', [False, True])
def test_symbolic_plan_matches_actual_dem_redecomposition(noise, comparative, remove):
    cc = ColorCode(d=3, rounds=3, comparative_decoding=comparative,
        remove_non_edge_like_errors=remove,
        noise_model=NoiseModel.uniform_circuit_noise(.1) if noise == 'uniform' else NoiseModel(**{noise: .3}))
    manager = cc.dem_manager
    sampler = PriorPerturbationEnsemble(manager, 3, 1, 9)
    for _ in range(2):
        sampler.next_shot()
        for member in (1, 2):
            q = sampler.probabilities[member]
            dem = sampler._builder.build(q)
            for color in COLORS:
                actual = sampler.decompositions[member][color]
                oracle = DemDecomp(org_dem=dem, color=color, remove_non_edge_like_errors=remove)
                assert_exact(actual.probs, oracle.probs)
                for stage in range(2):
                    assert (actual.Hs[stage] != oracle.Hs[stage]).nnz == 0
                    assert (actual.error_map_matrices[stage] != oracle.error_map_matrices[stage]).nnz == 0


@pytest.mark.parametrize('comparative', [False, True])
@pytest.mark.parametrize('full_output', [False, True])
def test_uneven_chunks_empty_batch_single_shot_and_stream(comparative, full_output):
    options = dict(size=3, alpha=.8, comparative=comparative, basis='original_dem')
    whole, split = code(**options), code(**options)
    shots, _ = whole.sample(5, seed=7)
    expected = whole.decode(shots, full_output=full_output)
    pieces = [split.decode(shots[:2], full_output=full_output),
              split.decode(shots[2:2], full_output=full_output),
              split.decode(shots[2], full_output=full_output),
              split.decode(shots[3:], full_output=full_output)]
    assert_exact(whole.concat_matching_decoder._perturbation_ensemble.get_state(),
                 split.concat_matching_decoder._perturbation_ensemble.get_state())
    for a, b in zip(whole.concat_matching_decoder._perturbation_ensemble.probabilities,
                    split.concat_matching_decoder._perturbation_ensemble.probabilities):
        assert_exact(a, b)
    if full_output:
        assert_exact(expected[0], np.concatenate([p[0] for p in pieces]))
        for name, axis in [('weights', 0), ('candidate_weights', 2),
                           ('candidate_original_corrections', 2), ('error_preds', 0),
                           ('candidate_generation_weights', 2), ('best_candidate_indices', 0)]:
            assert_exact(expected[1][name], np.concatenate([p[1][name] for p in pieces], axis=axis))
    else:
        assert_exact(expected, np.concatenate(pieces))


def construction_counter(monkeypatch):
    original = pymatching.Matching.from_check_matrix
    calls = []
    def counted(H, *args, **kwargs):
        calls.append(H.shape)
        return original(H, *args, **kwargs)
    monkeypatch.setattr(pymatching.Matching, 'from_check_matrix', staticmethod(counted))
    return calls


@pytest.mark.parametrize('size,alpha', [(1, .8), (3, 0)])
def test_fixed_graphs_constructed_only_once(monkeypatch, size, alpha):
    cc = code(size=size, alpha=alpha)
    shots, _ = cc.sample(4, seed=8)
    calls = construction_counter(monkeypatch)
    cc.decode(shots)
    assert len(calls) == 6
    cc.decode(shots[:1], full_output=True)
    cc.decode(shots[1:])
    assert len(calls) == 6


@pytest.mark.parametrize('original_stage2', [False, True])
def test_new_priors_rebuild_only_dynamic_graphs_and_bound_cache(monkeypatch, original_stage2):
    cc = code(size=2, alpha=1, use_original_prior_for_stage2=original_stage2)
    shots, _ = cc.sample(40, seed=12)
    calls = construction_counter(monkeypatch)
    cc.decode(shots[:1])
    cache = cc.concat_matching_decoder._matching_cache
    assert len(cache.fixed) == 6
    fixed = tuple(cache.fixed.values())
    before = len(calls)
    cc.decode(shots[1:])
    assert 0 < len(calls) - before <= 39 * (3 if original_stage2 else 6)
    assert tuple(cache.fixed.values()) == fixed
    assert len(cache.dynamic) <= cache.dynamic_limit == 32


def test_custom_prior_mutation_and_parallel_edge_choice_are_not_stale(monkeypatch):
    from scipy.sparse import csc_matrix
    cc = code(enabled=False)
    decoder = cc.concat_matching_decoder
    H = csc_matrix([[1, 1]], dtype=bool)
    p = np.array([.05, .2])
    custom = {'r': ((H, p), (H, p))}
    syndrome = np.ones((1, 1), dtype=bool)
    calls = construction_counter(monkeypatch)
    a = decoder._compiled_stage('r', 2, custom)
    assert_exact(a.matching.decode_batch(syndrome), [[0, 1]])
    p[:] = [.3, .05]
    b = decoder._compiled_stage('r', 2, custom)
    assert b is not a
    assert_exact(b.matching.decode_batch(syndrome), [[1, 0]])
    H[0, 1] = False
    c = decoder._compiled_stage('r', 2, custom)
    assert c is not b
    assert len(calls) == 3


def test_save_load_resumes_rng_without_serializing_matching_cache(tmp_path):
    cc = code(size=3, alpha=1, basis='original_dem')
    shots, _ = cc.sample(5, seed=13)
    cc.decode(shots[:3])
    path = tmp_path / 'active.pkl'
    cc.save(str(path))
    restored = ColorCode.load(str(path))
    assert not restored.concat_matching_decoder._matching_cache.fixed
    assert_exact(cc.decode(shots[3:], full_output=True), restored.decode(shots[3:], full_output=True))
    with path.open('rb') as f:
        saved = pickle.load(f)
    assert '_concat_matching_decoder' not in saved
    assert saved['_prior_perturbation_state']['shot_position'] == 3
    saved.pop('_prior_perturbation_state')
    with path.open('wb') as f:
        pickle.dump(saved, f)
    legacy = ColorCode.load(str(path))
    assert_exact(legacy.decode(shots), code(size=3, alpha=1, basis='original_dem').decode(shots))


@pytest.mark.parametrize('comparative', [False, True])
def test_ties_choose_first_class_and_member_in_fast_and_full_output(monkeypatch, comparative):
    original = CandidateEvaluator.evaluate
    def tied(self, *args, **kwargs):
        correction, native, weight, generation = original(self, *args, **kwargs)
        return correction, native, np.zeros_like(weight), generation
    monkeypatch.setattr(CandidateEvaluator, 'evaluate', tied)
    full, fast = code(size=3, comparative=comparative), code(size=3, comparative=comparative)
    shots, _ = full.sample(5, seed=2)
    prediction, extra = full.decode(shots, full_output=True)
    assert_exact(prediction, fast.decode(shots))
    assert_exact(extra['best_candidate_indices'], np.zeros(5, dtype=int))
    if comparative:
        assert not prediction.any()
        assert_exact(extra['logical_gaps'], np.zeros(5))


@pytest.mark.skipif((np.__version__, stim.__version__, pymatching.__version__) !=
                   ('1.26.4', '1.16.0', '2.2.dev2'),
                   reason='Pristine native fixtures target the audited NumPy/Stim/PyMatching environment')
@pytest.mark.parametrize('mode', ['ordinary_stage2', 'ordinary_original_dem', 'color_correlated', 'relifting', 'M1', 'alpha0'])
@pytest.mark.parametrize('comparative', [False, True])
def test_unaffected_full_outputs_match_prechange_fixture(mode, comparative):
    from pathlib import Path
    # Recorded from pristine decoder commit 072a87d; no new decoder code was
    # involved in generating these expected values or detector inputs.
    path = Path(__file__).parent / 'fixtures' / 'runtime_before_072a87d.npz'
    with np.load(path, allow_pickle=False) as fixture:
        key = mode + ('/comparative' if comparative else '/ordinary')
        shots = fixture[key + '/detectors']
        options = dict(d=3, rounds=3, noise_model=NoiseModel.uniform_circuit_noise(.02),
                       comparative_decoding=comparative)
        if mode.startswith('ordinary_'):
            options['color_correlated_weight_basis'] = mode.removeprefix('ordinary_')
        elif mode == 'color_correlated':
            options.update(enable_colorcorrelated_decoding=True, color_correlated_b=2)
        elif mode in ('M1', 'alpha0'):
            options.update(enable_prior_perturbation=True,
                           perturbation_ensemble_size=1 if mode == 'M1' else 3,
                           perturbation_alpha=1. if mode == 'M1' else 0.,
                           perturbation_seed=19, color_correlated_weight_basis='original_dem')
        else:
            options.update(enable_cross_color_relifting=True, remove_non_edge_like_errors=False)
        cc = ColorCode(**options)
        actual = cc.decode(shots, full_output=True, return_candidate_data=True, check_validity=True)
        def compare(prefix, value):
            if isinstance(value, dict):
                names = {f[len(prefix) + 1:].split('/')[0] for f in fixture.files if f.startswith(prefix + '/')}
                assert names == set(value)
                for name, entry in value.items():
                    compare(prefix + '/' + name, entry)
            elif isinstance(value, (tuple, list)):
                assert fixture[prefix + '/length'] == len(value)
                for i, entry in enumerate(value):
                    compare(prefix + '/' + str(i), entry)
            elif value is None:
                assert fixture[prefix] == '__NONE__'
            else:
                assert_exact(value, fixture[prefix])
        compare(key + '/output', actual)
        assert_exact(cc.decode(shots), actual[0])
        assert_exact(cc.decode(shots[0]), actual[0][:1])
        assert_exact(cc.decode(shots, check_validity=True), actual[0])


def test_identical_dynamic_draws_deduplicate_base_graphs(monkeypatch):
    cc = code(size=3, alpha=1)
    shots, _ = cc.sample(2, seed=8)
    cc.decode(shots[:1])
    class ZeroDraws:
        def uniform(self, low, high, size):
            return np.zeros(size)
    cc.concat_matching_decoder._perturbation_ensemble._rng = ZeroDraws()
    calls = construction_counter(monkeypatch)
    cc.decode(shots)
    assert not calls


def test_comparative_hypotheses_reuse_this_shots_dynamic_matchings(monkeypatch):
    cc = code(size=2, alpha=1, comparative=True)
    shots, _ = cc.sample(1, seed=8)
    calls = construction_counter(monkeypatch)
    cc.decode(shots)
    assert len(calls) == 12  # six base + six dynamic, shared by both classes


def test_hard_output_does_not_allocate_all_original_corrections(monkeypatch):
    cc, full = code(size=3), code(size=3)
    shots, _ = cc.sample(5, seed=18)
    unwanted = (1, 9, 5, cc.dem_manager.H.shape[1])
    allocations = []
    for name in ('zeros', 'empty'):
        original = getattr(np, name)
        def tracked(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple):
                allocations.append(shape)
            return _original(shape, *args, **kwargs)
        monkeypatch.setattr(np, name, tracked)
    hard = cc.decode(shots)
    assert unwanted not in allocations
    allocations.clear()
    pred, _ = full.decode(shots, full_output=True)
    assert unwanted in allocations
    assert_exact(hard, pred)


def test_identical_syndromes_receive_fresh_shot_major_draws():
    from color_code_stim.decoders.color_correlated_decoding import MATCHING_EPS
    cc = code(size=3, alpha=.8)
    first, _ = cc.sample(1, seed=16)
    shots = np.repeat(first, 4, axis=0)
    decoder = cc.concat_matching_decoder
    sampler = PriorPerturbationEnsemble(cc.dem_manager, 3, .8, 19)
    decoder._perturbation_ensemble = sampler
    captured = []
    next_shot = sampler.next_shot
    def capture():
        result = next_shot()
        captured.append([q.copy() for q in sampler.probabilities])
        return result
    sampler.next_shot = capture
    cc.decode(shots[:2])
    cc.decode(shots[2:])
    rng = np.random.default_rng(19)
    q0 = cc.dem_manager.probs_xz
    for ensemble in captured:
        assert_exact(ensemble[0], q0)
        for q in ensemble[1:]:
            assert_exact(q, np.clip(q0 * (1 + .8 * rng.uniform(-1, 1, len(q0))),
                                   MATCHING_EPS, 1 - MATCHING_EPS))
    assert len(captured) == 4
    assert not np.array_equal(captured[0][1], captured[1][1])


def test_guided_stage1_cache_is_bounded_and_exact_repeats_hit(monkeypatch):
    cc = ColorCode(d=3, rounds=3, noise_model=NoiseModel.uniform_circuit_noise(.02),
                  enable_colorcorrelated_decoding=True, color_correlated_b=2)
    decoder = cc.concat_matching_decoder
    base = cc.dem_manager.dems_decomposed['r']
    rng = np.random.default_rng(18)
    for _ in range(40):
        p = rng.uniform(.001, .4, len(base.probs[0]))
        custom = {'r': ((base.Hs[0], p), (base.Hs[1], base.probs[1]))}
        decoder._compiled_stage('r', 1, custom)
    assert len(decoder._matching_cache.dynamic) == 32
    calls = construction_counter(monkeypatch)
    decoder._compiled_stage('r', 1, custom)
    assert not calls


@pytest.mark.parametrize('partial', [False, True])
def test_predecoding_full_and_hard_equivalence(partial):
    cc = ColorCode(d=3, rounds=3, noise_model=NoiseModel.uniform_circuit_noise(.02),
                  comparative_decoding=True)
    shots, _ = cc.sample(24, seed=51)
    options = dict(erasure_matcher_predecoding=True, partial_correction_by_predecoding=partial)
    prediction, _ = cc.decode(shots, full_output=True, **options)
    assert_exact(prediction, cc.decode(shots, **options))



def test_public_custom_inputs_aliasing_base_are_fingerprinted():
    cc = code(enabled=False)
    decoder = cc.concat_matching_decoder
    base = cc.dem_manager.dems_decomposed['r']
    H, p = base.Hs[1], base.probs[1]
    custom = {'r': ((base.Hs[0], base.probs[0]), (H, p))}
    before = decoder._compiled_stage('r', 2, custom)
    original_p, original_H = p.copy(), H.copy()
    try:
        p[:] *= .5
        after_p = decoder._compiled_stage('r', 2, custom)
        assert after_p is not before
        H.data[0] = False
        after_H = decoder._compiled_stage('r', 2, custom)
        assert after_H is not after_p
        syndrome = H[:, :1].toarray().T.astype(bool)
        fresh = pymatching.Matching.from_check_matrix(H, weights=np.log((1-p)/p))
        assert_exact(after_H.matching.decode_batch(syndrome, return_weights=True),
                     fresh.decode_batch(syndrome, return_weights=True))
    finally:
        p[:] = original_p
        # PyMatching eliminates explicit zero entries in its input matrix.
        H.data = original_H.data
        H.indices = original_H.indices
        H.indptr = original_H.indptr
