import pickle
import numpy as np
import pymatching
import pytest

from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel
from test_prior_perturbation import code
from test_runtime_optimization import assert_exact
from native_stage1_reference import fresh_stage1_reference


@pytest.mark.parametrize('comparative', [False, True])
@pytest.mark.parametrize('basis', ['stage2', 'original_dem'])
@pytest.mark.parametrize('circuit', [False, True])
def test_all_candidates_match_independent_fresh_pipeline(comparative, basis, circuit):
    options = dict(d=3, rounds=3 if circuit else 1,
                   noise_model=NoiseModel.uniform_circuit_noise(.001) if circuit else NoiseModel(bitflip=.05),
                   stage1_perturbation=True, perturbation_ensemble_size=4,
                   perturbation_alpha=1, perturbation_seed=19,
                   comparative_decoding=comparative, color_correlated_weight_basis=basis)
    cc = ColorCode(**options)
    shots, _ = cc.sample(6, seed=51)
    expected = fresh_stage1_reference(cc, shots, offset=15)
    actual = cc.decode(shots, full_output=True, perturbation_shot_offset=15)
    assert_exact(actual, expected)
    assert_exact(cc.decode(shots, perturbation_shot_offset=15), actual[0])
    assert cc.enable_prior_perturbation and cc.use_original_prior_for_stage2


@pytest.mark.parametrize('comparative', [False, True])
@pytest.mark.parametrize('size,alpha', [(1, 1), (4, 0)])
@pytest.mark.parametrize('basis', ['stage2', 'original_dem'])
def test_unperturbed_modes_exactly_match_current_implementation(comparative, size, alpha, basis):
    ordinary = code(size=size, alpha=alpha, comparative=comparative, basis=basis,
                    use_original_prior_for_stage2=True)
    native = code(size=size, alpha=alpha, comparative=comparative, basis=basis,
                  stage1_perturbation=True)
    shots, _ = ordinary.sample(13, seed=61)
    assert_exact(native.decode(shots, full_output=True), ordinary.decode(shots, full_output=True))


def test_chunk_offsets_cache_and_no_reconstruction(monkeypatch):
    whole = code(size=4, alpha=1, stage1_perturbation=True)
    shots, _ = whole.sample(11, seed=63)
    expected = whole.decode(shots, full_output=True)
    split = code(size=4, alpha=1, stage1_perturbation=True)
    parts = [split.decode(p, full_output=True) for p in (shots[:3], shots[3:3], shots[3:])]
    assert_exact(expected[0], np.concatenate([p[0] for p in parts]))
    for key in ('candidate_weights', 'candidate_generation_weights', 'candidate_original_corrections'):
        assert_exact(expected[1][key], np.concatenate([p[1][key] for p in parts], axis=2))
    assert split.concat_matching_decoder._perturbation_ensemble.shot_position == 11
    fresh = code(size=4, alpha=1, stage1_perturbation=True)
    suffix = fresh.decode(shots[3:], full_output=True, perturbation_shot_offset=3)
    assert_exact(suffix[1]['candidate_weights'], expected[1]['candidate_weights'][:, :, 3:])
    ensemble = whole.concat_matching_decoder._perturbation_ensemble
    builds = {c: m._matching_graph.native_mwpm_build_count for c, m in ensemble.matchings.items()}
    assert set(builds.values()) == {2}
    def forbidden(*args, **kwargs):
        raise AssertionError('Warm decoding rebuilt a matching')
    monkeypatch.setattr(pymatching.Matching, 'from_check_matrix', forbidden)
    for _ in range(3):
        whole.decode(shots, perturbation_shot_offset=0)
    assert {c: m._matching_graph.native_mwpm_build_count for c, m in ensemble.matchings.items()} == builds
    cache = whole.concat_matching_decoder._matching_cache
    assert set(cache.fixed) == {(c, 2) for c in 'rgb'}
    assert not cache.dynamic
    assert all(v.matching._matching_graph.native_mwpm_build_count == 1 for v in cache.fixed.values())


def test_comparative_classes_replay_same_draws():
    cc = code(size=3, alpha=1, comparative=True, stage1_perturbation=True)
    shots, _ = cc.sample(5, seed=71)
    _, combined = cc.decode(shots, full_output=True, perturbation_shot_offset=4)
    for logical in (False, True):
        isolated = code(size=3, alpha=1, comparative=True, stage1_perturbation=True)
        _, extra = isolated.decode(shots, logical_value=[logical], full_output=True, perturbation_shot_offset=4)
        assert_exact(extra['candidate_weights'][0], combined['candidate_weights'][int(logical)])
        assert_exact(extra['candidate_original_corrections'][0], combined['candidate_original_corrections'][int(logical)])


def test_save_restore_resolved_seed_empty_and_legacy(tmp_path):
    cc = code(size=3, alpha=1, seed=None, stage1_perturbation=True)
    seed = cc.perturbation_seed
    assert type(seed) is int
    shots, _ = cc.sample(9, seed=73)
    cc.decode(shots[:4])
    path = tmp_path/'native.pkl'
    cc.save(path)
    restored = ColorCode.load(path)
    assert restored.stage1_perturbation and restored.perturbation_seed == seed
    assert_exact(restored.decode(shots[4:], full_output=True), cc.decode(shots[4:], full_output=True))
    state = restored.concat_matching_decoder._perturbation_ensemble.get_state()
    restored.decode(shots[:0], full_output=True)
    assert restored.concat_matching_decoder._perturbation_ensemble.get_state() == state
    with path.open('rb') as stream:
        saved = pickle.load(stream)
    saved.pop('stage1_perturbation')
    saved.pop('_prior_perturbation_state')
    with path.open('wb') as stream:
        pickle.dump(saved, stream)
    assert not ColorCode.load(path).stage1_perturbation


def test_swim_validity_and_stage1_syndromes():
    cc = code(size=3, alpha=.7, stage1_perturbation=True)
    shots, _ = cc.sample(8, seed=79)
    actual, extra = cc.decode(shots, full_output=True, check_validity=True, compute_swim_distance=True)
    assert extra['validity'].all()
    assert_exact(actual, cc.decode(shots, perturbation_shot_offset=0))
    for slot, color in enumerate(extra['candidate_target_colors']):
        h = cc.dem_manager.dems_decomposed[color].Hs[0]
        hypotheses = extra['candidate_stage1_hypotheses'][0][slot]
        keep = h.tocsr().getnnz(axis=1) > 0
        assert_exact(np.asarray((hypotheses.astype(np.uint8) @ h[keep].T) % 2, dtype=bool), shots[:, keep])
    assert np.isfinite(extra['selected_swim_distance']).all()


def test_native_options_and_missing_backend(monkeypatch):
    with pytest.raises(ValueError, match='boolean'):
        code(stage1_perturbation='yes')
    with pytest.raises(NotImplementedError):
        code(stage1_perturbation=True, enable_colorcorrelated_decoding=True)
    with pytest.raises(ValueError, match='shot_offset'):
        code().decode([[False]], perturbation_shot_offset=0)
    monkeypatch.setattr(pymatching.Matching, 'NATIVE_PERTURBATION_VERSION', 0)
    with pytest.raises(RuntimeError, match='PyMatching fork'):
        code(stage1_perturbation=True)
    ordinary = code(enabled=False)
    shots, _ = ordinary.sample(1, seed=81)
    ordinary.decode(shots)
