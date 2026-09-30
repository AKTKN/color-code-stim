"""Common X/Z DEM prior ensemble invariants."""

import numpy as np
import pytest

from color_code_stim import ColorCode
from color_code_stim.decoders.color_correlated_decoding import CandidateEvaluator
from color_code_stim.noise_model import NoiseModel


def code(*, size=2, alpha=.35, seed=19, comparative=False, basis="stage2", enabled=True, **kwargs):
    return ColorCode(
        d=3, rounds=1, noise_model=NoiseModel(bitflip=.05),
        comparative_decoding=comparative, enable_prior_perturbation=enabled,
        perturbation_ensemble_size=size, perturbation_alpha=alpha,
        perturbation_seed=seed, color_correlated_weight_basis=basis, **kwargs,
    )


def test_baseline_member_and_zero_alpha():
    baseline = code(enabled=False)
    shots, _ = baseline.sample(8, seed=31)
    expected, old = baseline.decode(shots, full_output=True)
    for size, alpha in ((1, .7), (3, 0)):
        actual, out = code(size=size, alpha=alpha).decode(shots, full_output=True)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(out["error_preds"], old["error_preds"])
        np.testing.assert_allclose(out["candidate_weights"][0, :3].min(axis=0), old["weights"], atol=1e-10)
        for member in range(1, size):
            np.testing.assert_array_equal(out["candidate_weights"][0, 3*member:3*member+3],
                                          out["candidate_weights"][0, :3])


def test_per_shot_common_prior_preserves_dem_and_base():
    cc = code(size=3)
    manager = cc.dem_manager
    original_dem = str(manager.dem_xz)
    original_q = manager.probs_xz.copy()
    shots, _ = cc.sample(3, seed=4)
    cc.decode(shots)
    ensemble = cc.concat_matching_decoder._perturbation_ensemble
    same = code(size=3).concat_matching_decoder
    # Same initial seed and shot positions reproduce the per-shot stream.
    same.decode(shots)
    for q1, q2 in zip(ensemble.probabilities, same._perturbation_ensemble.probabilities):
        np.testing.assert_array_equal(q1, q2)
    other = code(size=3, seed=20).concat_matching_decoder
    other.decode(shots)
    assert any(not np.array_equal(a, b) for a, b in zip(
        ensemble.probabilities[1:], other._perturbation_ensemble.probabilities[1:]))
    previous = [q.copy() for q in ensemble.probabilities]
    cc.decode(shots[:1])
    assert ensemble.shot_position == 4
    assert any(not np.array_equal(a, b) for a, b in zip(previous[1:], ensemble.probabilities[1:]))
    for member in (1, 2):
        q = ensemble.probabilities[member]
        dem = ensemble._builder.build(q)
        base_instructions = list(manager.dem_xz.flattened())
        temporary_instructions = list(dem)
        assert len(base_instructions) == len(temporary_instructions)
        for base, temp in zip(base_instructions, temporary_instructions):
            assert base.type == temp.type
            assert base.targets_copy() == temp.targets_copy()
            if base.type != "error":
                assert str(base) == str(temp)
        for color in "rgb":
            np.testing.assert_array_equal(ensemble.decompositions[member][color].org_prob, q)
    assert str(manager.dem_xz) == original_dem
    np.testing.assert_array_equal(manager.probs_xz, original_q)


@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("basis", ["stage2", "original_dem"])
def test_candidates_calls_batches_and_base_scoring(monkeypatch, comparative, basis):
    cc = code(size=2, comparative=comparative, basis=basis)
    shots, _ = cc.sample(6, seed=11)
    decoder = cc.concat_matching_decoder
    calls = [0, 0]
    old1, old2 = decoder._decode_stage1, decoder._decode_stage2

    def stage1(*args, **kwargs):
        calls[0] += 1
        return old1(*args, **kwargs)

    def stage2(*args, **kwargs):
        calls[1] += 1
        return old2(*args, **kwargs)

    monkeypatch.setattr(decoder, "_decode_stage1", stage1)
    monkeypatch.setattr(decoder, "_decode_stage2", stage2)
    pred, out = cc.decode(shots, full_output=True)
    n_classes = 2 if comparative else 1
    assert calls == [6*n_classes*len(shots), 6*n_classes*len(shots)]
    assert out["candidate_weights"].shape == (n_classes, 6, 6)
    assert out["candidate_ensemble_members"] == (0, 0, 0, 1, 1, 1)
    evaluator = CandidateEvaluator(cc.dem_manager, basis)
    for cls, row in enumerate(out["candidate_native_stage2_preds"]):
        for slot, aligned in enumerate(row):
            color = out["candidate_target_colors"][slot]
            scored = evaluator.evaluate(color, aligned, out["candidate_generation_weights"][cls, slot])[2]
            np.testing.assert_allclose(scored, out["candidate_weights"][cls, slot])
    split = code(size=2, comparative=comparative, basis=basis)
    pieces = [split.decode(part, full_output=True) for part in (shots[:2], shots[2:])]
    np.testing.assert_array_equal(pred, np.concatenate([p[0] for p in pieces]))
    np.testing.assert_allclose(out["weights"], np.concatenate([p[1]["weights"] for p in pieces]))
    if comparative:
        minima = out["candidate_weights"].min(axis=1)
        np.testing.assert_allclose(out["logical_gaps"], np.abs(minima[0] - minima[1]))


def test_generation_weights_do_not_select(monkeypatch):
    cc = code(size=2)
    shots, _ = cc.sample(4, seed=5)
    decoder = cc.concat_matching_decoder
    original = decoder._decode_stage2

    def changed(*args, **kwargs):
        native, weights = original(*args, **kwargs)
        return native, np.full_like(weights, -1e9 if args[3] is not None else 1e9)

    monkeypatch.setattr(decoder, "_decode_stage2", changed)
    _, out = cc.decode(shots, full_output=True)
    np.testing.assert_allclose(out["weights"], out["candidate_weights"].min(axis=(0, 1)))
    assert np.any(out["candidate_generation_weights"] != out["candidate_weights"])


def test_temporary_alignment_and_comparative_class_locality(monkeypatch):
    cc = code(size=2, comparative=True)
    shots, _ = cc.sample(3, seed=22)
    seen = []
    original = CandidateEvaluator.evaluate

    def checked(self, color, native, generation_weight, temporary=None):
        result = original(self, color, native, generation_weight, temporary)
        if temporary is not None:
            expected = temporary.map_errors_to_org_dem(native, stage=2)
            np.testing.assert_array_equal(result[0], expected)
            base = cc.dem_manager.dems_decomposed[color]
            np.testing.assert_array_equal(base.map_errors_to_org_dem(result[1], stage=2), expected)
            seen.append(color)
        return result

    monkeypatch.setattr(CandidateEvaluator, "evaluate", checked)
    _, combined = cc.decode(shots, full_output=True)
    assert seen == list("rgb") * (2 * len(shots))
    for cls, logical in enumerate((False, True)):
        _, isolated = code(size=2, comparative=True).decode(
            shots, logical_value=[logical], full_output=True)
        np.testing.assert_array_equal(combined["candidate_weights"][cls],
                                      isolated["candidate_weights"][0])


def test_persistence_and_incompatibilities(tmp_path):
    cc = code(size=2)
    path = tmp_path / "code.pkl"
    cc.save(str(path))
    restored = ColorCode.load(str(path))
    assert (restored.enable_prior_perturbation, restored.perturbation_ensemble_size,
            restored.perturbation_alpha, restored.perturbation_seed) == (True, 2, .35, 19)
    shots, _ = cc.sample(2, seed=9)
    np.testing.assert_array_equal(cc.decode(shots), restored.decode(shots))
    with pytest.raises(ValueError, match="ensemble_size"):
        code(size=0)
    with pytest.raises(ValueError, match="alpha"):
        code(alpha=1.1)
    with pytest.raises(NotImplementedError, match="color-correlated"):
        code(enable_colorcorrelated_decoding=True)
    with pytest.raises(NotImplementedError, match="relifting"):
        code(enable_cross_color_relifting=True, remove_non_edge_like_errors=False)
    bp_prediction, bp_extra = cc.decode(shots, bp_predecoding=True, metrics=["weights"])
    assert bp_prediction.shape == (len(shots),)
    assert bp_extra["bp_converged"].dtype == bool
    np.testing.assert_array_equal(np.ma.getmaskarray(bp_extra["weights"]), bp_extra["bp_converged"])
    _, scored = cc.decode(shots, compute_swim_distance=True, full_output=True)
    assert np.isfinite(scored["selected_swim_distance"]).all()
    with pytest.raises(NotImplementedError, match="custom DEM"):
        cc.concat_matching_decoder.decode(shots, custom_dem_data={})


def test_swim_uses_base_stage2_prior_and_selected_logical_class():
    cc = code(size=2, alpha=.7)
    shots, _ = cc.sample(8, seed=107)
    hard, extra = cc.decode(shots, full_output=True, compute_swim_distance=True)
    off = code(size=2, alpha=.7).decode(shots)
    np.testing.assert_array_equal(hard, off)
    decoder = cc.concat_matching_decoder
    for slot, color in enumerate(extra["candidate_target_colors"]):
        stage1 = extra["candidate_stage1_hypotheses"][0][slot]
        direct = decoder._decode_stage2(
            shots, stage1, color, compute_swim_distance=True)
        np.testing.assert_array_equal(
            direct.swim_distances, extra["candidate_swim_distances"][0, slot])
    observable = np.asarray((extra["candidate_original_corrections"][0].astype(
        np.uint8).reshape(-1, cc.dem_manager.H.shape[1]) @
        cc.dem_manager.obs_matrix.T) % 2, dtype=bool).reshape(6, 8)
    same = observable == hard[None, :]
    expected = np.min(np.where(same, extra["candidate_swim_distances"][0], np.inf), axis=0)
    np.testing.assert_array_equal(expected, extra["class_min_swim_distance"])
    assert np.all(expected <= extra["selected_swim_distance"] + 1e-12)


@pytest.mark.parametrize("original_stage2", [False, True])
@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("basis", ["stage2", "original_dem"])
def test_stage2_prior_switch(monkeypatch, original_stage2, comparative, basis):
    cc = code(size=3, alpha=.8, comparative=comparative, basis=basis,
              use_original_prior_for_stage2=original_stage2)
    shots, _ = cc.sample(16, seed=103)
    decoder = cc.concat_matching_decoder
    old1, old2 = decoder._decode_stage1, decoder._decode_stage2
    stage1_calls, stage2_calls = [], []

    def stage1(det, color, custom):
        stage1_calls.append(custom)
        return old1(det, color, custom)

    def stage2(det, hypothesis, color, custom):
        stage2_calls.append(custom)
        result = old2(det, hypothesis, color, custom)
        if original_stage2:
            expected = old2(det, hypothesis, color)
            np.testing.assert_array_equal(result[0], expected[0])
            np.testing.assert_array_equal(result[1], expected[1])
        return result

    monkeypatch.setattr(decoder, "_decode_stage1", stage1)
    monkeypatch.setattr(decoder, "_decode_stage2", stage2)
    prediction, extra = cc.decode(shots, full_output=True, check_validity=True)
    # Comparative DEMs append the logical constraint to the detector rows;
    # check against the selected class, which can differ from the sampled class.
    expected_syndrome = shots.copy()
    if comparative:
        expected_syndrome[:, -cc.num_obs:] = np.asarray(prediction).reshape(-1, cc.num_obs)
    mapped = extra["error_preds"]
    np.testing.assert_array_equal(
        np.asarray((mapped.astype(np.uint8) @ cc.dem_manager.H.T) % 2, dtype=bool),
        expected_syndrome)
    assert len(stage1_calls) == len(stage2_calls) == len(shots) * (18 if comparative else 9)
    assert stage1_calls[3] is not None  # stage 1 still uses perturbed priors
    for index, (custom1, custom2) in enumerate(zip(stage1_calls, stage2_calls)):
        if original_stage2 or index % 9 < 3:
            assert custom2 is None
        else:
            assert custom2 is custom1
    evaluator = CandidateEvaluator(cc.dem_manager, basis)
    for cls, row in enumerate(extra["candidate_native_stage2_preds"]):
        for slot, native in enumerate(row):
            color = extra["candidate_target_colors"][slot]
            score = evaluator.evaluate(color, native, 0)[2]
            np.testing.assert_allclose(score, extra["candidate_weights"][cls, slot])
    split = code(size=3, alpha=.8, comparative=comparative, basis=basis,
                 use_original_prior_for_stage2=original_stage2)
    pieces = [split.decode(part) for part in (shots[:5], shots[5:])]
    np.testing.assert_array_equal(prediction, np.concatenate(pieces))


@pytest.mark.parametrize("original_stage2", [False, True])
def test_stage2_switch_baseline_and_persistence(original_stage2, tmp_path):
    baseline = code(enabled=False)
    shots, _ = baseline.sample(8, seed=31)
    for size, alpha in ((1, .7), (3, 0)):
        cc = code(size=size, alpha=alpha, use_original_prior_for_stage2=original_stage2)
        np.testing.assert_array_equal(cc.decode(shots), baseline.decode(shots))
    cc = code(use_original_prior_for_stage2=original_stage2)
    path = tmp_path / "prior.pkl"
    cc.save(str(path))
    restored = ColorCode.load(str(path))
    assert restored.use_original_prior_for_stage2 is original_stage2
    assert restored.concat_matching_decoder.use_original_prior_for_stage2 is original_stage2
    np.testing.assert_array_equal(cc.decode(shots), restored.decode(shots))


def test_stage2_switch_requires_boolean():
    with pytest.raises(ValueError, match="use_original_prior_for_stage2"):
        code(use_original_prior_for_stage2="false")


@pytest.mark.parametrize("noise", ["bitflip", "depol", "uniform"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_original_stage2_circuit_corrections_and_legacy_false(noise, rounds):
    noise_model = (NoiseModel.uniform_circuit_noise(.003) if noise == "uniform" else
                   NoiseModel(**{noise: .03}))
    common = dict(d=3, rounds=rounds, noise_model=noise_model,
                  enable_prior_perturbation=True, perturbation_ensemble_size=3,
                  perturbation_alpha=1.0, perturbation_seed=17,
                  color_correlated_weight_basis="original_dem")
    legacy = ColorCode(**common)
    explicit_false = ColorCode(**common, use_original_prior_for_stage2=False)
    original = ColorCode(**common, use_original_prior_for_stage2=True)
    shots, _ = legacy.sample(16, seed=99)
    pred, out = legacy.decode(shots, full_output=True)
    false_pred, false_out = explicit_false.decode(shots, full_output=True)
    np.testing.assert_array_equal(pred, false_pred)
    for name in ("candidate_weights", "candidate_generation_weights",
                 "candidate_original_corrections", "error_preds", "best_candidate_indices"):
        np.testing.assert_array_equal(out[name], false_out[name])
    _, original_out = original.decode(shots, full_output=True, check_validity=True)
    assert original_out["validity"].all()
    np.testing.assert_array_equal(out["candidate_weights"][:, :3],
                                  original_out["candidate_weights"][:, :3])


def test_loading_saved_code_without_stage2_prior_option(tmp_path):
    import pickle
    cc = code()
    shots, _ = cc.sample(4, seed=31)
    path = tmp_path / "legacy.pkl"
    cc.save(str(path))
    with path.open("rb") as stream:
        data = pickle.load(stream)
    del data["use_original_prior_for_stage2"]
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    restored = ColorCode.load(str(path))
    assert restored.use_original_prior_for_stage2 is False
    np.testing.assert_array_equal(cc.decode(shots), restored.decode(shots))
