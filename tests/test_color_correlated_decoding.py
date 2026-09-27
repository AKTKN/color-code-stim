"""Checks for source-guided candidates and common-prior selection."""

import numpy as np
import pytest

from color_code_stim import ColorCode
from color_code_stim.decoders.color_correlated_decoding import (
    ColorCorrelatedPriorReweighter, candidate_schedule, candidate_specs, guide_union,
)
from color_code_stim.noise_model import NoiseModel
from color_code_stim.stim_utils import dem_to_parity_check


def code(*, comparative=False, enabled=True, weight_basis="stage2", uniform=False):
    return ColorCode(
        d=3, rounds=3 if uniform else 1, circuit_type="tri", cnot_schedule="tri_optimal",
        noise_model=NoiseModel.uniform_circuit_noise(.02) if uniform else NoiseModel(bitflip=.05),
        comparative_decoding=comparative,
        enable_colorcorrelated_decoding=enabled,
        color_correlated_weight_basis=weight_basis,
    )


def test_reweighted_original_dem_is_redecomposed_for_target_color():
    cc = code()
    manager = cc.dem_manager
    rw = ColorCorrelatedPriorReweighter(manager)
    base = manager.dems_decomposed["r"]
    mapping = base.error_map_matrices[0].tocsr()
    row = next(i for i, n in enumerate(np.diff(mapping.indptr)) if n >= 2)
    sources = mapping.indices[mapping.indptr[row]:mapping.indptr[row + 1]]
    guide = np.zeros(len(manager.probs_xz), dtype=bool)
    guide[sources[:2]] = True
    updated = rw.source_probabilities(guide)
    np.testing.assert_array_equal(updated[~guide], manager.probs_xz[~guide])
    np.testing.assert_allclose(updated[guide], 1 - 1e-14)
    _, _, dem_prob = dem_to_parity_check(rw.reweighted_dem(guide))
    np.testing.assert_array_equal(dem_prob, updated)
    rebuilt = rw.decomposition("r", guide)
    np.testing.assert_array_equal(rebuilt.org_prob, updated)
    expected = (1 - np.prod(1 - 2 * updated[sources])) / 2
    assert rebuilt.probs[0][row] == pytest.approx(expected)
    assert rebuilt.probs[0][row] != pytest.approx(base.probs[0][row])
    np.testing.assert_array_equal(base.org_prob, manager.probs_xz)


def test_guide_union_and_source_validation():
    specs = candidate_specs()
    assert len(specs) == 12
    assert tuple(s.target_color for s in specs) == tuple("rgb") + tuple("rrrgggbbb")
    corrections = {"g": np.array([1, 1, 0], dtype=bool),
                   "b": np.array([1, 0, 1], dtype=bool)}
    np.testing.assert_array_equal(guide_union(corrections, ("g", "b")), [1, 1, 1])
    rw = ColorCorrelatedPriorReweighter(code().dem_manager)
    with pytest.raises(ValueError, match="Guide must"):
        rw.source_probabilities(np.array([True]))


def test_candidate_schedule_in_original_dem_order():
    r = np.array([1, 0, 0], dtype=bool)
    g = np.array([0, 1, 0], dtype=bool)
    b = np.array([0, 0, 1], dtype=bool)
    assert candidate_schedule(np.array([r, r, r])) == (0, ())
    assert candidate_schedule(np.array([r, r, b])) == (1, (4, 7, 9))
    assert candidate_schedule(np.array([r, g, r])) == (1, (3, 6, 10))
    assert candidate_schedule(np.array([r, g, g])) == (1, (3, 6, 9))
    assert candidate_schedule(np.array([r, g, b])) == (2, tuple(range(3, 12)))


def test_uniform_shots_execute_only_scheduled_candidates(monkeypatch):
    cc = code(uniform=True)
    shots, _ = cc.sample(96, seed=51)
    calls = []
    original = ColorCorrelatedPriorReweighter.decomposition

    def counted(self, color, guide):
        calls.append(color)
        return original(self, color, guide)

    monkeypatch.setattr(ColorCorrelatedPriorReweighter, "decomposition", counted)
    _, out = cc.decode(shots, full_output=True, check_validity=True)
    assert out["validity"].all()
    assert set(out["color_correlated_run"]) == {0, 1, 2}
    executed = out["candidate_executed"]
    expected_count = np.array([3, 6, 12])
    np.testing.assert_array_equal(executed.sum(axis=1)[0],
                                  expected_count[out["color_correlated_run"]])
    assert len(calls) == sum((0, 3, 9)[category] for category in out["color_correlated_run"])
    assert np.isfinite(out["candidate_weights"][executed]).all()
    assert np.isposinf(out["candidate_weights"][~executed]).all()
    for shot, category in enumerate(out["color_correlated_run"]):
        if category == 0:
            assert not executed[0, 3:, shot].any()
        elif category == 1:
            assert executed[0, 3:, shot].sum() == 3
        else:
            assert executed[0, 3:, shot].all()


@pytest.mark.parametrize("comparative", [False, True])
def test_candidates_validity_and_class_gap(comparative):
    old, new = code(comparative=comparative, enabled=False), code(comparative=comparative)
    sampled, _ = old.sample(32, seed=443)
    _, sampled_out = old.decode(sampled, full_output=True, check_validity=True)
    shots = sampled[sampled_out["validity"]][:8]
    assert len(shots) == 8
    old_pred, old_out = old.decode(shots, full_output=True, check_validity=True)
    # The disabled option is the historical path, including its output shape.
    direct_pred, direct_out = code(comparative=comparative, enabled=False).decode(
        shots, full_output=True, check_validity=True
    )
    np.testing.assert_array_equal(old_pred, direct_pred)
    for key in old_out:
        np.testing.assert_array_equal(old_out[key], direct_out[key])

    _, out = new.decode(shots, full_output=True, check_validity=True)
    assert out["candidate_weights"].shape == (2 if comparative else 1, 12, 8)
    assert len(out["candidate_labels"]) == 12
    assert out["candidate_labels"][5] == "r<-g OR b"
    assert out["validity"].all()
    np.testing.assert_array_equal(
        out["best_colors"],
        np.array(["rgb".index(out["candidate_target_colors"][i])
                  for i in out["best_candidate_indices"]]),
    )
    np.testing.assert_allclose(out["weights"], np.min(out["candidate_weights"], axis=(0, 1)))
    assert np.all(out["color_correlated_run"] == 0)
    assert np.all(out["candidate_executed"][:, :3, :])
    if not comparative:
        assert not np.any(out["candidate_executed"][:, 3:, :])
    for cls, natives in enumerate(out["candidate_native_stage2_preds"]):
        for j, native in enumerate(natives):
            executed = out["candidate_executed"][cls, j]
            if native is None:
                assert not executed.any()
                assert np.isposinf(out["candidate_weights"][cls, j]).all()
                continue
            c = out["candidate_target_colors"][j]
            p = new.dems_decomposed[c].probs[1]
            llr = np.log((1 - p) / p)
            np.testing.assert_allclose(out["candidate_weights"][cls, j, executed],
                                       native[executed] @ llr)
    if comparative:
        minima = np.min(out["candidate_weights"], axis=1)
        np.testing.assert_allclose(out["logical_gaps"], np.abs(minima[0] - minima[1]))


def test_selection_ignores_generation_weights(monkeypatch):
    cc = code(uniform=True)
    shots, _ = cc.sample(48, seed=51)
    decoder = cc.concat_matching_decoder
    original = decoder._decode_stage2

    def inverted_generation(detectors, stage1, color, custom_dem_data=None, **kwargs):
        native, _ = original(detectors, stage1, color, custom_dem_data, **kwargs)
        if custom_dem_data is not None:
            native[:, 0] = True
            return native, np.full(len(native), -1e6)
        return native, np.zeros(len(native))

    monkeypatch.setattr(decoder, "_decode_stage2", inverted_generation)
    _, out = cc.decode(shots, full_output=True)
    np.testing.assert_allclose(out["weights"], out["candidate_weights"].min(axis=(0, 1)))
    assert np.any(out["candidate_executed"][:, 3:, :])
    assert np.any(out["candidate_weights"].argmin(axis=1) !=
                  out["candidate_generation_weights"].argmin(axis=1))


@pytest.mark.parametrize("comparative", [False, True])
def test_original_dem_basis_scores_mapped_corrections(comparative):
    cc = code(comparative=comparative, weight_basis="original_dem")
    shots, _ = cc.sample(12, seed=227)
    _, out = cc.decode(shots, full_output=True)
    assert out["candidate_weight_basis"] == "original_dem"
    q = cc.dem_manager.probs_xz
    llr = np.log((1 - q) / q)
    for cls, natives in enumerate(out["candidate_native_stage2_preds"]):
        for j, native in enumerate(natives):
            executed = out["candidate_executed"][cls, j]
            if native is None:
                assert not executed.any()
                continue
            decomp = cc.dems_decomposed[out["candidate_target_colors"][j]]
            mapped = decomp.map_errors_to_org_dem(native, stage=2)
            np.testing.assert_allclose(out["candidate_weights"][cls, j, executed],
                                       mapped[executed] @ llr)
    np.testing.assert_allclose(out["weights"], out["candidate_weights"].min(axis=(0, 1)))
    if comparative:
        minima = out["candidate_weights"].min(axis=1)
        np.testing.assert_allclose(out["logical_gaps"], np.abs(minima[0] - minima[1]))


def test_unsupported_options_and_colorcode_persistence(tmp_path):
    cc = code()
    shots, _ = cc.sample(1, seed=13)
    with pytest.raises(ValueError, match="all three colors"):
        cc.decode(shots, colors=["r", "g"])
    with pytest.raises(NotImplementedError, match="custom_dem_data"):
        cc.concat_matching_decoder.decode(shots, custom_dem_data={})
    with pytest.raises(NotImplementedError, match="swim"):
        cc.decode(shots, compute_swim_distance=True)
    with pytest.raises(NotImplementedError, match="BP"):
        cc.decode(shots, bp_predecoding=True)
    path = tmp_path / "color_code.pkl"
    cc.save(str(path))
    restored = ColorCode.load(str(path))
    assert restored.enable_colorcorrelated_decoding is True
    assert restored.concat_matching_decoder.enable_colorcorrelated_decoding is True


@pytest.mark.parametrize("partial", [False, True])
def test_erasure_predecoding_only_expands_remaining_samples(partial):
    cc = code(comparative=True)
    shots, _ = cc.sample(6, seed=55)
    before = {c: tuple(p.copy() for p in cc.dems_decomposed[c].probs) for c in "rgb"}
    _, out = cc.decode(
        shots, erasure_matcher_predecoding=True,
        partial_correction_by_predecoding=partial, full_output=True,
    )
    success = out["erasure_matcher_success"]
    assert np.any(success) and np.any(~success)
    assert out["candidate_weights"].shape == (2, 12, len(shots))
    assert np.isnan(out["candidate_weights"][:, :, success]).all()
    assert np.isfinite(out["candidate_weights"][out["candidate_executed"]]).all()
    assert np.isposinf(out["candidate_weights"][:, :, ~success]
                       [~out["candidate_executed"][:, :, ~success]]).all()
    assert np.all(out["color_correlated_run"][success] == -1)
    np.testing.assert_array_equal(out["best_candidate_indices"][success], -1)
    for c in "rgb":
        for current, original in zip(cc.dems_decomposed[c].probs, before[c]):
            np.testing.assert_array_equal(current, original)
