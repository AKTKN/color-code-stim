"""Checks for source-guided candidates and common-prior selection."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from color_code_stim import ColorCode
from color_code_stim.decoders.color_correlated_decoding import (
    ColorCorrelatedPriorReweighter, candidate_specs, guide_union,
)
from color_code_stim.noise_model import NoiseModel


def code(*, comparative=False, enabled=True):
    return ColorCode(
        d=3, rounds=1, circuit_type="tri", cnot_schedule="tri_optimal",
        noise_model=NoiseModel(bitflip=.05), comparative_decoding=comparative,
        enable_colorcorrelated_decoding=enabled,
    )


def synthetic_reweighter(q=(.1, .2, .3), rows=((0,), (0, 1), (0, 1, 2), (2,))):
    q = np.asarray(q)
    matrix = np.zeros((len(rows), len(q)), dtype=bool)
    for i, row in enumerate(rows):
        matrix[i, list(row)] = True
    base = np.array([(1 - np.prod(1 - 2 * q[list(row)])) / 2 for row in rows])
    symbolic = [SimpleNamespace(prob_muls=np.ones(len(row))) for row in rows]
    decomp = SimpleNamespace(
        org_prob=q, probs=(base, base),
        error_map_matrices=(csr_matrix(matrix), csr_matrix(matrix)),
        dems_symbolic=(symbolic, symbolic),
    )
    return ColorCorrelatedPriorReweighter(
        SimpleNamespace(dems_decomposed={c: decomp for c in "rgb"})
    )


def test_reweighting_formula_and_or_guide():
    rw = synthetic_reweighter()
    guide = np.array([True, True, False])
    p1, p2 = rw.probabilities("r", guide)
    np.testing.assert_array_equal(p1, p2)
    assert p1[0] == pytest.approx(1 - 1e-14)
    assert p1[3] == pytest.approx(.3)  # disjoint source
    assert p1[1] == pytest.approx(.9)  # max(.8 conditional on 0, .9 on 1)
    # Enumerate independent Bernoulli sources conditional on source 0 active.
    probability = 0.
    for x1 in (0, 1):
        for x2 in (0, 1):
            probability += (x1 + x2 + 1) % 2 * (.2 if x1 else .8) * (.3 if x2 else .7)
    assert rw.probabilities("r", np.array([True, False, False]))[0][2] == pytest.approx(probability)

    specs = candidate_specs()
    assert len(specs) == 12
    assert tuple(s.target_color for s in specs) == tuple("rgb") + tuple("rrrgggbbb")
    corrections = {"g": np.array([1, 1, 0], dtype=bool),
                   "b": np.array([1, 0, 1], dtype=bool)}
    np.testing.assert_array_equal(guide_union(corrections, ("g", "b")), [1, 1, 1])


def test_reject_nonunit_multiplier():
    q = np.array([.1])
    decomp = SimpleNamespace(
        org_prob=q, probs=(q, q),
        error_map_matrices=(csr_matrix([[1]]), csr_matrix([[1]])),
        dems_symbolic=([SimpleNamespace(prob_muls=np.array([.5]))],) * 2,
    )
    with pytest.raises(NotImplementedError, match="prob_muls"):
        ColorCorrelatedPriorReweighter(SimpleNamespace(dems_decomposed={c: decomp for c in "rgb"}))


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
    for cls, natives in enumerate(out["candidate_native_stage2_preds"]):
        for j, native in enumerate(natives):
            c = out["candidate_target_colors"][j]
            p = new.dems_decomposed[c].probs[1]
            llr = np.log((1 - p) / p)
            np.testing.assert_allclose(out["candidate_weights"][cls, j], native @ llr)
    if comparative:
        minima = np.min(out["candidate_weights"], axis=1)
        np.testing.assert_allclose(out["logical_gaps"], np.abs(minima[0] - minima[1]))


def test_selection_ignores_generation_weights(monkeypatch):
    cc = code()
    shots, _ = cc.sample(16, seed=593)
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
    assert np.any(out["candidate_weights"].argmin(axis=1) !=
                  out["candidate_generation_weights"].argmin(axis=1))


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
    assert np.isfinite(out["candidate_weights"][:, :, ~success]).all()
    np.testing.assert_array_equal(out["best_candidate_indices"][success], -1)
    for c in "rgb":
        for current, original in zip(cc.dems_decomposed[c].probs, before[c]):
            np.testing.assert_array_equal(current, original)
