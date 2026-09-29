"""Adaptive relifting, exact syndrome reuse, and public configuration."""

import pickle

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from color_code_stim import ColorCode
from color_code_stim.decoders.baseline_equivalence import classify_baselines
from color_code_stim.decoders import cross_color_relifting as relift
from color_code_stim.noise_model import NoiseModel


@pytest.mark.parametrize("vectors,expected", [
    ([[0, 1], [0, 1], [0, 1]], (0, (("r", "g", "b"),))),
    ([[1, 0], [0, 1], [0, 1]], (1, (("r",), ("g", "b")))),
    ([[1, 0], [0, 1], [1, 0]], (1, (("r", "b"), ("g",)))),
    ([[1, 0], [0, 1], [1, 1]], (2, (("r",), ("g",), ("b",)))),
])
def test_shared_baseline_equivalence(vectors, expected):
    array = np.asarray(vectors, dtype=bool)
    assert classify_baselines(array) == expected
    assert classify_baselines(dict(zip("rgb", array))) == expected
    assert relift.classify(dict(zip("rgb", array))) == expected[0]


def make_code(*, comparative=False, basis="stage2", enabled=True, p=.1):
    return ColorCode(
        d=3, rounds=3, circuit_type="tri", cnot_schedule="tri_optimal",
        noise_model=NoiseModel.uniform_circuit_noise(p),
        comparative_decoding=comparative,
        remove_non_edge_like_errors=False,
        enable_cross_color_relifting=enabled,
        color_correlated_weight_basis=basis,
    )


def test_projection_is_parity_and_anchored_formula():
    mapping = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=bool))
    original = np.array([1, 1, 0], dtype=bool)
    np.testing.assert_array_equal(relift.projection(original, mapping), [False, True])
    anchor = np.array([1, 0, 1], dtype=bool)
    first = np.array([0, 1, 1], dtype=bool)
    second = np.array([1, 1, 0], dtype=bool)
    np.testing.assert_array_equal(relift.anchored(anchor, first, second),
                                  anchor ^ first ^ second)


def test_classification_and_canonical_slots():
    a = np.array([1, 0], dtype=bool)
    b = np.array([0, 1], dtype=bool)
    c = np.array([1, 1], dtype=bool)
    specs = relift.candidate_specs()
    assert [s.label for s in specs] == [
        "r", "g", "b", "r<-g", "r<-b", "g<-r", "g<-b", "b<-r", "b<-g",
        "r<-g,b[anchored]", "g<-r,b[anchored]", "b<-r,g[anchored]",
    ]
    assert relift.classify(dict(r=a, g=a, b=a)) == 0
    pair = dict(r=a, g=b, b=b)
    assert relift.classify(pair) == 1
    assert relift.alias_for_class_one(specs[4], pair, specs) == 3
    assert relift.alias_for_class_one(specs[9], pair, specs) == 0
    assert relift.alias_for_class_one(specs[10], pair, specs) == 5
    assert relift.classify(dict(r=a, g=b, b=c)) == 2


@pytest.mark.parametrize("basis", ["stage2", "original_dem"])
def test_real_candidates_valid_and_counted(monkeypatch, basis):
    cc = make_code(p=.3, basis=basis)
    shots, _ = cc.sample(30, seed=3)
    decoder = cc.concat_matching_decoder
    original = decoder._decode_stage2
    calls = []

    def counted(detectors, stage1, color, *args, **kwargs):
        calls.append((color, len(detectors)))
        return original(detectors, stage1, color, *args, **kwargs)

    monkeypatch.setattr(decoder, "_decode_stage2", counted)
    _, out = cc.decode(shots, full_output=True, check_validity=True)
    assert out["candidate_weights"].shape == (1, 12, 30)
    assert out["validity"].all()
    assert set(out["relift_run_class"]) == {0, 1, 2}
    assert out["relift_extra_stage2_calls"][out["relift_run_class"] == 0].max() == 0
    assert out["relift_extra_stage2_calls"][out["relift_run_class"] == 1].max() <= 3
    assert out["relift_extra_stage2_calls"].max() <= 9
    assert sum(count == 1 for _, count in calls) == out["relift_extra_stage2_calls"].sum()
    assert len(calls) == 3 + out["relift_extra_stage2_calls"].sum()
    assert np.all(out["candidate_stage1_validity"])
    assert np.all(out["candidate_executed"][0, :3])
    np.testing.assert_allclose(out["weights"], out["candidate_weights"].min(axis=(0, 1)))
    assert out["candidate_weight_basis"] == basis
    for slot, target in enumerate(out["candidate_target_colors"]):
        native = out["candidate_native_stage2_preds"][0][slot]
        corrected = out["candidate_weights"][0, slot]
        assert np.isfinite(corrected).all()
        assert native is not None
        mapped = cc.dem_manager.dems_decomposed[target].map_errors_to_org_dem(native, stage=2)
        predicted = np.asarray((mapped.astype(np.uint8) @ cc.dem_manager.H.T) % 2, dtype=bool)
        np.testing.assert_array_equal(predicted, shots)
        for shot, alias in enumerate(out["candidate_alias_of"][0, slot]):
            if alias >= 0:
                assert not out["candidate_executed"][0, slot, shot]
                assert corrected[shot] == out["candidate_weights"][0, alias, shot]
                np.testing.assert_array_equal(native[shot], out["candidate_native_stage2_preds"][0][alias][shot])


def test_full_distinct_can_reach_fifteen_calls(monkeypatch):
    cc = make_code(p=.3)
    shots, _ = cc.sample(30, seed=3)
    decoder = cc.concat_matching_decoder
    original = decoder._decode_stage2
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(decoder, "_decode_stage2", counted)
    _, out = cc.decode(shots, full_output=True)
    assert np.any((out["relift_run_class"] == 2) &
                  (out["relift_extra_stage2_calls"] == 9))
    assert calls == 3 + int(out["relift_extra_stage2_calls"].sum())


def test_all_relift_hypotheses_preserve_stage1_syndrome_and_cache_aliases():
    cc = make_code(p=.3)
    shots, _ = cc.sample(30, seed=3)
    _, out = cc.decode(shots, full_output=True)
    decoder = cc.concat_matching_decoder
    class_two = np.flatnonzero(out["relift_run_class"] == 2)
    assert len(class_two)
    shot = class_two[0]
    baselines = {color: cc.dem_manager.dems_decomposed[color].map_errors_to_org_dem(
        out["candidate_native_stage2_preds"][0][index][shot], stage=2
    ).astype(bool) for index, color in enumerate("rgb")}
    for target in "rgb":
        decomp = cc.dem_manager.dems_decomposed[target]
        stage1 = decoder._decode_stage1(shots[shot:shot + 1], target)[0]
        projected = [relift.projection(baselines[source], decomp.error_map_matrices[0])
                     for source in "rgb" if source != target]
        for candidate in (*projected, relift.anchored(stage1, *projected)):
            assert relift.stage1_valid(candidate, decomp.Hs[0], shots[shot])
    aliases = out["candidate_alias_of"][0, 3:, class_two]
    assert np.any((aliases >= 0) & (aliases < 3))
    assert np.any(aliases >= 3)


def test_comparative_is_class_local():
    cc = make_code(comparative=True)
    shots, _ = cc.sample(5, seed=4)
    _, out = cc.decode(shots, full_output=True, check_validity=True)
    assert out["candidate_weights"].shape == (2, 12, 5)
    assert out["relift_run_class_by_logical_class"].shape == (2, 5)
    minima = out["candidate_weights"].min(axis=1)
    np.testing.assert_allclose(out["weights"], minima.min(axis=0))
    np.testing.assert_allclose(out["logical_gaps"], abs(minima[0] - minima[1]))
    assert out["validity"].all()


def test_incompatibilities_and_persistence(tmp_path):
    with pytest.raises(NotImplementedError, match="remove_non_edge_like_errors=False"):
        ColorCode(d=3, rounds=1, enable_cross_color_relifting=True)
    with pytest.raises(NotImplementedError, match="color-correlated"):
        ColorCode(d=3, rounds=1, remove_non_edge_like_errors=False,
                  enable_cross_color_relifting=True, enable_colorcorrelated_decoding=True)
    cc = make_code()
    shots, _ = cc.sample(1, seed=4)
    bp_prediction, bp_extra = cc.decode(shots, bp_predecoding=True, metrics=["weights"])
    assert bp_prediction.shape == (len(shots),)
    assert bp_extra["bp_converged"].dtype == bool
    np.testing.assert_array_equal(np.ma.getmaskarray(bp_extra["weights"]), bp_extra["bp_converged"])
    with pytest.raises(NotImplementedError, match="UNCLASSIFIED"):
        cc.decode(shots, compute_swim_distance=True)
    with pytest.raises(NotImplementedError, match="custom DEM"):
        cc.concat_matching_decoder.decode(shots, custom_dem_data={})
    with pytest.raises(ValueError, match="all three colors"):
        cc.decode(shots, colors="r")
    path = tmp_path / "code.pkl"
    cc.save(str(path))
    restored = ColorCode.load(str(path))
    assert restored.enable_cross_color_relifting
    assert restored.concat_matching_decoder.enable_cross_color_relifting
    with path.open("rb") as f:
        old = pickle.load(f)
    old.pop("enable_cross_color_relifting")
    with path.open("wb") as f:
        pickle.dump(old, f)
    assert not ColorCode.load(str(path)).enable_cross_color_relifting


def test_full_decomposition_graphlike_and_invalid_rejected(monkeypatch):
    cc = make_code()
    for color in "rgb":
        for H in cc.dem_manager.dems_decomposed[color].Hs:
            assert np.all(np.diff(H.tocsc().indptr) <= 2)
    shots, _ = cc.sample(1, seed=4)
    decomp = cc.dem_manager.dems_decomposed["r"]
    original = decomp.Hs
    bad = original[0].tolil()
    bad[:3, 0] = True
    decomp.Hs = (bad.tocsc(), original[1])
    with pytest.raises(NotImplementedError, match="not graphlike"):
        cc.decode(shots)
