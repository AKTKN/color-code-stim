"""Compact statistics must reproduce the established full-output oracle."""

import numpy as np
import pytest

from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def options(mode, basis="original_dem"):
    result = {"color_correlated_weight_basis": basis}
    if mode == "correlated":
        result["enable_colorcorrelated_decoding"] = True
    elif mode == "relift":
        result.update(enable_cross_color_relifting=True, remove_non_edge_like_errors=False)
    elif mode in ("original_dem", "native"):
        result.update(enable_prior_perturbation=True, stage1_perturbation=mode == "native",
                      perturbation_ensemble_size=4, perturbation_alpha=.5, perturbation_seed=719)
    return result


def expected_metrics(prediction, extra, actual, mode, baseline, comparative, swim):
    result = {"logical_error": prediction != actual, "weights": extra["weights"]}
    if mode != "ordinary":
        ordinary = np.min(extra["candidate_weights"][:, :3, :], axis=(0, 1))
        default_failure = baseline != actual
        result.update(default_logical_error=default_failure,
                      better_weight_by_color_correlated_decoding=(extra["weights"] < ordinary).astype(np.uint8),
                      effect_by_color_correlated_decoding=(default_failure & ~(prediction != actual)).astype(np.uint8))
    if mode == "correlated":
        result["color_correlated_run"] = extra["color_correlated_run"].astype(np.uint8)
    elif mode == "relift":
        result["relift_run"] = extra["relift_run_class"].astype(np.uint8)
    if comparative:
        result["logical_gap"] = extra["logical_gaps"]
    if swim:
        result["swim_distance"] = extra["class_min_swim_distance"]
    return result


@pytest.mark.parametrize("mode,basis", [
    (mode, basis) for mode in ("ordinary", "correlated", "relift", "original_dem", "native")
    for basis in ("stage2", "original_dem") if mode != "correlated" or basis == "original_dem"
])
@pytest.mark.parametrize("soft", ["none", "comparative", "swim"])
def test_all_statistics_match_full_output(mode, basis, soft):
    common = dict(d=3, rounds=1, noise_model=NoiseModel(bitflip=.1),
                  comparative_decoding=soft == "comparative")
    opts = options(mode, basis)
    reference, compact = (ColorCode(**common, **opts) for _ in range(2))
    detectors, actual = reference.sample(48, seed=391)
    # Exercise exact zero-syndrome ties as well as sampled nonzero candidates.
    detectors[:3] = False
    baseline = None
    if mode == "correlated":
        baseline = ColorCode(**common, color_correlated_weight_basis=basis).decode(detectors)
    prediction, extra = reference.decode(detectors, full_output=True,
                                         compute_swim_distance=soft == "swim")
    expected = expected_metrics(prediction, extra, actual, mode,
                                baseline if mode == "correlated" else extra.get("baseline_predictions"),
                                soft == "comparative", soft == "swim")
    got_prediction, metrics = compact.decode(
        detectors, metrics=tuple(expected), actual_observables=actual,
        baseline_predictions=baseline, compute_swim_distance=soft == "swim",
        check_validity=True,
    )
    np.testing.assert_array_equal(got_prediction, prediction)
    assert metrics.keys() == expected.keys()
    for name, values in expected.items():
        assert metrics[name].shape == (len(detectors),)
        assert metrics[name].dtype == values.dtype
        np.testing.assert_array_equal(metrics[name], values, err_msg=name)


@pytest.mark.parametrize("mode", ["ordinary", "correlated", "relift", "original_dem", "native"])
def test_only_requested_fields_and_empty_batches(mode):
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.05), **options(mode))
    detectors, actual = code.sample(3, seed=73)
    prediction, metrics = code.decode(detectors, metrics=("logical_error",), actual_observables=actual)
    assert set(metrics) == {"logical_error"}
    np.testing.assert_array_equal(metrics["logical_error"], prediction != actual)
    position = code.concat_matching_decoder._perturbation_ensemble
    cursor = position.shot_position if position is not None else None
    result, metrics = code.decode(detectors[:0], metrics=("logical_error", "weights"),
                                  actual_observables=actual[:0])
    assert result.shape == (0,)
    assert all(values.shape == (0,) for values in metrics.values())
    if position is not None:
        assert position.shot_position == cursor


@pytest.mark.parametrize("kwargs,match", [
    ({"metrics": ("unknown",)}, "Unknown"),
    ({"metrics": ("weights", "weights")}, "duplicate"),
    ({"metrics": "weights"}, "sequence"),
    ({"metrics": ("logical_error",)}, "actual_observables"),
    ({"metrics": ("weights",), "full_output": True}, "full_output=False"),
    ({"metrics": ("weights",), "return_candidate_data": True}, "return_candidate_data=False"),
    ({"metrics": ("logical_gap",)}, "comparative"),
    ({"metrics": ("swim_distance",)}, "scorer"),
    ({"metrics": ("relift_run",)}, "relifting"),
])
def test_invalid_requests_fail_before_decoding(kwargs, match):
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.05))
    detectors, _ = code.sample(2, seed=73)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        code.decode(detectors, **kwargs)
    # A rejected request must not poison later ordinary decoding.
    np.testing.assert_array_equal(code.decode(detectors), code.decode(detectors, full_output=True)[0])


def test_native_split_batches_and_small_request_preserve_rng():
    opts = options("native")
    common = dict(d=3, rounds=1, noise_model=NoiseModel(bitflip=.1), comparative_decoding=True)
    reference, compact = (ColorCode(**common, **opts) for _ in range(2))
    detectors, actual = reference.sample(17, seed=391)
    _, full = reference.decode(detectors, full_output=True, perturbation_shot_offset=100)
    chunks = []
    for start, stop in ((0, 5), (5, 6), (6, 17)):
        _, metrics = compact.decode(detectors[start:stop], metrics=("weights", "logical_gap"),
                                    perturbation_shot_offset=100 + start)
        chunks.append(metrics)
    for name, legacy in (("weights", "weights"), ("logical_gap", "logical_gaps")):
        np.testing.assert_array_equal(np.concatenate([chunk[name] for chunk in chunks]), full[legacy])
    # Advancing calls consume exactly the same shots for small and full outputs.
    reference, compact = (ColorCode(**common, **opts) for _ in range(2))
    for shots in (detectors[:5], detectors[5:]):
        _, full = reference.decode(shots, full_output=True)
        _, metrics = compact.decode(shots, metrics=("weights",))
        np.testing.assert_array_equal(metrics["weights"], full["weights"])


@pytest.mark.parametrize("mode", ["correlated", "relift", "native"])
def test_metrics_does_not_allocate_all_candidate_corrections(monkeypatch, mode):
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.05), **options(mode))
    detectors, actual = code.sample(8, seed=15)
    allocated = []
    original = np.zeros

    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocated.append(shape)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", zeros)
    code.decode(detectors, metrics=("logical_error",), actual_observables=actual)
    candidate_count = 12  # native M4, correlated/relift 12
    assert (1, candidate_count, len(detectors), code.H.shape[1]) not in allocated
    if mode == "native":
        import pymatching
        def rebuild(*args, **kwargs):
            raise AssertionError("Warm native metrics must reuse matching graphs")
        monkeypatch.setattr(pymatching.Matching, "from_check_matrix", rebuild)
        code.decode(detectors, metrics=("logical_error",), actual_observables=actual,
                    perturbation_shot_offset=100)


def test_actual_observables_only_affect_error_statistics():
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.1), **options("native"))
    detectors, actual = code.sample(12, seed=391)
    first, left = code.decode(detectors, metrics=("logical_error", "weights"),
                              actual_observables=actual, perturbation_shot_offset=73)
    second, right = code.decode(detectors, metrics=("logical_error", "weights"),
                                actual_observables=~actual, perturbation_shot_offset=73)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(left["weights"], right["weights"])
    np.testing.assert_array_equal(left["logical_error"], ~right["logical_error"])


@pytest.mark.parametrize("mode", ["native", "original_dem"])
@pytest.mark.parametrize("size,alpha", [(1, .5), (4, 0.)])
def test_unperturbed_batch_paths_match_full_output(mode, size, alpha):
    opts = options(mode) | {"perturbation_ensemble_size": size, "perturbation_alpha": alpha}
    reference, compact = (ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.1),
                                   comparative_decoding=True, **opts) for _ in range(2))
    detectors, actual = reference.sample(17, seed=391)
    prediction, extra = reference.decode(detectors, full_output=True)
    expected = expected_metrics(prediction, extra, actual, mode, extra['baseline_predictions'], True, False)
    got, metrics = compact.decode(detectors, metrics=tuple(expected), actual_observables=actual)
    np.testing.assert_array_equal(got, prediction)
    for name in expected:
        np.testing.assert_array_equal(metrics[name], expected[name])


@pytest.mark.parametrize("offset", [-1, True, 2**64])
def test_empty_native_batches_validate_offsets(offset):
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.05), **options("native"))
    detectors, _ = code.sample(1, seed=17)
    with pytest.raises(ValueError, match="uint64"):
        code.decode(detectors[:0], metrics=("weights",), perturbation_shot_offset=offset)


@pytest.mark.parametrize("bad_weight", [np.nan, np.inf, -np.inf])
def test_invalid_generated_scores_are_rejected_and_solver_recovers(monkeypatch, bad_weight):
    opts = options('native') | {'perturbation_ensemble_size': 1}
    code = ColorCode(d=3, rounds=1, noise_model=NoiseModel(bitflip=.05), **opts)
    detectors, actual = code.sample(8, seed=73)
    reference = code.decode(detectors)
    evaluator = code.concat_matching_decoder._candidate_evaluators['original_dem']
    evaluate = evaluator.evaluate
    def invalid(*args, **kwargs):
        correction, aligned, score, generation = evaluate(*args, **kwargs)
        return correction, aligned, np.full_like(score, bad_weight), generation
    monkeypatch.setattr(evaluator, 'evaluate', invalid)
    with pytest.raises(ValueError, match='finite'):
        code.decode(detectors, metrics=('better_weight_by_color_correlated_decoding',))
    monkeypatch.setattr(evaluator, 'evaluate', evaluate)
    prediction, metrics = code.decode(detectors, metrics=('logical_error',), actual_observables=actual)
    np.testing.assert_array_equal(prediction, reference)
    np.testing.assert_array_equal(metrics['logical_error'], prediction != actual)
