"""Scientific accounting of the example's exact and Monte Carlo summaries."""

from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def analysis():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "examples/evaluate_path_gap.py")
    )


def test_probability_weighting_and_randomized_whole_ties(analysis):
    failures = np.array([False, True, True])
    probabilities = np.array([0.7, 0.2, 0.1])
    scores = {"phi": np.array([2.0, 2.0, -1.0])}
    thresholds, conditional, matched = analysis["score_tables"](
        scores, failures, probabilities, True
    )
    assert thresholds[0]["conditional_failure"] == pytest.approx(0.3)
    assert thresholds[1]["conditional_failure"] == pytest.approx(2 / 9)
    # A tie cannot favor the failure-free first physical pattern.
    assert matched[-1]["conditional_failure"] == pytest.approx(2 / 9)
    assert matched[-1]["retained_fraction"] == pytest.approx(0.5)
    permutation = [2, 1, 0]
    other = analysis["score_tables"](
        {"phi": scores["phi"][permutation]},
        failures[permutation],
        probabilities[permutation],
        True,
    )[2]
    for first, second in zip(matched, other):
        assert first["conditional_failure"] == pytest.approx(
            second["conditional_failure"]
        )
    assert conditional[0]["value"] == -1


def test_zero_failures_keep_binomial_upper_bound(analysis):
    record = analysis["rate_record"](
        np.zeros(10, dtype=bool), np.full(10, 0.1), np.ones(10, dtype=bool), False
    )
    assert record["conditional_failure"] == 0
    assert record["ci_high"] > 0.2


def test_signed_class_correction_and_suboptimality_distinct(analysis):
    # Three-qubit repetition sector: H=[110;011], logical parity is q0.
    h = np.array([[1, 1, 0], [0, 1, 1]])
    logical = np.array([1, 0, 0])
    errors = np.array([[0, 0, 0], [1, 1, 1]])
    corrections = np.array([[0, 0, 0], [1, 1, 1]])
    rows = analysis["exact_quantities"](errors, corrections, np.ones(3), h, logical)
    assert rows[0]["delta_class"] == 3
    assert rows[1]["delta_class"] == -3
    assert rows[1]["optimal_class_gap"] == 3
    # Trivial H: q1 is a within-class nonminimal correction.
    rows = analysis["exact_quantities"](
        np.array([[0, 0]]),
        np.array([[0, 1]]),
        np.array([3.0, 2.0]),
        np.zeros((1, 2), dtype=int),
        np.array([1, 0]),
    )
    assert rows == [
        {
            "delta_class": 3.0,
            "delta_corr": 1.0,
            "within_class_suboptimality": 2.0,
            "optimal_class_gap": 3.0,
        }
    ]
