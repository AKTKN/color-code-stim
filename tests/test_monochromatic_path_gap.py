"""Physical and exhaustive oracles independent of production shortest paths."""

import copy
import itertools

import numpy as np
import pytest
import stim

from color_code_stim import ColorCode, NoiseModel
from color_code_stim.metrics import (
    COLOR_ORDER,
    ColorCodePathGap,
    DemQubitMap,
    MonochromaticPathGap,
    PathTopology,
    build_monochromatic_topologies,
)


def code(d=3, **kwargs):
    return ColorCode(d=d, rounds=1, noise_model=NoiseModel(bitflip=0.03), **kwargs)


def physical_matrices(cc):
    data = sorted(cc.qubit_groups["data"], key=lambda v: v["qid"])
    rows = [
        {v["qid"] for v in check.neighbors() if v["pauli"] is None}
        for check in cc.qubit_groups["anc_Z"]
    ]
    h = np.array([[v["qid"] in row for v in data] for row in rows], dtype=int)
    return h, np.array([v["obs"] for v in data], dtype=int)


def independent_faces(d):
    if d == 3:
        # Steane faces, physical positions ordered left-to-right bottom-to-top.
        return 7, [("r", {3, 4, 5, 6}), ("g", {1, 2, 3, 4}), ("b", {0, 1, 3, 5})]
    # Independent integer simplex construction, no package lattice/topology API.
    length = 3 * (d - 1) // 2
    points = [(i, j) for j in range(length + 1) for i in range(length - j + 1)]
    data = [q for q in points if (q[0] - q[1]) % 3 != 2]
    index = {q: k for k, q in enumerate(data)}
    offsets = [(-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)]
    faces = []
    for i, j in points:
        if (i - j) % 3 == 2:
            faces.append(
                (
                    "gbr"[j % 3],
                    {
                        index[(i + di, j + dj)]
                        for di, dj in offsets
                        if (i + di, j + dj) in index
                    },
                )
            )
    return len(data), faces


def oracle_paths(d, color):
    """Unsuppressed graph derived solely from physical face incidences.

    Group equal pairs of OTHER-color face/boundary labels into primal-edge
    vertices. A pair of boundary labels is the opposite corner terminal.
    Enumerate simple vertex paths by DFS; no igraph or metric builder calls.
    """
    n, faces = independent_faces(d)
    adjacency = {}
    source, target = ("corner",), ("boundary", color)
    for q in range(n):
        incident = {
            c: ("face", k) for k, (c, support) in enumerate(faces) if q in support
        }
        left = incident.get(color, target)
        other = tuple(
            incident.get(c, ("boundary", c)) for c in COLOR_ORDER if c != color
        )
        right = source if all(x[0] == "boundary" for x in other) else ("primal", other)
        adjacency.setdefault(left, []).append((right, q))
        adjacency.setdefault(right, []).append((left, q))
    paths = []

    def visit(v, seen, support):
        if v == target:
            paths.append(tuple(support))
            return
        for neighbor, qubit in adjacency[v]:
            if neighbor not in seen:
                visit(neighbor, seen | {neighbor}, support + [qubit])

    visit(source, {source}, [])
    return paths


@pytest.mark.parametrize(
    "weights", [np.ones(7), np.array([0.2, 1.7, 0.0, 4.1, 2.0, 0.3, 6.0])]
)
def test_all_128_masks_against_independent_path_enumeration(weights):
    cc = code()
    ids, topologies = build_monochromatic_topologies(cc.tanner_graph)
    masks = np.array(list(itertools.product((0, 1), repeat=7)), dtype=bool)
    result = MonochromaticPathGap(topologies, weights).evaluate(masks)
    expected = np.array(
        [
            [
                min(
                    sum(weights[q] for q in path if not e[q])
                    for path in oracle_paths(3, c)
                )
                for c in COLOR_ORDER
            ]
            for e in masks
        ]
    )
    np.testing.assert_allclose(result.distance_by_color, expected)
    np.testing.assert_allclose(
        result.phi_by_color, expected - (masks @ weights)[:, None]
    )
    np.testing.assert_allclose(result.phi, expected.min(axis=1) - masks @ weights)
    assert ids == (0, 1, 4, 7, 8, 9, 12)


def test_independent_d5_oracle():
    cc = code(5)
    _, topologies = build_monochromatic_topologies(cc.tanner_graph)
    n, faces = independent_faces(5)
    h, logical = physical_matrices(cc)
    assert {tuple(row) for row in h} == {
        tuple(int(q in support) for q in range(n)) for _, support in faces
    }
    rng = np.random.default_rng(734)
    weights = rng.uniform(0, 5, n)
    masks = rng.integers(0, 2, (32, n)).astype(bool)
    result = MonochromaticPathGap(topologies, weights).evaluate(masks)
    for c, color in enumerate(COLOR_ORDER):
        paths = oracle_paths(5, color)
        for path in paths:
            support = np.zeros(n, dtype=int)
            support[list(path)] = 1
            assert not np.any(h @ support % 2)
            assert logical @ support % 2 == 1
        expected = [
            min(sum(weights[q] for q in path if not e[q]) for path in paths)
            for e in masks
        ]
        np.testing.assert_allclose(result.distance_by_color[:, c], expected)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_topology_witnesses_uniform_analytics_and_bounds(d):
    cc = code(d)
    adapter = ColorCodePathGap(cc)
    n = len(adapter.qubit_ids)
    evaluator = MonochromaticPathGap(adapter.evaluator.topologies, np.ones(n))
    h, logical = physical_matrices(cc)
    masks = np.random.default_rng(d).integers(0, 2, (40, n))
    result = evaluator.evaluate(masks, diagnostics=True)
    for i, paths in enumerate(result.paths_by_color):
        for c, path in enumerate(paths):
            z = np.zeros(n, dtype=int)
            z[list(path)] = 1
            assert len(path) == z.sum() and len(path) % 2 == 1
            assert not np.any(h @ z % 2)
            assert logical @ z % 2 == 1
            assert result.distance_by_color[i, c] == np.count_nonzero(
                z & (1 - masks[i])
            )
            assert evaluator.evaluate(z).distance_by_color[c] == 0
    size = masks.sum(axis=1)
    assert np.all(result.phi >= d - 2 * size)
    assert np.all(result.phi <= d - size)
    empty = evaluator.evaluate(np.zeros(n))
    full = evaluator.evaluate(np.ones(n))
    np.testing.assert_equal(empty.distance_by_color, np.full(3, d))
    assert empty.phi == d and empty.minimizing_color == "r"
    np.testing.assert_equal(full.distance_by_color, np.zeros(3))
    assert full.phi == -n
    for color, topology in zip(COLOR_ORDER, evaluator.topologies):
        assert topology.source != topology.target
        singles = [s for s in topology.supports if len(s) == 1]
        assert len(singles) == 1
        v = cc.tanner_graph.vs.find(qid=adapter.qubit_ids[singles[0][0]])
        assert set(v["boundary"]) == set(COLOR_ORDER) - {color}


def test_pair_singleton_accounting_parallel_edges_outside_path_and_negative():
    # Parallel pair edges share endpoints, but must retain separate supports.
    t = PathTopology(3, ((0, 1), (1, 2), (1, 2)), ((0,), (1, 2), (3, 4)), 0, 2)
    w = np.array([2.0, 3.0, 5.0, 20.0, 40.0])
    evaluator = MonochromaticPathGap((t, t, t), w)
    for corner, first, second in itertools.product((0, 1), repeat=3):
        e = np.array([corner, first, second, 0, 0])
        result = evaluator.evaluate(e)
        assert result.distance_by_color[0] == 2 * (1 - corner) + 3 * (1 - first) + 5 * (
            1 - second
        )
        assert result.phi == result.distance_by_color[0] - w @ e
    e = np.array([0, 0, 0, 0, 1])  # corrected q4 is outside the shortest path
    result = evaluator.evaluate(e, diagnostics=True)
    assert 4 not in result.paths_by_color[0]
    assert result.correction_weight == 40 and result.phi == 10 - 40
    e = np.array([0, 0, 0, 1, 1])
    result = evaluator.evaluate(e, diagnostics=True)
    assert set(result.paths_by_color[0]) == {0, 3, 4}
    assert result.distance_by_color[0] == 2
    with pytest.raises(ValueError):
        evaluator.weights[0] = 9
    np.testing.assert_equal(w, [2, 3, 5, 20, 40])


def test_batch_cache_and_no_mutation():
    cc = code(5)
    adapter = ColorCodePathGap(cc)
    evaluator = adapter.evaluator
    n = len(adapter.qubit_ids)
    rng = np.random.default_rng(8)
    masks = rng.integers(0, 2, (10, n))
    saved = masks.copy()
    weights = evaluator.weights.copy()
    graph_edges = [g.get_edgelist() for g in evaluator._graphs]
    result = evaluator.evaluate(masks)
    for i, row in enumerate(masks):
        np.testing.assert_equal(
            result.phi_by_color[i], evaluator.evaluate(row).phi_by_color
        )
    np.testing.assert_equal(evaluator.evaluate(masks[::-1]).phi, result.phi[::-1])
    first = evaluator.evaluate(masks[0], diagnostics=True)
    evaluator.evaluate(masks[1])
    again = evaluator.evaluate(masks[0], diagnostics=True)
    np.testing.assert_equal(first.phi_by_color, again.phi_by_color)
    assert first.paths_by_color == again.paths_by_color
    empty = evaluator.evaluate(np.empty((0, n)), diagnostics=True)
    assert empty.phi.shape == (0,) and empty.distance_by_color.shape == (0, 3)
    assert empty.paths_by_color == ()
    np.testing.assert_equal(masks, saved)
    np.testing.assert_equal(evaluator.weights, weights)
    assert graph_edges == [g.get_edgelist() for g in evaluator._graphs]


@pytest.mark.parametrize("comparative", [False, True])
def test_mapping_integration_and_final_candidates(comparative):
    cc = code(5, comparative_decoding=comparative)
    before = (str(cc.circuit), cc.tanner_graph.get_edgelist(), cc._dem_manager)
    adapter = ColorCodePathGap(cc)
    assert before == (str(cc.circuit), cc.tanner_graph.get_edgelist(), cc._dem_manager)
    dets, obs = cc.circuit.compile_detector_sampler(seed=938).sample(
        256, separate_observables=True
    )
    if comparative:
        dets[:, -1] = False  # actual observable never used to supply the answer
    dets_before = dets.copy()
    pred, extra = cc.decode(dets, full_output=True)
    saved = copy.deepcopy(extra)
    hard = pred.copy()
    failures = hard != obs[:, 0]
    result = adapter.evaluate(extra)
    physical = adapter.to_physical(extra)
    h, logical = physical_matrices(cc)
    checks = {check["qid"]: k for k, check in enumerate(cc.qubit_groups["anc_Z"])}
    expected_syndrome = physical.astype(int) @ h.T % 2
    for detector, (check, time) in enumerate(cc.detectors_checks_map):
        if time == 0:
            np.testing.assert_equal(
                expected_syndrome[:, checks[check["qid"]]], dets[:, detector]
            )
    np.testing.assert_equal(physical.astype(int) @ logical % 2, hard)
    # Audit every DEM column against physical H and the decoder's obs matrix.
    columns = adapter.mapping.qubit_to_column
    np.testing.assert_equal(cc.obs_matrix.toarray()[0, list(columns)], logical)
    assert tuple(columns) != tuple(range(len(columns)))
    for detector, (check, time) in enumerate(cc.detectors_checks_map):
        np.testing.assert_equal(
            cc.H.toarray()[detector, list(columns)],
            h[checks[check["qid"]]] if time == 0 else np.zeros(len(columns)),
        )
    red_pred, red_extra = cc.decode(dets, colors="r", full_output=True)
    green_pred, green_extra = cc.decode(dets, colors="g", full_output=True)
    different = np.any(red_extra["error_preds"] != green_extra["error_preds"], axis=1)
    chosen_other = (extra["best_colors"] != 0) & np.any(
        extra["error_preds"] != red_extra["error_preds"], axis=1
    )
    assert np.any(different) and np.any(chosen_other)
    i = np.flatnonzero(chosen_other)[0]
    np.testing.assert_equal(
        result.distance_by_color[i],
        adapter.evaluator.evaluate(physical[i]).distance_by_color,
    )
    forged = dict(
        extra,
        best_colors=np.zeros(256),
        weights=np.full(256, 1e9),
        candidates=red_extra["error_preds"],
    )
    np.testing.assert_equal(adapter.evaluate(forged).phi, result.phi)
    for key in saved:
        np.testing.assert_equal(extra[key], saved[key])
    np.testing.assert_equal(pred, hard)
    np.testing.assert_equal(pred != obs[:, 0], failures)
    np.testing.assert_equal(dets, dets_before)
    _, single_extra = cc.decode(dets[0], full_output=True)
    np.testing.assert_equal(adapter.evaluate(single_extra).phi, result.phi[:1])
    # Actual partial/predecoder path, whose final assembly must remain untouched.
    if comparative:  # The existing erasure predecoder requires comparative mode.
        pre_pred, pre_extra = cc.decode(
            dets,
            full_output=True,
            erasure_matcher_predecoding=True,
            partial_correction_by_predecoding=True,
        )
        pre_saved = pre_pred.copy()
        adapter.evaluate(pre_extra)
        np.testing.assert_equal(
            adapter.to_physical(pre_extra).astype(int) @ logical % 2, pre_saved
        )
        np.testing.assert_equal(pre_pred, pre_saved)


def test_composed_correction_boundary_uses_only_final_xor():
    adapter = ColorCodePathGap(code())
    partial = np.array([1, 0, 1, 0, 0, 0, 0], dtype=bool)
    remaining = np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)
    final = partial ^ remaining
    extra = {"error_preds": final, "partial": partial, "stage2": remaining}
    expected = adapter.evaluator.evaluate(adapter.mapping.to_physical(final))
    assert adapter.evaluate(extra).phi == expected.phi
    assert adapter.evaluate({"error_preds": np.zeros(7)}).phi != expected.phi


def test_error_only_indices_nonerror_instructions_shifts_and_permutation():
    # Physical signatures q0=D1 L0, q1=D2, q2=D1 D2 L0.
    dem = stim.DetectorErrorModel("""
        shift_detectors 1
        detector(0, 0) D0
        error(0.1) D1
        logical_observable L0
        error(0.1) D0 D1 L0
        detector(1, 0) D0
        error(0.1) D0 L0
    """)
    mapping = DemQubitMap.from_dem(dem, {1: (0, 2), 2: (1, 2)}, [1, 0, 1])
    assert mapping.qubit_to_column == (2, 0, 1)
    np.testing.assert_equal(mapping.to_physical([0, 1, 0]), [0, 0, 1])
    with pytest.raises(ValueError, match="bijectively"):
        DemQubitMap.from_dem(stim.DetectorErrorModel("error(.1) D9"), {0: (0,)}, [0])
    with pytest.raises(ValueError, match="missing"):
        DemQubitMap.from_dem(stim.DetectorErrorModel(), {0: (0,)}, [0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("rounds", 2),
        ("circuit_type", "rec"),
        ("cnot_schedule", [1] * 12),
        ("superdense_circuit", True),
        ("temp_bdry_type", "X"),
        ("d", 4),
        ("perfect_first_syndrome_extraction", True),
    ],
)
def test_reject_scope(field, value):
    cc = code()
    setattr(cc, field, value)
    with pytest.raises(ValueError):
        ColorCodePathGap(cc)


@pytest.mark.parametrize(
    "channel", [key for key in NoiseModel().keys() if key != "bitflip"]
)
def test_reject_every_noise_channel_including_overrides(channel):
    cc = code()
    cc.noise_model[channel] = 0.001
    with pytest.raises(ValueError, match="noise channel/override"):
        ColorCodePathGap(cc)


@pytest.mark.parametrize("p", [0, 0.5, np.nan, np.inf])
def test_reject_invalid_probability(p):
    cc = code()
    cc.noise_model["bitflip"] = p
    with pytest.raises(ValueError, match="bitflip"):
        ColorCodePathGap(cc)


def test_reject_missing_duplicate_and_unexpected_circuit_noise():
    cc = code(perfect_first_syndrome_extraction=True)
    with pytest.raises(ValueError, match="suppresses"):
        ColorCodePathGap(cc)
    cc = code()
    cc.circuit.append("X_ERROR", cc.qubit_groups["data"]["qid"], 0.03)
    with pytest.raises(ValueError, match="exactly one"):
        ColorCodePathGap(cc)
    cc = code()
    cc.circuit.append("Z_ERROR", [0], 0.01)
    with pytest.raises(ValueError, match="Unsupported noisy"):
        ColorCodePathGap(cc)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(6),
        np.zeros((1, 1, 7)),
        [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, np.nan],
        ["0"] * 7,
    ],
)
def test_bad_corrections(bad):
    adapter = ColorCodePathGap(code())
    with pytest.raises(ValueError):
        adapter.evaluator.evaluate(bad)
    with pytest.raises(ValueError):
        adapter.evaluate({"error_preds": bad})


@pytest.mark.parametrize(
    "bad", [[-1] * 7, [np.nan] * 7, [np.inf] * 7, [[1] * 7], [], [1] * 6]
)
def test_bad_weights(bad):
    adapter = ColorCodePathGap(code())
    with pytest.raises(ValueError):
        MonochromaticPathGap(adapter.evaluator.topologies, bad)


def test_malformed_graph_diagnostics():
    with pytest.raises(ValueError, match="disconnected"):
        PathTopology(3, ((0, 1),), ((0,),), 0, 2)
    with pytest.raises(ValueError, match="terminals"):
        PathTopology(2, ((0, 1),), ((0,),), 0, 0)
    with pytest.raises(ValueError, match="supports"):
        PathTopology(2, ((0, 1), (0, 1)), ((0,), (0,)), 0, 1)
    with pytest.raises(ValueError, match="endpoints"):
        PathTopology(2, ((0, 4),), ((0,),), 0, 1)
    cc = code()
    corner = cc.qubit_groups["data"][-1]
    corner["boundary"] = "r"
    with pytest.raises(ValueError, match="Malformed"):
        build_monochromatic_topologies(cc.tanner_graph)


def test_real_weights_zero_cost_witnesses_and_invalid_mapping():
    adapter = ColorCodePathGap(code())
    with pytest.raises(ValueError, match="real"):
        MonochromaticPathGap(
            adapter.evaluator.topologies, np.ones(7, dtype=complex) + 1j
        )
    result = adapter.evaluator.evaluate(np.ones(7), diagnostics=True)
    assert all(path for path in result.paths_by_color)
    np.testing.assert_equal(result.distance_by_color, np.zeros(3))
    with pytest.raises(ValueError, match="permutation"):
        DemQubitMap((0, 0))
    with pytest.raises(ValueError, match="bijectively"):
        DemQubitMap.from_dem(stim.DetectorErrorModel("error(.1) D0 L0"), {0: (0,)}, [0])


def test_vertex_permutation_preserves_stable_physical_ids():
    cc = code(5)
    ids, topologies = build_monochromatic_topologies(cc.tanner_graph)
    permutation = np.random.default_rng(18).permutation(cc.tanner_graph.vcount())
    ids2, topologies2 = build_monochromatic_topologies(
        cc.tanner_graph.permute_vertices(permutation)
    )
    assert ids == ids2
    e = np.random.default_rng(19).integers(0, 2, (5, len(ids)))
    a = MonochromaticPathGap(topologies, np.arange(len(ids))).evaluate(e)
    b = MonochromaticPathGap(topologies2, np.arange(len(ids))).evaluate(e)
    np.testing.assert_equal(a.distance_by_color, b.distance_by_color)
