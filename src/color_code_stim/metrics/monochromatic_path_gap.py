"""Final-correction monochromatic path gap (a heuristic, not a logical LLR).

All physical arrays use ascending, stable Tanner ``qid`` order. Graph supports
refer to positions in that order, never mutable igraph vertex indices.
"""

from dataclasses import dataclass
from collections import Counter
from collections.abc import Mapping, Sequence
from time import perf_counter

import igraph as ig
import numpy as np
import stim

from ..config import CNOT_SCHEDULES

COLOR_ORDER = ("r", "g", "b")


def _binary(value, width):
    a = np.asarray(value)
    if a.ndim not in (1, 2) or a.shape[-1] != width:
        raise ValueError(
            f"Expected binary correction shape ({width},) or (shots, {width})"
        )
    if a.dtype.kind not in "biuf" or not np.all((a == 0) | (a == 1)):
        raise ValueError("Corrections must contain only binary 0/1 values")
    return a.astype(bool, copy=True)


@dataclass(frozen=True)
class PathTopology:
    """Immutable multigraph; supports are canonical physical array positions.

    Each qubit occurs exactly once, including in any retained self-loops.
    ``edges`` and ``supports`` have the same order. Parallel edges are distinct.
    """

    num_vertices: int
    edges: tuple[tuple[int, int], ...]
    supports: tuple[tuple[int, ...], ...]
    source: int
    target: int

    def __post_init__(self):
        object.__setattr__(self, "edges", tuple(tuple(e) for e in self.edges))
        object.__setattr__(self, "supports", tuple(tuple(s) for s in self.supports))
        if not isinstance(self.num_vertices, int) or self.num_vertices < 2:
            raise ValueError("Malformed topology: need at least two vertices")
        if self.source == self.target or any(
            not isinstance(v, (int, np.integer)) or not 0 <= v < self.num_vertices
            for v in (self.source, self.target)
        ):
            raise ValueError(
                "Malformed topology: terminals must be distinct valid vertices"
            )
        if len(self.edges) != len(self.supports) or not self.edges:
            raise ValueError(
                "Malformed topology: edges/supports mismatch or empty graph"
            )
        if any(
            len(e) != 2
            or any(
                not isinstance(v, (int, np.integer)) or not 0 <= v < self.num_vertices
                for v in e
            )
            for e in self.edges
        ):
            raise ValueError("Malformed topology: invalid edge endpoints")
        flat = [q for support in self.supports for q in support]
        if (
            any(len(s) not in (1, 2) for s in self.supports)
            or any(not isinstance(q, (int, np.integer)) or q < 0 for q in flat)
            or len(set(flat)) != len(flat)
        ):
            raise ValueError(
                "Malformed topology: supports must be disjoint singleton/pairs"
            )
        graph = ig.Graph(n=self.num_vertices, edges=self.edges, directed=False)
        if not np.isfinite(graph.distances(self.source, self.target)[0][0]):
            raise ValueError("Malformed topology: disconnected terminals")


def build_monochromatic_topologies(tanner_graph):
    """Build three suppressed physical lattices using stable qids and incidence.

    A c-colored primal link joins two qubits; suppressing its degree-two
    monochromatic vertex yields one pair edge. Missing c faces end at t_c.
    The unique qubit at the intersection of the other boundaries supplies the
    singleton s_c edge. Self-loops are retained (nonnegative loops cannot improve
    a shortest path); parallel edges are never simplified.

    Returns ``(qubit_ids, topologies)`` in ascending qid / r,g,b order.
    """
    data = sorted(tanner_graph.vs.select(pauli=None), key=lambda v: v["qid"])
    ids = tuple(v["qid"] for v in data)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Malformed physical lattice: missing or duplicate data qids")
    position = {qid: i for i, qid in enumerate(ids)}
    topologies = []
    for color in COLOR_ORDER:
        faces = sorted(
            tanner_graph.vs.select(pauli="Z", color=color), key=lambda v: v["qid"]
        )
        face_vertex = {v["qid"]: i for i, v in enumerate(faces)}
        source, target = len(faces), len(faces) + 1
        endpoints = {}
        for v in data:
            incident = [
                u for u in v.neighbors() if u["pauli"] == "Z" and u["color"] == color
            ]
            boundary = v["boundary"] or ""
            if len(incident) == 1 and color not in boundary:
                endpoints[v["qid"]] = face_vertex[incident[0]["qid"]]
            elif not incident and color in boundary:
                endpoints[v["qid"]] = target
            else:
                raise ValueError(
                    f"Malformed {color} lattice: face/boundary incidence at qid {v['qid']}"
                )
        edges, supports, used = [], [], []
        for link in tanner_graph.es.select(kind="lattice", color=color):
            qids = (link.source_vertex["qid"], link.target_vertex["qid"])
            if any(q not in position for q in qids):
                raise ValueError("Malformed lattice link: non-data endpoint")
            edges.append(tuple(endpoints[q] for q in qids))
            supports.append(tuple(position[q] for q in qids))
            used.extend(qids)
        corners = [
            v for v in data if set(v["boundary"] or "") == set(COLOR_ORDER) - {color}
        ]
        if len(corners) != 1:
            raise ValueError(f"Malformed {color} lattice: need one opposite corner")
        corner = corners[0]["qid"]
        edges.append((source, endpoints[corner]))
        supports.append((position[corner],))
        used.append(corner)
        if Counter(used) != Counter(ids):
            raise ValueError(
                f"Malformed {color} lattice: qubits do not partition into primal pairs and corner"
            )
        topologies.append(
            PathTopology(len(faces) + 2, tuple(edges), tuple(supports), source, target)
        )
    return ids, tuple(topologies)


@dataclass(frozen=True)
class PathGapResult:
    """Single: scalars and (3,) arrays; batch: (shots,) and (shots,3).

    ``minimizing_color`` breaks exact ties r, then g, then b. Optional witnesses
    are tuples of physical array positions: [color][qubit] for a single input,
    [shot][color][qubit] for batches; None by default. Path ties follow igraph.
    """

    phi: object
    phi_by_color: np.ndarray
    distance_by_color: np.ndarray
    correction_weight: object
    minimizing_color: object
    paths_by_color: object = None
    color_order: tuple[str, ...] = COLOR_ORDER


class MonochromaticPathGap:
    """Cache three physical topologies and original finite nonnegative weights.

    ``evaluate`` accepts a binary (n,) correction or (shots,n) batch, including
    (0,n). Every color uses the same final correction. O(n log n) time per shot,
    O(n) working storage plus O(shots) results (optional paths: O(shots*n)).
    """

    def __init__(self, topologies: Sequence[PathTopology], weights):
        if np.asarray(weights).dtype.kind not in "biuf":
            raise ValueError("Weights must be real finite nonnegative numbers")
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or not len(w) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(
                "Weights must be a nonempty 1D array of finite nonnegative values"
            )
        if not np.isfinite(w.sum()):
            raise ValueError("Total original weight must be finite")
        self.weights = np.frombuffer(w.tobytes(), dtype=float)
        self.topologies = tuple(topologies)
        if len(self.topologies) != 3:
            raise ValueError("Need three topologies in r,g,b order")
        self._graphs, self._support_indices = [], []
        for topology in self.topologies:
            if not isinstance(topology, PathTopology):
                raise ValueError("Expected PathTopology objects")
            flat = [q for s in topology.supports for q in s]
            if sorted(flat) != list(range(len(w))):
                raise ValueError("Topology supports must partition all physical qubits")
            self._graphs.append(ig.Graph(n=topology.num_vertices, edges=topology.edges))
            self._support_indices.append(
                np.array(
                    [
                        (s[0], s[1] if len(s) == 2 else len(w))
                        for s in topology.supports
                    ],
                    dtype=int,
                )
            )

    def evaluate(self, correction, *, diagnostics=False):
        """Evaluate without altering inputs, cached graphs, or original weights."""
        e = np.asarray(correction)
        if e.ndim not in (1, 2) or e.shape[-1] != len(self.weights):
            raise ValueError(
                f"Expected binary correction shape ({len(self.weights)},) or (shots, {len(self.weights)})"
            )
        if e.dtype.kind not in "biuf":
            raise ValueError("Corrections must contain only binary 0/1 values")
        single = e.ndim == 1
        batch = e[None, :] if single else e
        distances = np.empty((len(batch), 3))
        correction_weight = np.empty(len(batch))
        witnesses = [] if diagnostics else None
        for i, row in enumerate(batch):
            row = _binary(row, len(self.weights))
            correction_weight[i] = self.weights[row].sum()
            residual = np.append(np.where(row, 0, self.weights), 0.0)
            paths = []
            for c, (graph, topology, indices) in enumerate(
                zip(self._graphs, self.topologies, self._support_indices)
            ):
                edge_weights = residual[indices].sum(axis=1)
                distance = graph.distances(
                    topology.source,
                    topology.target,
                    weights=edge_weights,
                    algorithm="dijkstra",
                )[0][0]
                if not np.isfinite(distance):
                    raise ValueError(
                        f"Disconnected terminals in {COLOR_ORDER[c]} topology"
                    )
                distances[i, c] = distance
                if diagnostics:
                    path = graph.get_shortest_paths(
                        topology.source,
                        to=topology.target,
                        weights=edge_weights,
                        output="epath",
                    )[0]
                    if not path:
                        raise ValueError("Empty terminal path in malformed topology")
                    paths.append(
                        tuple(q for edge in path for q in topology.supports[edge])
                    )
            if diagnostics:
                witnesses.append(tuple(paths))
        phi_by_color = distances - correction_weight[:, None]
        minimizing = np.argmin(phi_by_color, axis=1)
        phi = phi_by_color[np.arange(len(batch)), minimizing]
        colors = np.asarray(COLOR_ORDER)[minimizing]
        if single:
            return PathGapResult(
                float(phi[0]),
                phi_by_color[0],
                distances[0],
                float(correction_weight[0]),
                str(colors[0]),
                witnesses[0] if diagnostics else None,
            )
        return PathGapResult(
            phi,
            phi_by_color,
            distances,
            correction_weight,
            colors,
            tuple(witnesses) if diagnostics else None,
        )


@dataclass(frozen=True)
class DemQubitMap:
    """Verified error-only DEM-column -> canonical physical-qubit permutation."""

    qubit_to_column: tuple[int, ...]

    def __post_init__(self):
        columns = tuple(self.qubit_to_column)
        if (
            not columns
            or any(not isinstance(c, (int, np.integer)) for c in columns)
            or sorted(columns) != list(range(len(columns)))
        ):
            raise ValueError(
                "DEM mapping must be a nonempty permutation of error-only columns"
            )
        object.__setattr__(self, "qubit_to_column", columns)

    @classmethod
    def from_dem(cls, dem, detector_supports, logical_support):
        """Match full detector/observable signatures against physical incidence.

        detector_supports maps absolute detector IDs to physical positions.
        logical_support is the physical support of Z_L. Flatten shifts/repeats
        and increment the column counter ONLY on error instructions.
        """
        logical = _binary(logical_support, len(logical_support))
        if logical.ndim != 1:
            raise ValueError("Logical support must be 1D")
        signatures = [set() for _ in logical]
        for detector, support in detector_supports.items():
            for q in support:
                if not 0 <= q < len(logical):
                    raise ValueError("Invalid physical detector support")
                signatures[q].symmetric_difference_update({("D", detector)})
        for q in np.flatnonzero(logical):
            signatures[q].add(("L", 0))
        lookup = {frozenset(s): q for q, s in enumerate(signatures)}
        if len(lookup) != len(logical):
            raise ValueError("Physical signatures do not identify unique qubits")
        columns = [-1] * len(logical)
        column = 0
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            signature = set()
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    item = ("D", target.val)
                elif target.is_logical_observable_id():
                    item = ("L", target.val)
                else:
                    raise ValueError("Unsupported separated DEM error mechanism")
                signature.symmetric_difference_update({item})
            q = lookup.get(frozenset(signature))
            if q is None or columns[q] != -1:
                raise ValueError(
                    "DEM columns do not bijectively match physical check/logical signatures"
                )
            columns[q] = column
            column += 1
        if column != len(logical) or -1 in columns:
            raise ValueError("DEM is missing the single physical data-error layer")
        return cls(tuple(columns))

    def to_physical(self, error_preds):
        """Map (n,) or (shots,n) error-only columns; never raw instruction IDs."""
        return _binary(error_preds, len(self.qubit_to_column))[
            ..., self.qubit_to_column
        ]


class ColorCodePathGap:
    """Read-only postprocessor for a fixed supported ColorCode experiment.

    Construct once, then call ``evaluate(extra_outputs)`` AFTER decode returns.
    Rebuild the adapter if the code configuration/circuit is changed. No decoder
    internals or per-color candidate metadata are consulted.
    """

    def __init__(self, code):
        required = {
            "circuit_type": "tri",
            "rounds": 1,
            "superdense_circuit": False,
            "temp_bdry_type": "Z",
        }
        for name, expected in required.items():
            if getattr(code, name) != expected:
                raise ValueError(f"Path gap requires {name}={expected!r}")
        if list(code.cnot_schedule) != CNOT_SCHEDULES["tri_optimal"]:
            raise ValueError("Path gap requires cnot_schedule='tri_optimal'")
        if code.d < 3 or code.d % 2 != 1:
            raise ValueError("Path gap requires odd distance d >= 3")
        p = code.noise_model["bitflip"]
        if not np.isfinite(p) or not 0 < p < 0.5:
            raise ValueError("Path gap requires 0 < bitflip < 1/2")
        for name, value in code.noise_model.items():
            if name != "bitflip" and value != 0:
                raise ValueError(f"Unsupported noise channel/override: {name}={value}")
        if code.perfect_first_syndrome_extraction:
            raise ValueError(
                "perfect_first_syndrome_extraction suppresses the required data-error layer"
            )
        start = perf_counter()
        self.qubit_ids, topologies = build_monochromatic_topologies(code.tanner_graph)
        topology_time = perf_counter() - start
        start = perf_counter()
        if len(self.qubit_ids) != (3 * code.d**2 + 1) // 4:
            raise ValueError("Not a standard triangular 6.6.6 patch")
        layers = []
        for instruction in code.circuit.flattened():
            if not stim.gate_data(instruction.name).is_noisy_gate:
                continue
            args = instruction.gate_args_copy()
            if not args or not any(args):
                continue
            if instruction.name != "X_ERROR" or args != [p]:
                raise ValueError(
                    f"Unsupported noisy circuit instruction: {instruction.name}"
                )
            layers.append(tuple(t.value for t in instruction.targets_copy()))
        if len(layers) != 1 or Counter(layers[0]) != Counter(self.qubit_ids):
            raise ValueError(
                "Circuit must contain exactly one X_ERROR layer on all data qubits"
            )
        position = {qid: q for q, qid in enumerate(self.qubit_ids)}
        data = {v["qid"]: v for v in code.tanner_graph.vs.select(pauli=None)}
        logical = np.array([data[q]["obs"] for q in self.qubit_ids], dtype=bool)
        checks = {
            (v["x"], v["y"]): tuple(
                position[u["qid"]] for u in v.neighbors() if u["pauli"] is None
            )
            for v in code.tanner_graph.vs.select(pauli="Z")
        }
        detector_supports = {}
        for detector, coords in code.circuit.get_detector_coordinates().items():
            if len(coords) == 6 and coords[-1] == 0:
                detector_supports[detector] = tuple(np.flatnonzero(logical))
            elif len(coords) == 5 and coords[3] == 2 and coords[2] in (0, 1):
                key = tuple(coords[:2])
                if key not in checks:
                    raise ValueError("Unrecognized physical detector coordinates")
                detector_supports[detector] = checks[key] if coords[2] == 0 else ()
            else:
                raise ValueError("Unrecognized detector ordering/configuration")
        # For pure X noise, DemManager's depolarizing separation is the identity.
        # Use the circuit DEM directly to avoid initializing/mutating lazy decoders.
        dem = code.circuit.detector_error_model(flatten_loops=True)
        self.mapping = DemQubitMap.from_dem(dem, detector_supports, logical)
        mapping_time = perf_counter() - start
        start = perf_counter()
        self.evaluator = MonochromaticPathGap(
            topologies, np.full(len(self.qubit_ids), np.log1p(-p) - np.log(p))
        )
        self.setup_timings = {
            "topology_setup": topology_time + perf_counter() - start,
            "mapping_setup_and_circuit_audit": mapping_time,
        }

    def to_physical(self, extra_outputs: Mapping):
        """Read ONLY final error_preds from decode(..., full_output=True)."""
        if not isinstance(extra_outputs, Mapping) or "error_preds" not in extra_outputs:
            raise ValueError(
                'Expected decode extra_outputs containing final "error_preds"'
            )
        return self.mapping.to_physical(extra_outputs["error_preds"])

    def evaluate(self, extra_outputs: Mapping, *, diagnostics=False):
        return self.evaluator.evaluate(
            self.to_physical(extra_outputs), diagnostics=diagnostics
        )
