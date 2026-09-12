"""Slow independent interval oracle for note Lemma 11.2 and Theorem 11.4.

All-pairs original distances determine ball/edge intersections. Explicit
interval union determines uncovered lengths. Original edge IDs are retained
in shortest-path witnesses even when their cost is zero. This module never
runs a matching decoder or infers a dual certificate from supplied radii.
"""
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReferenceResult:
    distance: float
    edge_ids: tuple[int, ...] | None
    residual_weights: dict[int, float]
    coverage: dict[int, tuple[tuple[float, float], ...]]


def reference_metric(
    num_vertices: int,
    edges: Sequence[tuple[int, int, int, float]],
    terminals: tuple[int, int],
    *,
    radii: Sequence[float] | None = None,
    intervals: Mapping[int, Sequence[tuple[float, float]]] | None = None,
) -> ReferenceResult:
    """Compute a supplied ball or interval metric and a labelled path witness.

    Give exactly one of ``radii`` and ``intervals``. Missing keys in an explicit
    interval mapping mean empty coverage; missing growth data as a whole is
    rejected. Edges are (unique id, endpoint u, endpoint v, original length).
    Parallel and zero-length edges are supported. Disconnected terminals
    return infinity with no witness.
    """
    if (radii is None) == (intervals is None):
        raise ValueError('Supply exactly one of radii or explicit intervals')
    if num_vertices <= 0 or any(v < 0 or v >= num_vertices for v in terminals):
        raise ValueError('Invalid vertex count or terminal')
    labels = set()
    distances = [[inf] * num_vertices for _ in range(num_vertices)]
    for u in range(num_vertices):
        distances[u][u] = 0.
    for label, u, v, w in edges:
        if label in labels or not 0 <= u < num_vertices or not 0 <= v < num_vertices:
            raise ValueError('Invalid edge identity or endpoint')
        if not isfinite(w) or w < 0:
            raise ValueError('Weights must be finite and nonnegative')
        labels.add(label)
        distances[u][v] = distances[v][u] = min(distances[u][v], w)
    if radii is not None:
        if len(radii) != num_vertices or any(not isfinite(r) or r < 0 for r in radii):
            raise ValueError('One finite nonnegative radius is required per vertex')
        for k in range(num_vertices):
            for u in range(num_vertices):
                for v in range(num_vertices):
                    distances[u][v] = min(distances[u][v], distances[u][k] + distances[k][v])
    elif set(intervals) - labels:
        raise ValueError('Coverage refers to an unknown edge')

    residual, coverage = {}, {}
    adjacency = [[] for _ in range(num_vertices)]
    for label, u, v, w in edges:
        parts = []
        if radii is not None:
            for source, radius in enumerate(radii):
                prefix = radius - distances[source][u]
                suffix = radius - distances[source][v]
                if prefix >= 0:
                    parts.append((0., min(w, prefix)))
                if suffix >= 0:
                    parts.append((max(0., w-suffix), w))
        else:
            parts = list(intervals.get(label, ()))
            if any(not isfinite(a) or not isfinite(b) or not 0 <= a <= b <= w for a,b in parts):
                raise ValueError('Invalid covered interval')
        merged = []
        for a,b in sorted(parts):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a,b))
        coverage[label] = tuple(merged)
        cost = max(0., w - sum(b-a for a,b in merged))
        residual[label] = cost
        adjacency[u].append((v,label,cost))
        adjacency[v].append((u,label,cost))

    source, target = terminals
    best = [inf] * num_vertices
    best[source] = 0.
    predecessor = {}
    queue = [(0., source)]
    while queue:
        dist,u = heappop(queue)
        if dist != best[u]:
            continue
        if u == target:
            break
        for v,label,w in adjacency[u]:
            candidate = dist+w
            if candidate < best[v]:
                best[v] = candidate
                predecessor[v] = (u,label)
                heappush(queue, (candidate,v))
    witness = None
    if isfinite(best[target]):
        path, current = [], target
        while current != source:
            current,label = predecessor[current]
            path.append(label)
        witness = tuple(reversed(path))
    return ReferenceResult(best[target], witness, residual, coverage)
