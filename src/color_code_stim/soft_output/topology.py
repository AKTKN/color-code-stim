"""Typed stage-2 provenance and the validated spatial two-terminal topology."""
from dataclasses import dataclass, replace
from enum import Enum
import numpy as np


class Stage2RowRole(str, Enum):
    PHYSICAL_C_DETECTOR = 'PHYSICAL_C_DETECTOR'
    STAGE1_VIRTUAL = 'STAGE1_VIRTUAL'
    # Historical H2 retains original other-color rows as zero padding. These
    # are not c detectors or virtual constraints and must not be mislabelled.
    INACTIVE_PADDING = 'INACTIVE_PADDING'


class BoundaryRole(str, Enum):
    C_SIDE = 'C_SIDE'
    OPPOSITE_CORNER = 'OPPOSITE_CORNER'
    TEMPORAL_INITIAL = 'TEMPORAL_INITIAL'
    TEMPORAL_FINAL = 'TEMPORAL_FINAL'
    OTHER = 'OTHER'
    UNCLASSIFIED = 'UNCLASSIFIED'


@dataclass(frozen=True)
class Stage2RowMeta:
    row_id: int
    role: Stage2RowRole
    source_id: int
    source_detector_ids: tuple[int, ...]
    source_coordinates: tuple[tuple[float, ...], ...]
    active: bool


@dataclass(frozen=True)
class Stage2EdgeMeta:
    column_id: int
    original_dem_ids: tuple[int, ...]
    endpoint_rows: tuple[int, ...]
    endpoint_coordinates: tuple[tuple[tuple[float, ...], ...], ...]
    boundary_role: BoundaryRole | None
    physical_qubit_id: int | None = None
    coordinates: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Stage2Topology:
    color: str
    rows: tuple[Stage2RowMeta, ...]
    edges: tuple[Stage2EdgeMeta, ...]
    active_rows: tuple[int, ...]
    resolved_edges: tuple[tuple[int, int, int, float], ...]
    terminals: tuple[int, int]


def decomposition_metadata(decomp):
    """Freeze row creation provenance and sorted-column original DEM mapping."""
    H = decomp.Hs[1]
    coords = decomp.org_dem.get_detector_coordinates()
    row_counts = np.asarray(H.getnnz(axis=1)).ravel()
    rows = []
    for row in range(H.shape[0]):
        if row in decomp._stage2_virtual_row_sources:
            source, detectors = decomp._stage2_virtual_row_sources[row]
            role = Stage2RowRole.STAGE1_VIRTUAL
        else:
            source, detectors = row, (row,)
            role = (Stage2RowRole.PHYSICAL_C_DETECTOR
                    if coords[row][4] == {'r':0,'g':1,'b':2}[decomp.color]
                    else Stage2RowRole.INACTIVE_PADDING)
            if role == Stage2RowRole.INACTIVE_PADDING and row_counts[row]:
                raise ValueError('Non-c physical row unexpectedly active in H2')
        rows.append(Stage2RowMeta(row, role, source, detectors,
                    tuple(tuple(coords[d]) for d in detectors), bool(row_counts[row])))
    edges = []
    mapping = decomp.error_map_matrices[1]
    for col in range(H.shape[1]):
        endpoints = tuple(map(int, H.indices[H.indptr[col]:H.indptr[col+1]]))
        original = tuple(map(int, mapping.indices[mapping.indptr[col]:mapping.indptr[col+1]]))
        edges.append(Stage2EdgeMeta(col, original, endpoints,
                     tuple(rows[r].source_coordinates for r in endpoints),
                     BoundaryRole.UNCLASSIFIED if len(endpoints) == 1 else None))
    return tuple(rows), tuple(edges)


def physical_error_map(manager):
    """Independent Tanner check/observable incidence audit for data-only X errors.

    Uses physical check adjacency, not stage-2 row order or surviving row roles.
    Only detector rows with actual error incidence participate; repeated
    perfect-measurement rows remain in the stored original DEM metadata.
    """
    H = manager.H.tocsc()
    active = set(map(int, np.flatnonzero(np.asarray(H.getnnz(axis=1)).ravel())))
    signature_to_qubit = {}
    for q in manager.tanner_graph.vs.select(pauli=None):
        neighbors = {v.index for v in q.neighbors()}
        syndrome = tuple(r for r in sorted(active)
                         if manager.detectors_checks_map[r][0].index in neighbors)
        signature = (syndrome, (int(bool(q['obs'])),))
        if signature in signature_to_qubit:
            raise ValueError('Physical single-qubit error signatures are not unique')
        signature_to_qubit[signature] = q
    output = {}
    obs = manager.obs_matrix.tocsc()
    for col in range(H.shape[1]):
        signature = (tuple(map(int,H.indices[H.indptr[col]:H.indptr[col+1]])),
                     tuple(map(int,obs[:,col].toarray().ravel())))
        if signature not in signature_to_qubit:
            raise ValueError('DEM error is not an audited single-qubit X mechanism')
        output[col] = signature_to_qubit[signature]
    if len(output) != len(signature_to_qubit) or len({q['qid'] for q in output.values()}) != len(output):
        raise ValueError('Original DEM and physical qubits are not bijective')
    return output


def build_spatial_topology(manager, color):
    if not manager.swim_data_only:
        raise NotImplementedError('Swim topology requires validated single-round triangular data-only X noise; boundary roles remain UNCLASSIFIED')
    decomp = manager.dems_decomposed[color]
    rows, edges = decomp.stage2_rows, decomp.stage2_edges
    physical = physical_error_map(manager)
    active_rows = tuple(r.row_id for r in rows if r.active)
    vertices = {row:i for i,row in enumerate(active_rows)}
    terminals = (len(vertices),len(vertices)+1)
    weights = np.log((1-decomp.probs[1])/decomp.probs[1])
    resolved, classified = [], []
    counts = {BoundaryRole.C_SIDE:0, BoundaryRole.OPPOSITE_CORNER:0}
    for edge in edges:
        if len(edge.endpoint_rows) not in (1,2) or len(edge.original_dem_ids) != 1:
            raise ValueError('Stage-2 columns must be graphlike with unique physical provenance')
        qubit = physical[edge.original_dem_ids[0]]
        endpoints = [vertices[r] for r in edge.endpoint_rows]
        role = None
        if len(endpoints) == 1:
            row_role = rows[edge.endpoint_rows[0]].role
            if row_role == Stage2RowRole.STAGE1_VIRTUAL:
                role = BoundaryRole.C_SIDE
                endpoints.append(terminals[0])
            elif row_role == Stage2RowRole.PHYSICAL_C_DETECTOR:
                role = BoundaryRole.OPPOSITE_CORNER
                endpoints.append(terminals[1])
            else:
                raise ValueError('Inactive padding cannot terminate an error edge')
            counts[role] += 1
        boundary = qubit['boundary'] or ''
        expected = (BoundaryRole.C_SIDE if color in boundary else
                    BoundaryRole.OPPOSITE_CORNER if len(boundary) == 2 else None)
        if role != expected:
            raise ValueError('Row-role boundary classifier disagrees with Tanner geometry')
        classified.append(replace(edge, boundary_role=role, physical_qubit_id=qubit['qid'],
                                  coordinates=(qubit['x'],qubit['y'])))
        resolved.append((edge.column_id,*endpoints,float(weights[edge.column_id])))
    distance = sum(color in (q['boundary'] or '') for q in manager.tanner_graph.vs.select(pauli=None))
    if counts != {BoundaryRole.C_SIDE:distance, BoundaryRole.OPPOSITE_CORNER:1}:
        raise ValueError(f'Boundary count gate failed: {counts}, expected {distance},1')
    return Stage2Topology(color, rows, tuple(classified), active_rows, tuple(resolved), terminals)
