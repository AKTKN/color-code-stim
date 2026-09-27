from dataclasses import replace
import numpy as np
import pytest
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel
from color_code_stim.soft_output.topology import (
    BoundaryRole, Stage2RowRole, build_spatial_topology)
from color_code_stim.soft_output.reference import reference_metric


@pytest.mark.parametrize('d', [3,5,7])
@pytest.mark.parametrize('color', ['r','g','b'])
def test_boundary_geometry_and_provenance(d, color):
    cc = ColorCode(d=d,rounds=1,circuit_type='tri',noise_model=NoiseModel(bitflip=.03))
    topo = build_spatial_topology(cc.dem_manager,color)
    decomp = cc.dems_decomposed[color]
    assert sum(e.boundary_role == BoundaryRole.C_SIDE for e in topo.edges) == d
    assert sum(e.boundary_role == BoundaryRole.OPPOSITE_CORNER for e in topo.edges) == 1
    data = {q['qid']:q for q in cc.tanner_graph.vs.select(pauli=None)}
    assert len(topo.edges) == len(data)
    for row in topo.rows:
        assert row.source_coordinates
        if row.active:
            assert row.role in (Stage2RowRole.PHYSICAL_C_DETECTOR, Stage2RowRole.STAGE1_VIRTUAL)
        if row.role == Stage2RowRole.INACTIVE_PADDING:
            assert decomp.Hs[1].getrow(row.row_id).nnz == 0
    for edge in topo.edges:
        assert edge.endpoint_rows == tuple(decomp.Hs[1][:,edge.column_id].nonzero()[0])
        assert edge.original_dem_ids == tuple(decomp.error_map_matrices[1].getrow(edge.column_id).indices)
        q = data[edge.physical_qubit_id]
        assert edge.coordinates == (q['x'],q['y'])
        if edge.boundary_role == BoundaryRole.C_SIDE:
            assert color in q['boundary']
        if edge.boundary_role == BoundaryRole.OPPOSITE_CORNER:
            assert len(q['boundary']) == 2 and color not in q['boundary']
    assert reference_metric(len(topo.active_rows)+2,topo.resolved_edges,topo.terminals,
           radii=[0]*(len(topo.active_rows)+2)).distance == pytest.approx(d*np.log(.97/.03))


def test_permuted_rows_and_columns():
    cc = ColorCode(d=5,rounds=1,noise_model=NoiseModel(bitflip=.03))
    dm = cc.dem_manager
    before = build_spatial_topology(dm,'r')
    decomp = cc.dems_decomposed['r']
    H = decomp.Hs[1]
    rng = np.random.default_rng(431)
    rows, cols = rng.permutation(H.shape[0]), rng.permutation(H.shape[1])
    inverse = np.argsort(rows)
    old_rows, old_edges = decomp.stage2_rows, decomp.stage2_edges
    decomp.stage2_rows = tuple(replace(old_rows[r],row_id=i) for i,r in enumerate(rows))
    decomp.stage2_edges = tuple(replace(old_edges[c],column_id=i,
        endpoint_rows=tuple(int(inverse[r]) for r in old_edges[c].endpoint_rows)) for i,c in enumerate(cols))
    decomp.Hs = (decomp.Hs[0],H[rows,:][:,cols].tocsc())
    decomp.probs = (decomp.probs[0],decomp.probs[1][cols])
    decomp.error_map_matrices = (decomp.error_map_matrices[0],decomp.error_map_matrices[1][cols,:])
    after = build_spatial_topology(dm,'r')
    assert {e.physical_qubit_id:e.boundary_role for e in before.edges} == {
        e.physical_qubit_id:e.boundary_role for e in after.edges}
    # Radius data move with their semantic vertices, not raw row positions.
    radii = {row:float(rng.random()) for row in before.active_rows}
    a = [radii[row] for row in before.active_rows]+[0,0]
    b = [radii[int(rows[row])] for row in after.active_rows]+[0,0]
    x = reference_metric(len(a),before.resolved_edges,before.terminals,radii=a)
    y = reference_metric(len(b),after.resolved_edges,after.terminals,radii=b)
    assert x.distance == pytest.approx(y.distance)


def test_circuit_metadata_is_retained_and_topology_refused():
    cc = ColorCode(d=3,rounds=3,noise_model=NoiseModel.uniform_circuit_noise(.001))
    for c in 'rgb':
        decomp = cc.dems_decomposed[c]
        assert any(coords[2] > 0 for row in decomp.stage2_rows for coords in row.source_coordinates)
        assert all(e.boundary_role == BoundaryRole.UNCLASSIFIED for e in decomp.stage2_edges if len(e.endpoint_rows)==1)
        with pytest.raises(NotImplementedError,match='UNCLASSIFIED'):
            build_spatial_topology(cc.dem_manager,c)
