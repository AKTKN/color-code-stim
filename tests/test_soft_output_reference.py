import numpy as np
import pytest
import pymatching
from pymatching.soft_output import SoftOutputConfig, metric_from_radii
from color_code_stim.soft_output.reference import reference_metric


@pytest.mark.parametrize('radii, expected', [([0,0],5), ([1,2],2), ([5,0],0)])
def test_partial_and_terminal_contacts(radii, expected):
    edges = ((0,0,1,5),)
    result = reference_metric(2, edges, (0,1), radii=radii)
    assert result.distance == expected
    assert result.edge_ids == (0,)
    actual, weights = metric_from_radii(SoftOutputConfig((-1,-1),edges,((0,1),)), radii)
    assert actual == [expected]
    assert weights == [expected]


def test_alternative_zero_route():
    edges = ((0,0,1,10), (1,0,2,1), (2,2,1,1))
    result = reference_metric(3,edges,(0,1),radii=[0,0,1])
    assert result.residual_weights[0] == 10
    assert result.distance == 0
    assert result.edge_ids == (1,2)
    out, residual = metric_from_radii(SoftOutputConfig((-1,)*3,edges,((0,1),)),[0,0,1])
    assert out == [0]
    assert residual == [10,0,0]


def test_explicit_intervals_and_disconnection():
    result = reference_metric(3, ((7,0,1,5),), (0,1), intervals={7: [(0,1),(3,5),(4,5)]})
    assert result.distance == 2
    assert result.edge_ids == (7,)
    result = reference_metric(3, ((7,0,1,0),), (0,2), radii=[0,0,0])
    assert np.isinf(result.distance)
    assert result.edge_ids is None
    with pytest.raises(ValueError, match='exactly one'):
        reference_metric(2, ((0,0,1,1),), (0,1))


@pytest.mark.parametrize('seed', range(40))
def test_random_graphs_and_pairs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3,10))
    endpoints = [(u,u+1) for u in range(n-1)]
    endpoints += [tuple(map(int, rng.choice(n,2,replace=False))) for _ in range(n)]
    edges = tuple((i,u,v,float(rng.integers(0,30))/4) for i,(u,v) in enumerate(endpoints))
    radii = rng.integers(0,20,size=n)/4
    pairs = ((0,n-1), (1,n-1), (0,1))
    outputs, residual = metric_from_radii(SoftOutputConfig((-1,)*n,edges,pairs), radii)
    for i,pair in enumerate(pairs):
        ref = reference_metric(n,edges,pair,radii=radii)
        assert outputs[i] == pytest.approx(ref.distance)
        np.testing.assert_allclose(residual, list(ref.residual_weights.values()), atol=1e-12)
        assert ref.edge_ids is not None
        assert sum(ref.residual_weights[e] for e in ref.edge_ids) == pytest.approx(ref.distance)


@pytest.mark.parametrize('seed', range(10))
def test_actual_mwpm_growth_against_reference(seed):
    rng = np.random.default_rng(seed)
    n = 6
    m = pymatching.Matching()
    edges = []
    for u in range(n-1):
        w = float(rng.integers(1,10))
        m.add_edge(u,u+1,weight=w,fault_ids=u)
        edges.append((u,u,u+1,w))
    m.add_boundary_edge(0,weight=3,fault_ids=n)
    m.add_boundary_edge(n-1,weight=4,fault_ids=n+1)
    edges.extend(((n,n,0,3.),(n+1,n-1,n+1,4.)))
    cfg = SoftOutputConfig(tuple(range(n))+(-1,-1), edges, ((n,n+1),))
    m.configure_soft_output(cfg)
    shots = rng.integers(0,2,size=(16,n),dtype=np.uint8)
    output = m.decode_batch_with_soft_output(shots, include_radii=True)
    for i,radii in enumerate(output.radii):
        ref = reference_metric(n+2, edges, (n,n+1), radii=radii)
        assert output.soft_outputs[i,0] == pytest.approx(ref.distance, abs=1e-10)
