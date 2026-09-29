"""Independent enumeration of CSS marginals, not the implementation formula."""
import itertools
import numpy as np
import pytest
import stim

from color_code_stim import ColorCode
from color_code_stim.dem_utils.global_dem import GlobalDemProjection
from color_code_stim.stim_utils import dem_to_parity_check


def toy():
    return stim.DetectorErrorModel('''
        error(0.1) D0
        error(0.2) D0 D1 L0
        error(0.3) D1 L0
        detector(0,0,0,0,0) D0
        detector(0,0,0,2,1) D1
    ''')


@pytest.mark.parametrize('memory', ['X', 'Z'])
def test_projection_distribution_by_exhaustive_global_and_projected_patterns(memory):
    plan = GlobalDemProjection(toy(), memory)
    dem = plan.project(plan.priors)
    h, obs, p = dem_to_parity_check(dem)
    observable_sector = {'X':0, 'Z':1}[memory]
    for sector in range(2):
        global_dist = np.zeros(4 if sector == observable_sector else 2)
        projected_dist = np.zeros_like(global_dist)
        for bits in itertools.product((0,1), repeat=len(plan.priors)):
            probability = np.prod([q if b else 1-q for q,b in zip(plan.priors,bits)])
            parity = np.asarray(plan.H @ np.asarray(bits)) % 2
            logical = int((plan.observables @ np.asarray(bits))[0] % 2)
            index = int(parity[sector]) + (2*logical if sector == observable_sector else 0)
            global_dist[index] += probability
        for bits in itertools.product((0,1), repeat=len(p)):
            probability = np.prod([q if b else 1-q for q,b in zip(p,bits)])
            parity = np.asarray(h @ np.asarray(bits)) % 2
            logical = int((obs @ np.asarray(bits))[0] % 2)
            index = int(parity[sector]) + (2*logical if sector == observable_sector else 0)
            projected_dist[index] += probability
        np.testing.assert_allclose(projected_dist, global_dist, atol=1e-15)


def test_collision_not_overwritten_and_reweight_keeps_sources():
    plan = GlobalDemProjection(toy(), 'Z')
    p = plan.probabilities(plan.priors)
    np.testing.assert_allclose(p, [.26, .38])
    q = np.array([.4,.5,.02])
    dem = plan.reweighted(q)
    _, _, actual = dem_to_parity_check(dem)
    np.testing.assert_array_equal(actual, q)
    assert [i.targets_copy() for i in dem] == [i.targets_copy() for i in plan.dem]


@pytest.mark.parametrize('memory', ['X','Z'])
@pytest.mark.parametrize('superdense', [False,True])
def test_original_global_dem_css_marginals_match_existing_separated_circuit(memory,superdense):
    cc = ColorCode(d=3, rounds=3, temp_bdry_type=memory, p_circuit=.001,
                   superdense_circuit=superdense)
    plan = GlobalDemProjection(cc.circuit.detector_error_model(flatten_loops=True), memory)
    projected = plan.project(plan.priors)
    def probability_map(dem):
        return {frozenset(map(str,i.targets_copy())):i.args_copy()[0]
                for i in dem if i.type == 'error'}
    a,b = probability_map(projected), probability_map(cc.dem_xz)
    assert a.keys() == b.keys()
    np.testing.assert_allclose([a[k] for k in a], [b[k] for k in a], rtol=2e-12, atol=2e-15)


def test_y_and_invalid_probabilities_rejected():
    with pytest.raises(NotImplementedError, match='Y memory'):
        GlobalDemProjection(toy(), 'Y')
    plan = GlobalDemProjection(toy(), 'Z')
    for q in ([.1], [.1,float('nan'),.2], [.1,-.1,.2]):
        with pytest.raises(ValueError):
            plan.project(q)


def test_zero_half_and_tiny_probability_contraction():
    plan=GlobalDemProjection(toy(),'Z')
    np.testing.assert_array_equal(plan.probabilities([0,.5,0]),[.5,.5])
    np.testing.assert_allclose(plan.probabilities([1e-20,2e-20,3e-20]),[3e-20,5e-20],rtol=1e-15,atol=0)
    np.testing.assert_allclose(plan.probabilities([1,.2,1]),[.8,.8],rtol=1e-15)


def test_negative_log_weights_apply_after_css_aggregation():
    plan=GlobalDemProjection(toy(),'Z')
    q=np.array([.9,.2,.3])
    # Enumerated odd-parity probabilities for sources (0,1) and (1,2).
    p=np.array([.9*.8+.1*.2, .2*.7+.8*.3])
    raw=plan.project(q)
    weighted=plan.project(q,negative_log_weights=True)
    h,obs,effective=dem_to_parity_check(weighted)
    np.testing.assert_allclose(np.log((1-effective)/effective),-np.log(p),rtol=1e-14)
    assert [i.targets_copy() for i in weighted]==[i.targets_copy() for i in raw]
    # Neither early transformation nor clipping q at 0.5 has these weights.
    assert not np.allclose(effective,plan.probabilities(q/(1+q)))
    capped=plan.probabilities(np.minimum(q,.5))
    assert not np.allclose(effective,capped/(1+capped))


@pytest.mark.parametrize('p',[0.,1e-20,.1,.5,.9,.99,1.])
def test_negative_log_effective_probability_endpoints(p):
    plan=GlobalDemProjection(toy(),'Z')
    _,_,effective=dem_to_parity_check(plan.project([p,0,0],negative_log_weights=True))
    assert np.isfinite(effective).all()
    np.testing.assert_allclose(effective,[p/(1+p),0],rtol=1e-14,atol=0)
    if p:
        np.testing.assert_allclose(np.log((1-effective[0])/effective[0]),-np.log(p),rtol=1e-13,atol=1e-15)


def test_posterior_column_reordering_preserves_source_maps():
    from color_code_stim.decoders.prior_perturbation import DecompositionPlan
    c=ColorCode(d=3,rounds=3,p_circuit=.02)
    projection=c.dem_manager.global_projection
    q=np.linspace(.001,.49,len(projection.priors))
    first=c.dem_manager.with_dem(projection.project(q))
    q2=np.linspace(.49,.001,len(first.probs_xz))
    dem=stim.DetectorErrorModel()
    source=0
    for instruction in first.dem_xz:
        if instruction.type=='error':
            dem.append('error',float(q2[source]),instruction.targets_copy())
            source+=1
        else:
            dem.append(instruction)
    fresh=first.with_dem(dem)
    changed=False
    for color in 'rgb':
        symbolic=DecompositionPlan(first.dems_decomposed[color],color)
        updated=symbolic.evaluate(q2)
        direct=fresh.dems_decomposed[color]
        changed |= not np.array_equal(symbolic.base.error_map_matrices[1].toarray(),updated.error_map_matrices[1].toarray())
        for stage in (0,1):
            np.testing.assert_array_equal(updated.Hs[stage].toarray(),direct.Hs[stage].toarray())
            np.testing.assert_allclose(updated.probs[stage],direct.probs[stage],rtol=1e-14,atol=1e-15)
            np.testing.assert_array_equal(updated.error_map_matrices[stage].toarray(),direct.error_map_matrices[stage].toarray())
    assert changed, 'The test must exercise a changed stage-2 column order'
