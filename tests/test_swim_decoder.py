from pathlib import Path
import numpy as np
import pytest
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def code(d=3, **kwargs):
    return ColorCode(d=d,rounds=1,circuit_type='tri',cnot_schedule='tri_optimal',
                     noise_model=NoiseModel(bitflip=.05),**kwargs)


def test_frozen_baseline():
    fixture_path = Path(__file__).resolve().parents[3] / 'implementation_artifacts/baseline/color_d3.npz'
    if not fixture_path.exists():
        pytest.skip('Workspace M0 fixture not present in standalone package checkout')
    fixture = np.load(fixture_path)
    cc = code()
    for enabled in (False, True):
        pred, extra = cc.decode(fixture['detectors'], full_output=True,compute_swim_distance=enabled)
        np.testing.assert_array_equal(pred,fixture['predictions'])
        np.testing.assert_array_equal(extra['weights'],fixture['weights'])
        np.testing.assert_array_equal(extra['best_colors'],fixture['best_colors'])
        np.testing.assert_array_equal(pred != fixture['actual_observable'],fixture['failures'])


@pytest.mark.parametrize('d', [3,5,7])
@pytest.mark.parametrize('colors', ['all','g',['b','r']])
def test_regression_shapes_direct_backend_and_cache(d,colors):
    cc = code(d)
    shots, obs = cc.sample(48,seed=20260912)
    old, off = cc.decode(shots,colors=colors,full_output=True,check_validity=True)
    new, on = cc.decode(shots,colors=colors,full_output=True,check_validity=True,compute_swim_distance=True)
    np.testing.assert_array_equal(old,new)
    for key in off:
        np.testing.assert_array_equal(off[key],on[key])
    ncolors = len(on['color_order'])
    assert on['stage2_weights_by_color'].shape == (48,ncolors)
    assert on['swim_distances_by_color'].shape == (48,ncolors)
    assert on['selected_swim_distance'].shape == (48,)
    assert np.isfinite(on['swim_distances_by_color']).all()
    assert (on['swim_distances_by_color'] >= 0).all()
    assert on['swim_bound_certified'] is False
    decoder = cc.concat_matching_decoder
    cached = dict(decoder._swim_backends)
    for i,c in enumerate(on['color_order']):
        stage1 = decoder._decode_stage1(shots,c)
        direct = decoder._decode_stage2(shots,stage1,c,compute_swim_distance=True)
        historical = decoder._decode_stage2(shots,stage1,c)
        np.testing.assert_array_equal(direct.predictions,historical[0])
        np.testing.assert_array_equal(direct.solution_weights,historical[1])
        np.testing.assert_array_equal(direct.swim_distances,on['swim_distances_by_color'][:,i])
        assert decoder._swim_backends[c] is cached[c]
    _, reverse = cc.decode(shots[::-1],colors=colors,full_output=True,compute_swim_distance=True)
    np.testing.assert_array_equal(reverse['swim_distances_by_color'][::-1],on['swim_distances_by_color'])
    chosen = [on['color_order'].index('rgb'[c]) for c in on['best_colors']]
    np.testing.assert_array_equal(on['selected_swim_distance'],on['swim_distances_by_color'][np.arange(48),chosen])


def test_empty_and_refusals():
    cc = code()
    pred, extra = cc.decode(np.empty((0,cc.circuit.num_detectors)),
                           full_output=True,compute_swim_distance=True,check_validity=True)
    assert pred.shape == (0,)
    assert extra['stage2_weights_by_color'].shape == (0,3)
    assert extra['swim_distances_by_color'].shape == (0,3)
    assert extra['selected_swim_distance'].shape == (0,)
    with pytest.raises(NotImplementedError,match='custom_dem_data'):
        cc.concat_matching_decoder.decode(np.zeros((1,cc.circuit.num_detectors)),
                                          compute_swim_distance=True,custom_dem_data={})
    cc = code(comparative_decoding=True)
    with pytest.raises(NotImplementedError,match='comparative'):
        cc.decode(np.zeros((1,cc.circuit.num_detectors)),compute_swim_distance=True)
    cc = ColorCode(d=3,rounds=3,noise_model=NoiseModel.uniform_circuit_noise(.001))
    with pytest.raises(NotImplementedError,match='UNCLASSIFIED'):
        cc.decode(np.zeros((1,cc.circuit.num_detectors)),compute_swim_distance=True)
