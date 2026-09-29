import numpy as np
import pytest
from scipy.special import logit
from color_code_stim import ColorCode
from color_code_stim.decoders.concat_matching_decoder import ConcatMatchingDecoder


def code(memory='Z',**kwargs):
    return ColorCode(d=3,rounds=3,temp_bdry_type=memory,p_circuit=.02,**kwargs)


def test_real_bp_zero_syndrome_skips_matching(monkeypatch):
    c=code()
    def fail(*args,**kwargs):
        raise AssertionError('Converged BP must not execute matching')
    monkeypatch.setattr(ConcatMatchingDecoder,'decode',fail)
    monkeypatch.setattr('color_code_stim.dem_utils.dem_manager.separate_depolarizing_errors',fail)
    monkeypatch.setattr('color_code_stim.dem_utils.dem_manager.DemDecomp',fail)
    det=np.zeros((4,c.circuit.num_detectors),dtype=bool)
    pred,extra=c.decode(det,bp_predecoding=True,metrics=['logical_error'],actual_observables=np.zeros(4))
    assert not pred.any() and extra['bp_converged'].all()
    assert np.ma.getmaskarray(extra['logical_error']).all()


@pytest.mark.parametrize('memory',['X','Z'])
def test_real_bp_syndrome_consistency_and_split_batches(memory):
    c=code(memory)
    det,actual=c.sample(24,seed=7)
    # Add the known no-fault syndrome explicitly; random convergence is not a test invariant.
    det=np.vstack((np.zeros((1,det.shape[1]),dtype=bool),det))
    actual=np.concatenate(([False],actual))
    corr,llr,conv=c.decode_bp(det,max_iter=1)
    h,_=c.bp_decoder._prepare_bp_inputs()
    np.testing.assert_array_equal((corr[conv] @ h.T)%2,det[conv])
    pred,extra=c.decode(det,bp_predecoding=True,metrics=['logical_error'],actual_observables=actual,bp_prms={'max_iter':1},check_validity=True)
    assert conv.any() and (~conv).any()
    np.testing.assert_array_equal(extra['bp_converged'],conv)
    np.testing.assert_array_equal(extra['logical_error'].compressed(),(pred!=actual)[~conv])
    pieces=[]
    for start,end in ((0,7),(7,len(det))):
        pieces.append(code(memory).decode(det[start:end],bp_predecoding=True,metrics=['logical_error'],actual_observables=actual[start:end],bp_prms={'max_iter':1})[0])
    np.testing.assert_array_equal(np.concatenate(pieces),pred)


@pytest.mark.parametrize('strategy',[{}, {'comparative_decoding':True},
    {'enable_colorcorrelated_decoding':True,'color_correlated_b':4},
    {'enable_cross_color_relifting':True,'remove_non_edge_like_errors':False},
    {'enable_prior_perturbation':True,'perturbation_ensemble_size':3,'perturbation_alpha':1,'perturbation_seed':19},
    {'stage1_perturbation':True,'perturbation_ensemble_size':3,'perturbation_alpha':1,'perturbation_seed':19}])
def test_fallback_matches_direct_posterior_dem_decoder_and_caps_probabilities(monkeypatch,strategy):
    c=code(**strategy)
    det,actual=c.sample(1,seed=4)
    wrapper=c.belief_concat_matching_decoder
    plan=c.dem_manager.global_projection
    raw=np.linspace(.01,.9,len(plan.priors))
    monkeypatch.setattr(wrapper.bp_decoder,'decode',lambda *args,**kwargs:
        (np.zeros((1,len(raw)),dtype=np.uint8),-logit(raw)[None,:],np.array([False])))
    q=np.clip(raw,1e-14,.5)
    manager=c.dem_manager.with_dem(plan.project(q))
    options=wrapper.options
    reference=ConcatMatchingDecoder(manager,**options)
    if strategy.get('enable_prior_perturbation') and not strategy.get('stage1_perturbation'):
        from color_code_stim.decoders.prior_perturbation import PriorPerturbationEnsemble
        reference._perturbation_ensemble=PriorPerturbationEnsemble(manager,3,1,19)
        reference._perturbation_ensemble._rng=np.random.default_rng(np.random.SeedSequence([19,0]))
    call={'perturbation_shot_offset':0} if strategy.get('stage1_perturbation') else {}
    expected,reference_extra=reference.decode(det,check_validity=True,full_output=True,**call)
    original=str(c.dem_xz)
    pred,extra=c.decode(det,bp_predecoding=True,metrics=['logical_error','weights'],actual_observables=actual,check_validity=True,full_output=True)
    np.testing.assert_array_equal(pred,expected)
    assert not extra['bp_converged'][0] and not np.ma.getmaskarray(extra['weights'])[0]
    assert str(c.dem_xz)==original
    assert all(np.all((d.probs[0]>=0)&(d.probs[0]<=.5)) for d in manager.dems_decomposed.values())
    correction=extra['concat_outputs'][0]['error_preds'].astype(np.uint8)
    physical_h=manager.H[:-manager.circuit.num_observables] if manager.comparative_decoding else manager.H
    np.testing.assert_array_equal((correction @ physical_h.T)%2,det[:,:physical_h.shape[0]])
    np.testing.assert_array_equal(extra['concat_outputs'][0]['error_preds'],reference_extra['error_preds'])
    np.testing.assert_array_equal((correction @ manager.obs_matrix.T)%2,pred.reshape(1,-1))


def test_y_rejected_even_for_empty_batch_and_unknown_metrics():
    c=code('Y')
    with pytest.raises(NotImplementedError,match='Y memory'):
        c.decode(np.empty((0,c.circuit.num_detectors)),bp_predecoding=True)
    c=code()
    with pytest.raises(ValueError,match='metrics'):
        c.decode(np.zeros((1,c.circuit.num_detectors)),bp_predecoding=True,metrics=['nonsense'])


def test_save_load_and_empty_batches(tmp_path):
    c=code(enable_prior_perturbation=True,perturbation_ensemble_size=3,perturbation_alpha=.3,perturbation_seed=19)
    det,_=c.sample(12,seed=7)
    c.decode(det[:5],bp_predecoding=True,bp_prms={'max_iter':1})
    path=tmp_path/'decoder.pkl';c.save(path)
    restored=ColorCode.load(path)
    np.testing.assert_array_equal(c.decode(det[5:],bp_predecoding=True,bp_prms={'max_iter':1}),restored.decode(det[5:],bp_predecoding=True,bp_prms={'max_iter':1}))
    out,extra=restored.decode(det[:0],bp_predecoding=True,metrics=['weights'])
    assert out.shape==(0,) and extra['bp_converged'].shape==(0,)


@pytest.mark.parametrize('native',[False,True])
def test_physical_shot_ids_preserved_with_skipped_shots_and_new_decoders(native):
    kwargs=dict(enable_prior_perturbation=True,stage1_perturbation=native,
                perturbation_ensemble_size=3,perturbation_alpha=1.,perturbation_seed=19)
    c=code(**kwargs)
    det,_=c.sample(8,seed=7)
    det=np.vstack((np.zeros((1,det.shape[1]),dtype=bool),det))
    expected=c.decode(det,bp_predecoding=True,bp_prms={'max_iter':1})
    parts=[]
    for start,end in ((0,4),(4,len(det))):
        parts.append(code(**kwargs).decode(det[start:end],bp_predecoding=True,
            bp_prms={'max_iter':1},bp_shot_offset=start))
    np.testing.assert_array_equal(np.concatenate(parts),expected)


def test_zero_noise_full_output_preserves_observable_dimension():
    c=ColorCode(d=3,rounds=1)
    det=np.zeros((3,c.circuit.num_detectors),dtype=bool)
    pred,extra=c.decode(det,bp_predecoding=True,full_output=True)
    assert pred.shape==(3,) and not pred.any()
    assert extra['bp_converged'].all() and np.ma.getmaskarray(extra['weights']).all()


@pytest.mark.parametrize('options,call,error',[
    ({},{'metrics':['swim_distance']},ValueError),
    ({'comparative_decoding':True},{'compute_swim_distance':True},NotImplementedError),
    ({'enable_prior_perturbation':True},{'colors':['r']},ValueError),
    ({},{'actual_observables':np.zeros((1,2))},ValueError),
])
def test_invalid_options_rejected_even_when_all_bp_shots_converge(options,call,error):
    c=code(**options)
    with pytest.raises(error):
        c.decode(np.zeros((2,c.circuit.num_detectors)),bp_predecoding=True,**call)
