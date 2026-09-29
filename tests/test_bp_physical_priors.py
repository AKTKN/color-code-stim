"""Independent stage/source alignment and BP generation-prior checks."""
import numpy as np
import pytest
import stim

from color_code_stim import ColorCode
from color_code_stim.decoders.concat_matching_decoder import ConcatMatchingDecoder
from color_code_stim.decoders.color_correlated_decoding import CandidateEvaluator, ColorCorrelatedPriorReweighter
from color_code_stim.decoders.matching_cache import MatchingCache
from color_code_stim.decoders.native_stage1_perturbation import NativeStage1Ensemble
from color_code_stim.decoders.prior_perturbation import PriorPerturbationEnsemble


def keys(manager):
    return [frozenset(map(str,i.targets_copy())) for i in manager.dem_xz if i.type=='error']


def stage1_keys(decomp,source_ids):
    h=decomp.Hs[0].tocsc();mapping=decomp.error_map_matrices[0].tocsr()
    return [(tuple(h[:,j].indices),tuple(sorted(source_ids[mapping[j].indices])))
            for j in range(h.shape[1])]


@pytest.mark.parametrize('memory',['X','Z'])
@pytest.mark.parametrize('comparative',[False,True])
@pytest.mark.parametrize('superdense',[False,True])
def test_two_graph_reference_keeps_stage1_and_restores_actual_physical_graph(memory,comparative,superdense):
    c=ColorCode(d=3,rounds=3,p_circuit=.02,temp_bdry_type=memory,
                comparative_decoding=comparative,superdense_circuit=superdense)
    base=c.dem_manager;plan=base.global_projection
    before=str(base.dem_xz)
    dem=plan.project(np.linspace(.01,.9,len(plan.priors)),negative_log_weights=True)
    posterior=base.with_dem(dem);hybrid=base.with_bp_stage1_dem(dem)
    base_lookup={k:i for i,k in enumerate(keys(base))}
    source_ids=np.array([base_lookup[k] for k in keys(hybrid)])
    np.testing.assert_array_equal(hybrid.probs_xz,base.probs_xz[source_ids])
    np.testing.assert_array_equal(hybrid.bp_stage1_probs_xz,posterior.probs_xz)
    assert not np.allclose(hybrid.probs_xz,hybrid.bp_stage1_probs_xz)
    assert (hybrid.H!=base.H[:,source_ids]).nnz==0
    assert (hybrid.obs_matrix!=base.obs_matrix[:,source_ids]).nnz==0
    for color in 'rgb':
        bd=base.dems_decomposed[color];pd=posterior.dems_decomposed[color];hd=hybrid.dems_decomposed[color]
        assert (hd.Hs[0]!=pd.Hs[0]).nnz==0
        np.testing.assert_array_equal(hd.probs[0],pd.probs[0])
        assert str(hd[0])==str(pd[0])
        lookup={k:i for i,k in enumerate(stage1_keys(bd,np.arange(len(base.probs_xz))))}
        stage1_perm=np.array([lookup[k] for k in stage1_keys(hd,source_ids)])
        assert len(set(stage1_perm))==bd.Hs[0].shape[1]
        assert (hd.Hs[0]!=bd.Hs[0][:,stage1_perm]).nnz==0
        bm=bd.error_map_matrices[1].tocsr();hm=hd.error_map_matrices[1].tocsr()
        assert np.all(np.diff(bm.indptr)==1) and np.all(np.diff(hm.indptr)==1)
        lookup={source:i for i,source in enumerate(bm.indices)}
        stage2_perm=np.array([lookup[source] for source in source_ids[hm.indices]])
        row_perm=np.r_[np.arange(c.circuit.num_detectors),c.circuit.num_detectors+stage1_perm]
        assert (hd.Hs[1]!=bd.Hs[1][row_perm,:][:,stage2_perm]).nnz==0
        np.testing.assert_array_equal(hd.probs[1],bd.probs[1][stage2_perm])
        # Final score uses original probabilities, independent of BP weights.
        correction=np.ones((1,hd.Hs[1].shape[1]),dtype=bool)
        mapped,_,score,_=CandidateEvaluator(hybrid,'original_dem').evaluate(color,correction,0.)
        np.testing.assert_allclose(score,mapped@np.log((1-base.probs_xz[source_ids])/base.probs_xz[source_ids]))
    assert str(base.dem_xz)==before
    assert base.bp_stage1_probs_xz is None and not base.bp_prior_clipping


def test_all_generation_strategies_retain_posterior_priors_and_native_draws():
    c=ColorCode(d=5,rounds=2,p_depol=.05,perfect_first_syndrome_extraction=True)
    base=c.dem_manager;plan=base.global_projection
    dem=plan.project(np.linspace(.01,.9,len(plan.priors)),negative_log_weights=True)
    posterior=base.with_dem(dem);hybrid=base.with_bp_stage1_dem(dem)
    det,_=c.sample(4,seed=2930)
    native=[NativeStage1Ensemble(m,4,.6,20260929,MatchingCache()) for m in (posterior,hybrid)]
    for color in 'rgb':
        np.testing.assert_array_equal(native[0].decode_stage1(det,color,31),native[1].decode_stage1(det,color,31))
    old,new=(PriorPerturbationEnsemble(m,4,.6,20260929) for m in (posterior,hybrid))
    for _ in range(2):
        a,b=old.next_shot(),new.next_shot()
        np.testing.assert_array_equal(old.probabilities,new.probabilities)
        for member in range(4):
            for color in 'rgb':
                np.testing.assert_array_equal(a[member][color].probs[0],b[member][color].probs[0])
    guide=np.arange(len(hybrid.probs_xz))%3==0
    old,new=(ColorCorrelatedPriorReweighter(m,4) for m in (posterior,hybrid))
    for color in 'rgb':
        np.testing.assert_array_equal(old.stage1_probabilities(color,guide),new.stage1_probabilities(color,guide))


@pytest.mark.parametrize('native',[False,True])
def test_bp_policies_override_ordinary_flags_without_mutating_no_bp_decoder(native):
    c=ColorCode(d=3,rounds=3,p_circuit=.02,enable_prior_perturbation=True,
        stage1_perturbation=native,perturbation_ensemble_size=3,perturbation_alpha=.6,
        perturbation_seed=23,use_original_prior_for_stage2=False,color_correlated_weight_basis='stage2')
    det,_=c.sample(12,seed=5)
    before=c.decode(det)
    wrapper=c.belief_concat_matching_decoder
    assert wrapper.options['use_original_prior_for_stage2']
    assert wrapper.options['color_correlated_weight_basis']=='original_dem'
    _,extra=c.decode(det,bp_predecoding=True,bp_prms={'max_iter':1},full_output=True,check_validity=True)
    for row in extra['concat_outputs']:
        if row is not None:
            assert row['candidate_weight_basis']=='original_dem'
    assert c.color_correlated_weight_basis=='stage2'
    # Use a fresh seeded ordinary decoder to compare the first absolute shots.
    fresh=ColorCode(d=3,rounds=3,p_circuit=.02,enable_prior_perturbation=True,
        stage1_perturbation=native,perturbation_ensemble_size=3,perturbation_alpha=.6,
        perturbation_seed=23,use_original_prior_for_stage2=False,color_correlated_weight_basis='stage2')
    np.testing.assert_array_equal(before,fresh.decode(det))
    np.testing.assert_array_equal(c.decode(det),fresh.decode(det))


def test_previously_failed_two_qubit_example_is_corrected_with_physical_selection():
    c=ColorCode(d=5,rounds=2,p_depol=.03,perfect_first_syndrome_extraction=True)
    flat=c.circuit.flattened();index=next(i for i,inst in enumerate(flat) if inst.name=='DEPOLARIZE1')
    faulty=flat[:index]+stim.Circuit('X_ERROR(1) 0 24')+flat[index+1:]
    det,actual=faulty.compile_detector_sampler(seed=73).sample(1,separate_observables=True)
    pred,extra=c.decode(det,bp_predecoding=True,bp_prms={'bp_method':'min_sum','max_iter':20,'schedule':'parallel'},
                        full_output=True,check_validity=True)
    assert not extra['bp_converged'][0]
    assert pred[0]==actual[0,0]
    np.testing.assert_allclose(extra['weights'],[2*np.log(.98/.02)],rtol=1e-13)
