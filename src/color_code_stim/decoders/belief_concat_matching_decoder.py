"""Global-mechanism BP, early termination, and posterior CSS matching."""
from copy import deepcopy
import numpy as np
from scipy.special import expit
from ..dem_utils.global_dem import GLOBAL_BP_VERSION, GLOBAL_BP_WEIGHT_RULE

from .base import BaseDecoder
from .bp_decoder import BPDecoder
from .concat_matching_decoder import ConcatMatchingDecoder
from .experiment_metrics import METRIC_NAMES
from .prior_perturbation import PriorPerturbationEnsemble


class BeliefConcatMatchingDecoder(BaseDecoder):
    def __init__(self, dem_manager, bp_cache_inputs=True, decoder_options=None):
        self.dem_manager = dem_manager
        self.circuit_type = dem_manager.circuit_type
        self.num_obs = dem_manager.circuit.num_observables
        self.comparative_decoding = dem_manager.comparative_decoding
        self._concat_decoder = None
        self.options = dict(decoder_options or {})
        # BP supplies generation information only for stage 1. These effective
        # policies override ordinary-decoder options without mutating ColorCode.
        self.options['use_original_prior_for_stage2'] = True
        self.options['color_correlated_weight_basis'] = 'original_dem'
        self.bp_decoder = BPDecoder(dem_manager, cache_inputs=bp_cache_inputs)
        self.shot_position = 0
        import secrets
        seed = self.options.get('perturbation_seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError('Global BP perturbation_seed must fit uint64')
        self.resolved_seed = secrets.randbits(64) if seed is None else int(seed)
        self.rng = np.random.default_rng(self.resolved_seed)

    def supports_comparative_decoding(self):
        return True

    def supports_predecoding(self):
        return True

    def get_bp_decoder(self):
        return self.bp_decoder

    @property
    def concat_decoder(self):
        """The ordinary decoder; posterior fallback uses a separate shot view."""
        if self._concat_decoder is None:
            self._concat_decoder = ConcatMatchingDecoder(self.dem_manager, **self.options)
        return self._concat_decoder

    def get_concat_decoder(self):
        return self.concat_decoder

    def get_state(self):
        return dict(kind='global_bp', version=GLOBAL_BP_VERSION,
                    weight_rule=GLOBAL_BP_WEIGHT_RULE, probability_cap=.5, seed=self.resolved_seed,
                    stage2_prior='original_physical', selection_prior='original_physical',
                    options=self.options, shot_position=self.shot_position,
                    rng=deepcopy(self.rng.bit_generator.state))

    def set_state(self, state):
        current = self.get_state()
        if set(state) != set(current) or any(state[k] != current[k] for k in (
                'kind','version','weight_rule','probability_cap','options','stage2_prior','selection_prior')):
            raise ValueError('Global BP configuration differs (including weighting version/rule)')
        if type(state['shot_position']) is not int or not 0 <= state['shot_position'] < 2**64:
            raise ValueError('Invalid BP shot position')
        if type(state['seed']) is not int or not 0 <= state['seed'] < 2**64:
            raise ValueError('Invalid BP seed')
        if self.options.get('perturbation_seed') is not None and state['seed'] != self.resolved_seed:
            raise ValueError('BP seed differs')
        self.resolved_seed = state['seed']
        self.rng.bit_generator.state = deepcopy(state['rng'])
        self.shot_position = state['shot_position']

    def decode(self, detector_outcomes, colors='all', logical_value=None,
               bp_prms=None, full_output=False, metrics=None,
               actual_observables=None, baseline_predictions=None,
               compute_swim_distance=False, candidate_scorer=None,
               perturbation_shot_offset=None, bp_shot_offset=None, **kwargs):
        manager = self.dem_manager
        if manager.circuit_type == 'cult+growing':
            raise NotImplementedError('Global BP does not support cultivation postselection')
        projection = manager.global_projection  # Also rejects Y before any shots.
        detectors = np.asarray(detector_outcomes, dtype=bool)
        if detectors.ndim not in (1,2):
            raise ValueError('detector_outcomes must be 1D or 2D')
        if detectors.ndim == 1:
            detectors = detectors[None, :]
        shots, count = len(detectors), manager.circuit.num_observables
        if detectors.shape[1] != manager.circuit.num_detectors:
            raise ValueError('detector_outcomes must match the circuit detectors')
        selected_colors = ['r', 'g', 'b'] if colors == 'all' else [colors] if isinstance(colors, str) else list(colors)
        if not selected_colors or any(color not in ('r', 'g', 'b') for color in selected_colors):
            raise ValueError('colors must select r, g, or b')
        if any(self.options.get(option, False) for option in (
                'enable_colorcorrelated_decoding', 'enable_cross_color_relifting', 'enable_prior_perturbation')):
            if len(selected_colors) != 3 or set(selected_colors) != {'r', 'g', 'b'}:
                raise ValueError('This decoding strategy requires all three colors')
        if logical_value is not None and np.asarray(logical_value).size != count:
            raise ValueError(f'logical_value must have length {count}')
        if compute_swim_distance and (manager.comparative_decoding or not manager.swim_data_only
                or kwargs.get('erasure_matcher_predecoding') or kwargs.get('partial_correction_by_predecoding')):
            raise NotImplementedError('Swim requires supported non-comparative data-only decoding')
        if bp_shot_offset is not None and perturbation_shot_offset is not None and bp_shot_offset != perturbation_shot_offset:
            raise ValueError('BP and perturbation offsets disagree')
        supplied = bp_shot_offset if bp_shot_offset is not None else perturbation_shot_offset
        offset = self.shot_position if supplied is None else supplied
        if type(offset) is not int or not 0 <= offset < 2**64 or shots > 2**64-1-offset:
            raise ValueError('BP shot offset and length must fit uint64')
        if perturbation_shot_offset is not None and not self.options.get('stage1_perturbation', False):
            raise ValueError('perturbation_shot_offset requires stage1_perturbation=True')
        names = tuple(metrics or ())
        if isinstance(metrics,str) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names) or set(names)-METRIC_NAMES-{'bp_converged'}:
            raise ValueError('Unknown or duplicate BP metrics')
        requested = tuple(name for name in names if name != 'bp_converged')
        if 'logical_gap' in requested and (not manager.comparative_decoding or logical_value is not None):
            raise ValueError('logical_gap requires all comparative classes')
        if 'color_correlated_run' in requested and not self.options.get('enable_colorcorrelated_decoding'):
            raise ValueError('color_correlated_run requires color-correlated decoding')
        if 'relift_run' in requested and not self.options.get('enable_cross_color_relifting'):
            raise ValueError('relift_run requires cross-color relifting')
        if 'swim_distance' in requested and (count != 1 or manager.comparative_decoding
                or (not compute_swim_distance and candidate_scorer is None)):
            raise ValueError('swim_distance requires a supported non-comparative scorer')
        if requested and (kwargs.get('erasure_matcher_predecoding') or kwargs.get('partial_correction_by_predecoding')):
            raise NotImplementedError('metrics does not support erasure predecoding')
        actual = None
        if actual_observables is not None:
            actual = np.asarray(actual_observables, dtype=bool)
            if count == 1 and actual.shape == (shots,):
                actual = actual[:, None]
            if actual.shape != (shots, count):
                raise ValueError('actual_observables must align by shot and observable')
        if set(requested) & {'logical_error','default_logical_error','effect_by_color_correlated_decoding'} and actual is None:
            raise ValueError('Error metrics require actual_observables')
        pred, llrs, converged = self.bp_decoder.decode(detectors, **(bp_prms or {}))
        converged = np.asarray(converged,dtype=bool).reshape(shots)
        prediction = np.asarray((np.asarray(pred,dtype=np.uint8) @ projection.observables.T) % 2,dtype=bool)
        # A claimed convergence must satisfy exactly the BP checks, not a guessed correction.
        bp_h, _ = self.bp_decoder._prepare_bp_inputs()
        measured = detectors[:, :bp_h.shape[0]]
        if np.any(converged & ~np.all(np.asarray((pred @ bp_h.T) % 2,dtype=bool) == measured,axis=1)):
            raise RuntimeError('BP convergence flag disagrees with the syndrome')
        scalar = {}
        for name in requested:
            dtype = bool if name in ('logical_error','default_logical_error') else np.uint8 if name in (
                'effect_by_color_correlated_decoding','better_weight_by_color_correlated_decoding',
                'color_correlated_run','relift_run') else float
            scalar[name] = np.ma.masked_all(shots,dtype=dtype)
        diagnostics = np.empty(shots,dtype=object)
        diagnostics[:] = None
        old_rng = deepcopy(self.rng.bit_generator.state)
        try:
            for i in np.flatnonzero(~converged):
                q = expit(-llrs[i])
                local = manager.with_bp_stage1_dem(projection.project(q, negative_log_weights=True))
                decoder = ConcatMatchingDecoder(local, **self.options)
                native = self.options.get('stage1_perturbation', False)
                if self.options.get('enable_prior_perturbation') and not native:
                    ensemble = PriorPerturbationEnsemble(local, self.options['perturbation_ensemble_size'],
                        self.options['perturbation_alpha'], self.options.get('perturbation_seed'))
                    self.rng = np.random.default_rng(np.random.SeedSequence([self.resolved_seed,offset+int(i)]))
                    ensemble._rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
                    decoder._perturbation_ensemble = ensemble
                baseline = None
                if self.options.get('enable_colorcorrelated_decoding'):
                    baseline = ConcatMatchingDecoder(local, color_correlated_weight_basis='original_dem').decode(
                        detectors[i:i+1],colors=colors,logical_value=logical_value)
                scorer = candidate_scorer
                owner = getattr(candidate_scorer,'__self__',None)
                if owner is not None and hasattr(owner,'for_dem_manager'):
                    scorer = owner.for_dem_manager(local).score_candidate
                call = dict(kwargs, colors=colors, logical_value=logical_value,
                            compute_swim_distance=compute_swim_distance)
                if requested:
                    call.pop('return_candidate_data',None)
                if native:
                    call['perturbation_shot_offset'] = offset+int(i)
                if requested:
                    result, values = decoder.decode(detectors[i:i+1], **call,
                        metrics=requested, actual_observables=None if actual is None else actual[i:i+1],
                        baseline_predictions=baseline, candidate_scorer=scorer)
                    for name in requested:
                        scalar[name][i] = values[name][0]
                    if full_output:
                        # Replaying with a fresh decoder retains the same candidate draws.
                        diagnostic_decoder = ConcatMatchingDecoder(local, **self.options)
                        if not native and decoder._perturbation_ensemble is not None:
                            diagnostic_decoder._perturbation_ensemble = PriorPerturbationEnsemble(local,
                                self.options['perturbation_ensemble_size'],self.options['perturbation_alpha'],
                                self.options.get('perturbation_seed'))
                            diagnostic_decoder._perturbation_ensemble._rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
                        _, diagnostics[i] = diagnostic_decoder.decode(detectors[i:i+1], **call,full_output=True)
                elif full_output:
                    result, diagnostics[i] = decoder.decode(detectors[i:i+1], **call,full_output=True)
                else:
                    result = decoder.decode(detectors[i:i+1], **call)
                prediction[i] = np.asarray(result,dtype=bool).reshape(count)
                if decoder._perturbation_ensemble is not None and not native:
                    self.rng.bit_generator.state = deepcopy(decoder._perturbation_ensemble._rng.bit_generator.state)
        except Exception:
            self.rng.bit_generator.state = old_rng
            raise
        self.shot_position = max(self.shot_position,offset+shots)
        output = prediction.ravel() if count == 1 else prediction
        if full_output or metrics is not None:
            extra = dict(scalar, bp_converged=converged)
            if full_output:
                # Original corrections have a different source space per BP shot.
                # Keep full per-shot records; skipped matching records are None.
                extra['concat_outputs'] = diagnostics
                for key in ('weights','best_colors','error_preds','baseline_predictions','logical_gaps'):
                    if key in extra:
                        continue
                    available = [row[key] for row in diagnostics if row is not None and key in row]
                    if available and available[0] is not None:
                        first = np.asarray(available[0])
                        aligned = np.ma.masked_all((shots,) + first.shape[1:],dtype=first.dtype)
                        for i,row in enumerate(diagnostics):
                            if row is not None and row.get(key) is not None:
                                aligned[i] = np.asarray(row[key])[0]
                        extra[key] = aligned
                if 'weights' not in extra:
                    extra['weights'] = np.ma.masked_all(shots,dtype=float)
            return output, extra
        return output
