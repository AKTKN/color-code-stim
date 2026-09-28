"""Native stage-1 candidate generation, with fixed original decompositions."""

import secrets
import numpy as np
import pymatching

from .color_correlated_decoding import COLORS


def resolve_native_seed(seed):
    if getattr(pymatching.Matching, 'NATIVE_PERTURBATION_VERSION', 0) != 1:
        raise RuntimeError('stage1_perturbation=True requires the native-stage1-perturbation PyMatching fork; build/install that backend')
    if seed is None:
        return secrets.randbits(64)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Native perturbation_seed must be an integer in [0, 2**64)')
    return seed


class NativeStage1Ensemble:
    """One native ensemble matcher per colour; no per-shot DEM reconstruction.

    The public decode call owns a single physical-shot position. Explicit
    offsets let all logical classes replay that shot's colour-specific draws.
    """

    def __init__(self, manager, size, alpha, seed, cache):
        self.size, self.alpha, self.seed = size, alpha, resolve_native_seed(seed)
        self.shot_position = 0
        self.matchings, self.structures = {}, {}
        for stream, color in enumerate(COLORS):
            base = manager.dems_decomposed[color]
            structure = cache.base_structure(color, 1, base.Hs[0])
            p = base.probs[0]
            self.structures[color] = structure
            self.matchings[color] = pymatching.Matching.from_check_matrix(
                structure.matrix, weights=np.log((1-p)/p), error_probabilities=p,
                apply_perturbation=True, alpha=alpha, seed=self.seed,
                ensemble_size=size, stream_id=stream,
            )

    def start(self, offset, count):
        offset = self.shot_position if offset is None else offset
        if type(offset) is not int or not 0 <= offset < 2**64 or count > 2**64-1-offset:
            raise ValueError('perturbation_shot_offset and batch length must fit uint64')
        return offset

    def decode_stage1(self, detectors, color, offset):
        structure = self.structures[color]
        result = self.matchings[color].decode_batch(
            detectors[:, structure.checks_to_keep], shot_offset=offset)
        return np.asarray(result, dtype=bool).reshape(
            len(detectors), self.size, structure.matrix.shape[1])

    def advance(self, offset, count):
        if count:
            self.shot_position = max(self.shot_position, offset + count)

    def get_state(self):
        return dict(kind='native_stage1', scheme_version=1, seed=self.seed,
                    ensemble_size=self.size, alpha=self.alpha, shot_position=self.shot_position)

    def set_state(self, state):
        current = self.get_state()
        if set(state) != set(current) or any(state[k] != current[k] for k in current if k != 'shot_position'):
            raise ValueError('Native perturbation configuration or RNG scheme version differs')
        self.shot_position = self.start(state['shot_position'], 0)
