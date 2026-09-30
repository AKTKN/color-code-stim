"""Decoder-owned fixed matchings and a bounded exact-prior LRU.

Only internal immutable stage structures receive precomputed identities. Public
custom DEM inputs are fingerprinted by content, so changing an input array or
sparse matrix cannot return a stale weighted matching.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, issparse


@dataclass
class StageStructure:
    matrix: object
    key: tuple
    checks_to_keep: np.ndarray | None

    @classmethod
    def prepare(cls, H, stage, key=None):
        if key is None:
            snapshot = H.tocsc() if issparse(H) else csc_matrix(H)
            key = (snapshot.shape, snapshot.dtype.str, snapshot.indptr.tobytes(),
                   snapshot.indices.tobytes(), snapshot.data.tobytes())
        checks = H.tocsr().getnnz(axis=1) > 0 if stage == 1 else None
        return cls(H[checks, :] if stage == 1 else H, key, checks)


@dataclass
class CompiledMatching:
    structure: StageStructure
    matching: object


@dataclass
class StagePrior:
    """Internal immutable prior; its compiled graph lives at most for this shot."""
    structure: StageStructure
    probabilities: np.ndarray
    compiled: CompiledMatching | None = None

    def __iter__(self):
        # Consumers which need detector filtering use structure.checks_to_keep.
        yield self.structure.matrix
        yield self.probabilities


class MatchingCache:
    def __init__(self, dynamic_limit=32):
        self.dynamic_limit = dynamic_limit
        self.fixed = {}
        self.fixed_prior_keys = {}
        self.dynamic = OrderedDict()
        self.base_structures = {}
        self.detector_masks = {}

    def base_structure(self, color, stage, H):
        key = (color, stage)
        if key not in self.base_structures:
            self.base_structures[key] = StageStructure.prepare(
                H, stage, key=('base', color, stage))
        return self.base_structures[key]

    def stage2_detector_mask(self, color, width, retained_ids):
        key = (color, width)
        if key not in self.detector_masks:
            mask = np.ones(width, dtype=bool)
            mask[retained_ids] = False
            self.detector_masks[key] = mask
        return self.detector_masks[key]

    @staticmethod
    def compile(structure, p):
        weights = np.log((1 - p) / p)
        return CompiledMatching(structure, pymatching.Matching.from_check_matrix(
            structure.matrix, weights=weights))

    def get(self, color, stage, structure, p, *, fixed=False):
        if fixed:
            key = (color, stage)
            if key not in self.fixed:
                self.fixed[key] = self.compile(structure, p)
                values = np.asarray(p)
                self.fixed_prior_keys[key] = (color, stage, structure.key, values.dtype.str,
                                              values.shape, values.tobytes())
            return self.fixed[key]
        p = np.asarray(p)
        key = (color, stage, structure.key, p.dtype.str, p.shape, p.tobytes())
        if key == self.fixed_prior_keys.get((color, stage)):
            return self.fixed[(color, stage)]
        if key in self.dynamic:
            self.dynamic.move_to_end(key)
            return self.dynamic[key]
        compiled = self.compile(structure, p)
        self.dynamic[key] = compiled
        if len(self.dynamic) > self.dynamic_limit:
            self.dynamic.popitem(last=False)
        return compiled
