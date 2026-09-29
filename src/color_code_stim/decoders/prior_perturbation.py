"""Independent per-shot ensembles of common X/Z DEM priors."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .color_correlated_decoding import COLORS, MATCHING_EPS, OriginalDemProbabilityBuilder

from .matching_cache import StagePrior, StageStructure


class SymbolicProbabilityPlan:
    """Evaluate the existing symbolic products with the same term order.

    Grouping equal-length rows permits vectorization without padding or
    reassociating any product. Stage-2 sorting is still done for every draw.
    """

    def __init__(self, symbolic):
        self.count = len(symbolic._ems)
        lengths = {}
        for row, em in enumerate(symbolic._ems):
            lengths.setdefault(len(em.prob_vars), []).append(row)
        self.groups = []
        for rows in lengths.values():
            self.groups.append((np.asarray(rows),
                                np.stack([symbolic._ems[i].prob_vars for i in rows]),
                                np.stack([symbolic._ems[i].prob_muls for i in rows])))

    def evaluate(self, q):
        p = np.empty(self.count, dtype=float)
        for rows, variables, multipliers in self.groups:
            p[rows] = (1 - np.prod(1 - 2 * multipliers * q[variables], axis=1)) / 2
        return p


@dataclass
class PerturbedDecomposition:
    """Internal generation/mapping view, with stage-2 columns in current order."""
    Hs: tuple
    probs: tuple
    error_map_matrices: tuple
    org_prob: np.ndarray
    stage_priors: tuple
    base_decomposition: object

    def map_errors_to_org_dem(self, errors, *, stage):
        return np.asarray(errors, dtype=np.uint8) @ self.error_map_matrices[stage - 1]


class DecompositionPlan:
    def __init__(self, base, color):
        self.base, self.color = base, color
        self.probabilities = tuple(SymbolicProbabilityPlan(s) for s in base.dems_symbolic)
        self.stage1_structure = StageStructure.prepare(
            base.Hs[0], 1, key=('base', color, 1))
        order = base.dems_symbolic[1].inds_probs_sorted(base.org_prob)
        self.base_order = order
        self.stage2_structure = StageStructure.prepare(base.Hs[1], 2, key=('base', color, 2))
        self.unsorted_H2 = base.Hs[1][:, np.argsort(order)]
        self.unsorted_map2 = base.dems_symbolic[1].error_map_matrix.tocsr()
        base_map = base.error_map_matrices[1].tocsr()
        if (not np.all(np.diff(self.unsorted_map2.indptr) == 1)
                or not np.all(np.diff(base_map.indptr) == 1)
                or set(self.unsorted_map2.indices) != set(base_map.indices)):
            raise NotImplementedError('Reweighted stage-2 source set differs from base DEM')

    def evaluate(self, q):
        p1, unsorted_p2 = (plan.evaluate(q) for plan in self.probabilities)
        order = np.argsort(unsorted_p2, kind='stable')[::-1]
        p2 = unsorted_p2[order]
        if np.array_equal(order, self.base_order):
            H2, structure2 = self.base.Hs[1], self.stage2_structure
        else:
            H2 = self.unsorted_H2[:, order]
            structure2 = StageStructure.prepare(H2, 2, key=('permuted', self.color, order.tobytes()))
        return PerturbedDecomposition(
            (self.base.Hs[0], H2), (p1, p2),
            (self.base.error_map_matrices[0], self.unsorted_map2[order].tocsr()), q,
            (StagePrior(self.stage1_structure, p1), StagePrior(structure2, p2)), self.base)


class PriorPerturbationEnsemble:
    """Advance a shot-major random stream; share each member across all colors.

    M=1 and alpha=0 consume no random numbers, but still advance shot_position.
    Only the current shot's ensemble is retained.
    """

    def __init__(self, manager, size: int, alpha: float, seed: int | None):
        self._manager = manager
        self._builder = OriginalDemProbabilityBuilder(manager)
        self._plans = ({color: DecompositionPlan(manager.dems_decomposed[color], color)
                        for color in COLORS} if size > 1 and alpha > 0 else {})
        self.size, self.alpha = size, alpha
        self._rng = np.random.default_rng(seed)
        self.shot_position = 0
        self.probabilities = []
        self.decompositions = []

    def get_state(self):
        return dict(rng=deepcopy(self._rng.bit_generator.state),
                    shot_position=self.shot_position)

    def set_state(self, state):
        self._rng.bit_generator.state = deepcopy(state['rng'])
        self.shot_position = state['shot_position']

    def next_shot(self):
        self.probabilities = [self._builder.base_q]
        self.decompositions = [self._manager.dems_decomposed]
        for _ in range(1, self.size):
            if self.alpha == 0:
                q = self._builder.base_q.copy()
                decompositions = self._manager.dems_decomposed
            else:
                xi = self._rng.uniform(-1, 1, len(self._builder.base_q))
                q = np.clip(self._builder.base_q * (1 + self.alpha * xi),
                            MATCHING_EPS, .5 if self._manager.bp_prior_clipping else 1 - MATCHING_EPS)
                decompositions = {color: plan.evaluate(q)
                                  for color, plan in self._plans.items()}
            self.probabilities.append(q)
            self.decompositions.append(decompositions)
        self.shot_position += 1
        return self.decompositions
