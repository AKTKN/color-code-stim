"""Fixed ensemble of common X/Z DEM priors for concatenated matching."""

import numpy as np

from .color_correlated_decoding import COLORS, MATCHING_EPS, OriginalDemProbabilityBuilder
from ..dem_utils.dem_decomp import DemDecomp


class PriorPerturbationEnsemble:
    def __init__(self, manager, size: int, alpha: float, seed: int | None):
        self._manager = manager
        self._builder = OriginalDemProbabilityBuilder(manager)
        rng = np.random.default_rng(seed)
        self.probabilities = [self._builder.base_q.copy()]
        self.decompositions = [manager.dems_decomposed]
        for _ in range(1, size):
            xi = rng.uniform(-1, 1, len(self._builder.base_q))
            q = (self._builder.base_q.copy() if alpha == 0 else
                 np.clip(self._builder.base_q * (1 + alpha * xi),
                         MATCHING_EPS, 1 - MATCHING_EPS))
            self.probabilities.append(q)
            dem = self._builder.build(q)
            self.decompositions.append({
                color: DemDecomp(
                    org_dem=dem, color=color,
                    remove_non_edge_like_errors=manager.remove_non_edge_like_errors,
                ) for color in COLORS
            })
