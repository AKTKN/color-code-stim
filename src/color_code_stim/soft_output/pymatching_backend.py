"""Adapter from typed spatial topology to generic PyMatching analysis."""
import numpy as np
import pymatching
from .topology import build_spatial_topology
from .results import Stage2DecodeResult


class Stage2Backend:
    def __init__(self, manager, color):
        try:
            from pymatching.soft_output import SoftOutputConfig
        except ImportError as exc:
            raise ImportError('Swim output requires the Phase-2A PyMatching fork') from exc
        self.topology = build_spatial_topology(manager,color)
        decomp = manager.dems_decomposed[color]
        self._H = decomp.Hs[1].copy()
        self._p = decomp.probs[1].copy()
        self.matcher = pymatching.Matching.from_check_matrix(
            self._H, weights=np.log((1-self._p)/self._p))
        self.config = SoftOutputConfig(self.topology.active_rows + (-1,-1),
                                      self.topology.resolved_edges, (self.topology.terminals,))
        self.matcher.configure_soft_output(self.config)

    def matches(self, decomp):
        H = decomp.Hs[1]
        return (H.shape == self._H.shape and (H != self._H).nnz == 0
                and np.array_equal(decomp.probs[1], self._p))

    def decode(self, stage2_shots):
        result = self.matcher.decode_batch_with_soft_output(stage2_shots)
        return Stage2DecodeResult(result.predictions, result.solution_weights,
                                  result.soft_outputs[:,0])
