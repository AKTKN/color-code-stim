from dataclasses import dataclass
import numpy as np

GROWTH_CONVENTION = 'sparse_blossom_final_defect_metric_balls_v1'


@dataclass(frozen=True)
class Stage2DecodeResult:
    predictions: np.ndarray
    solution_weights: np.ndarray
    swim_distances: np.ndarray
