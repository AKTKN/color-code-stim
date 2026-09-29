"""Global-mechanism BP model and independent-prior CSS projection.

By default, projection preserves each sector's marginal in the independent surrogate
whose mechanism probabilities are supplied by BP. It does not preserve the
joint BP posterior or correlations between the two projected sectors.
The optional negative-log encoding instead produces effective matching priors.
"""
from collections import OrderedDict

import numpy as np
import stim

from ..stim_utils import dem_to_parity_check

MATCHING_EPS = 1e-14
GLOBAL_BP_VERSION = 3
GLOBAL_BP_WEIGHT_RULE = "negative_log_xz_probability"


class GlobalDemProjection:
    """Keep global source IDs separate from projected CSS mechanism IDs."""

    def __init__(self, dem, temp_bdry_type):
        if temp_bdry_type not in ("X", "Z"):
            raise NotImplementedError("Global BP predecoding supports X/Z memory; Y memory is unsupported")
        self.dem = dem.flattened()
        self.H, self.observables, self.priors = dem_to_parity_check(self.dem)
        if not len(self.priors):
            from scipy.sparse import csc_matrix
            self.H = csc_matrix((self.dem.num_detectors, 0),dtype=np.uint8)
            self.observables = csc_matrix((self.dem.num_observables, 0),dtype=np.uint8)
        coords = self.dem.get_detector_coordinates()
        observable_sector = {"X": 0, "Z": 2}[temp_bdry_type]
        groups = OrderedDict()
        metadata = stim.DetectorErrorModel()
        source = 0
        for inst in self.dem:
            if inst.type != "error":
                metadata.append(inst)
                continue
            parts = {0: [], 2: []}
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    coordinate = coords.get(target.val, ())
                    if len(coordinate) < 5 or coordinate[3] not in (0, 2):
                        raise ValueError("Global CSS projection requires X/Z detector Pauli metadata")
                    parts[int(coordinate[3])].append(target)
                elif target.is_logical_observable_id():
                    parts[observable_sector].append(target)
                else:
                    raise ValueError("Global BP DEM must use undecomposed mechanisms without separators")
            for sector, targets in parts.items():
                if targets:
                    key = (sector, tuple(sorted(targets, key=str)))
                    groups.setdefault(key, []).append(source)
            source += 1
        self.metadata = metadata
        self.targets = tuple(key[1] for key in groups)
        self.sources = tuple(np.asarray(ids, dtype=np.int64) for ids in groups.values())

    def probabilities(self, probabilities):
        q = np.asarray(probabilities, dtype=float)
        if q.shape != self.priors.shape or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("Global probabilities must align with all global DEM mechanisms and lie in [0,1]")
        result = []
        for ids in self.sources:
            values = q[ids]
            if np.any(values == .5):
                result.append(.5)
                continue
            # Same XOR law, without cancellation for very small probabilities.
            log_magnitude = np.sum(np.log1p(-2*np.minimum(values, 1-values)))
            if np.count_nonzero(values > .5) % 2:
                result.append((1 + np.exp(log_magnitude)) / 2)
            else:
                result.append(-np.expm1(log_magnitude) / 2)
        return np.asarray(result)

    def project(self, probabilities, *, negative_log_weights=False):
        """Aggregate parity, retaining distinct logical labels and detector IDs.

        With negative_log_weights, encode -log(p) for each aggregated X/Z
        mechanism as an effective probability p/(1+p). Downstream log odds
        then reproduce that weight. This transformation happens before color
        decomposition, but strictly after global-to-CSS aggregation.
        """
        result = stim.DetectorErrorModel()
        for p, targets in zip(self.probabilities(probabilities), self.targets):
            if negative_log_weights:
                # Equivalent to w=-log(p), p_eff=expit(-w); stable at p=0,1.
                p = p / (1 + p)
            result.append("error", float(p), list(targets))
        result += self.metadata
        # Observable declarations retain dimensions even when a sector is empty.
        for obs in range(self.dem.num_observables):
            result.append("logical_observable", [], [stim.target_logical_observable_id(obs)])
        return result

    def reweighted(self, probabilities):
        """Replace only global mechanism probabilities, preserving source order."""
        q = np.asarray(probabilities, dtype=float)
        self.probabilities(q)  # Validate without changing any source labels.
        result = stim.DetectorErrorModel()
        source = 0
        for inst in self.dem:
            if inst.type == "error":
                result.append("error", float(q[source]), inst.targets_copy())
                source += 1
            else:
                result.append(inst)
        return result
