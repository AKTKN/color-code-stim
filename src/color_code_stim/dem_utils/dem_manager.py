"""
Detector Error Model (DEM) Manager for Color Code

This module provides the DEMManager class for handling detector error model
generation, decomposition, and detector information management. It extracts
DEM-related functionality from the main ColorCode class to improve modularity
and separation of concerns.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import stim
import igraph as ig
from scipy.sparse import csc_matrix

from ..config import COLOR_LABEL, color_val_to_color
from .dem_decomp import DemDecomp
from ..stim_utils import (
    dem_to_parity_check,
    separate_depolarizing_errors,
)


class DemManager:
    """
    Manages detector error model generation, decomposition, and detector information.

    This class handles DEM generation from quantum circuits, color-based decomposition
    for concatenated decoding, and detector ID mappings for visualization and analysis.

    Attributes
    ----------
    circuit : stim.Circuit
        The quantum circuit for which to generate the DEM
    tanner_graph : ig.Graph
        The Tanner graph representation of the code
    circuit_type : str
        Type of circuit (tri, rec, rec_stability, growing, cult+growing)
    comparative_decoding : bool
        Whether to use comparative decoding
    remove_non_edge_like_errors : bool
        Whether to remove non-edge-like errors in decomposition
    dem_xz : stim.DetectorErrorModel
        The main detector error model
    H : csc_matrix
        Parity check matrix
    obs_matrix : csc_matrix
        Observable matrix
    probs_xz : np.ndarray
        Error probabilities
    detector_info : dict
        Dictionary containing detector ID mappings and metadata
    dems_decomposed : dict
        Color-decomposed DEMs for concatenated decoding
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        tanner_graph: ig.Graph,
        circuit_type: str,
        comparative_decoding: bool = False,
        remove_non_edge_like_errors: bool = True,
        swim_data_only: bool = False,
        temp_bdry_type: str = "Z",
        lazy_dem: bool = False,
    ):
        """
        Initialize DEMManager with circuit and configuration.

        Parameters
        ----------
        circuit : stim.Circuit
            The quantum circuit for which to generate the DEM
        tanner_graph : ig.Graph
            The Tanner graph representation of the code
        circuit_type : str
            Type of circuit (tri, rec, rec_stability, growing, cult+growing)
        comparative_decoding : bool, default False
            Whether to use comparative decoding
        remove_non_edge_like_errors : bool, default True
            Whether to remove non-edge-like errors in decomposition
        """
        # Store configuration
        self.temp_bdry_type = temp_bdry_type
        self.bp_prior_clipping = False
        self.bp_stage1_probs_xz = None
        self._global_projection = None
        self.swim_data_only = swim_data_only
        self.circuit = circuit
        self.tanner_graph = tanner_graph
        self.circuit_type = circuit_type
        self.comparative_decoding = comparative_decoding
        self.remove_non_edge_like_errors = remove_non_edge_like_errors

        # Generate detector information first (needed for DEM generation)
        self.detector_info = self._generate_detector_info()

        # BP needs only the original global DEM. Defer CSS preparation until
        # matching is requested, so converged BP never separates circuit noise.
        self._dem_xz = self._H = self._obs_matrix = self._probs_xz = None
        self._dems_decomposed = None
        if not lazy_dem:
            self._ensure_dem()
            self.dems_decomposed = self._decompose_dems()

    def _ensure_dem(self):
        if self._dem_xz is None:
            self.dem_xz, self.H, self.obs_matrix, self.probs_xz = self._generate_dem()

    @property
    def dem_xz(self):
        self._ensure_dem()
        return self._dem_xz

    @dem_xz.setter
    def dem_xz(self, value):
        self._dem_xz = value

    @property
    def H(self):
        self._ensure_dem()
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    @property
    def obs_matrix(self):
        self._ensure_dem()
        return self._obs_matrix

    @obs_matrix.setter
    def obs_matrix(self, value):
        self._obs_matrix = value

    @property
    def probs_xz(self):
        self._ensure_dem()
        return self._probs_xz

    @probs_xz.setter
    def probs_xz(self, value):
        self._probs_xz = value

    @property
    def dems_decomposed(self):
        if self._dems_decomposed is None:
            self._dems_decomposed = self._decompose_dems()
        return self._dems_decomposed

    @dems_decomposed.setter
    def dems_decomposed(self, value):
        self._dems_decomposed = value

    def _generate_detector_info(self) -> Dict[str, Any]:
        """
        Generate detector ID mappings and metadata.

        Extracts detector information including color-based groupings,
        cultivation/interface detector identification, and detector-to-qubit mappings.

        Returns
        -------
        dict
            Dictionary containing:
            - 'by_color': Dict mapping colors to detector ID lists
            - 'cult_ids': List of cultivation detector IDs
            - 'interface_ids': List of interface detector IDs
            - 'checks_map': List mapping detector IDs to (qubit, time) tuples
        """
        tanner_graph = self.tanner_graph

        detector_coords_dict = self.circuit.get_detector_coordinates()
        detector_ids_by_color = {
            "r": [],
            "g": [],
            "b": [],
        }
        cult_detector_ids = []
        interface_detector_ids = []
        detectors_checks_map = []

        for detector_id in range(self.circuit.num_detectors):
            coords = detector_coords_dict[detector_id]
            if self.circuit_type == "cult+growing" and len(coords) == 6:
                # The detector is in the cultivation circuit or the interface region
                flag = coords[-1]
                if flag == -1:
                    interface_detector_ids.append(detector_id)
                elif flag == -2:
                    cult_detector_ids.append(detector_id)
                    continue

            x = round(coords[0])
            y = round(coords[1])
            t = round(coords[2])
            pauli = round(coords[3])
            color = color_val_to_color(round(coords[4]))
            is_obs = len(coords) == 6 and round(coords[-1]) >= 0

            if not is_obs:
                # Ordinary X/Z detectors
                if pauli == 0:
                    name = f"{x}-{y}-X"
                    qubit = tanner_graph.vs.find(name=name)
                    color = qubit["color"]
                elif pauli == 2:
                    name = f"{x}-{y}-Z"
                    qubit = tanner_graph.vs.find(name=name)
                    color = qubit["color"]
                elif pauli == 1:
                    try:
                        name_X = f"{x + 2}-{y}-X"
                        name_Z = f"{x}-{y}-Z"
                        qubit_X = tanner_graph.vs.find(name=name_X)
                        qubit_Z = tanner_graph.vs.find(name=name_Z)
                    except ValueError:
                        name_X = f"{x}-{y}-X"
                        name_Z = f"{x - 2}-{y}-Z"
                        qubit_X = tanner_graph.vs.find(name=name_X)
                        qubit_Z = tanner_graph.vs.find(name=name_Z)
                    qubit = (qubit_X, qubit_Z)
                    color = qubit_X["color"]
                else:
                    print(coords)
                    raise ValueError(f"Invalid pauli: {pauli}")

                detectors_checks_map.append((qubit, t))

            detector_ids_by_color[color].append(detector_id)

        return {
            "by_color": detector_ids_by_color,
            "cult_ids": cult_detector_ids,
            "interface_ids": interface_detector_ids,
            "checks_map": detectors_checks_map,
        }

    def _generate_dem(
        self,
    ) -> Tuple[stim.DetectorErrorModel, csc_matrix, csc_matrix, np.ndarray]:
        """
        Generate detector error model from the quantum circuit.

        Creates the detector error model by separating depolarizing errors and
        generating the DEM. Handles special cases for cult+growing circuits
        including detector filtering and probability adjustment.

        Returns
        -------
        tuple
            (dem_xz, H, obs_matrix, probs_xz) where:
            - dem_xz: Detector error model
            - H: Parity check matrix
            - obs_matrix: Observable matrix
            - probs_xz: Error probabilities
        """
        circuit_xz = separate_depolarizing_errors(self.circuit)
        dem_xz = circuit_xz.detector_error_model(flatten_loops=True)

        if self.circuit_type == "cult+growing":
            # Remove error mechanisms that involve detectors that will be post-selected
            dem_xz_new = stim.DetectorErrorModel()
            all_detids_in_dem_xz = set()
            cult_detector_ids = self.detector_info["cult_ids"]

            for inst in dem_xz:
                keep = True
                if inst.type == "error":
                    detids = []
                    for target in inst.targets_copy():
                        if target.is_relative_detector_id():
                            detid = int(str(target)[1:])
                            detids.append(detid)
                            if (
                                detid
                                in cult_detector_ids
                                # + self.interface_detector_ids
                            ):
                                keep = False
                                continue
                    if keep:
                        all_detids_in_dem_xz.update(detids)
                if keep:
                    dem_xz_new.append(inst)
            dem_xz = dem_xz_new
            probs_dem_xz = [
                em.args_copy()[0] for em in dem_xz.flattened() if em.type == "error"
            ]

            # After removing, some detectors during growth may not be involved
            # in any error mechanisms. Although such detectors have very low probability
            # to be flipped, we add them in the DEM with an arbitrary very small
            # probability to prevent PyMatching errors.
            detids_to_add = set(range(self.circuit.num_detectors))
            detids_to_add -= all_detids_in_dem_xz
            detids_to_add -= set(cult_detector_ids)
            detids_to_add = list(detids_to_add)
            p_very_small = max(probs_dem_xz) ** 2
            for detid in detids_to_add:
                dem_xz.append(
                    "error",
                    p_very_small,
                    [stim.DemTarget.relative_detector_id(detid)],
                )

        H, obs_matrix, probs_xz = dem_to_parity_check(dem_xz)

        return dem_xz, H, obs_matrix, probs_xz

    def _decompose_dems(self) -> Dict[COLOR_LABEL, DemDecomp]:
        """
        Decompose DEM by color for concatenated decoding.

        Creates color-specific DEM decompositions for the concatenated MWPM decoder.

        Returns
        -------
        dict
            Dictionary mapping colors ('r', 'g', 'b') to DemDecomp objects
        """
        dems_decomposed = {}
        for c in ["r", "g", "b"]:
            dem_decomp = DemDecomp(
                org_dem=self.dem_xz,
                color=c,
                remove_non_edge_like_errors=self.remove_non_edge_like_errors,
            )
            dems_decomposed[c] = dem_decomp
        return dems_decomposed

    @property
    def detector_ids_by_color(self) -> Dict[COLOR_LABEL, List[int]]:
        """Get detector IDs grouped by color."""
        return self.detector_info["by_color"]

    @property
    def cult_detector_ids(self) -> List[int]:
        """Get cultivation detector IDs."""
        return self.detector_info["cult_ids"]

    @property
    def interface_detector_ids(self) -> List[int]:
        """Get interface detector IDs."""
        return self.detector_info["interface_ids"]

    @property
    def detectors_checks_map(self) -> List[Tuple[ig.Vertex, int]]:
        """Get detector-to-qubit mapping."""
        return self.detector_info["checks_map"]

    def get_decomposed_dems(
        self, color: COLOR_LABEL
    ) -> Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
        """
        Get decomposed detector error models for a specific color.

        Parameters
        ----------
        color : COLOR_LABEL
            Color ('r', 'g', or 'b') for which to get decomposed DEMs

        Returns
        -------
        tuple
            (dem1, dem2) - Stage 1 and stage 2 detector error models
        """
        dem1 = self.dems_decomposed[color][0].copy()
        dem2 = self.dems_decomposed[color][1].copy()
        return dem1, dem2

    @property
    def global_projection(self):
        """Build BP before noise separation; retain the full mechanism space."""
        if self._global_projection is None:
            from .global_dem import GlobalDemProjection
            global_dem = self.circuit.detector_error_model(flatten_loops=True, decompose_errors=False)
            for obs in range(self.circuit.num_observables):
                global_dem.append("logical_observable", [], [stim.target_logical_observable_id(obs)])
            self._global_projection = GlobalDemProjection(global_dem, self.temp_bdry_type)
        return self._global_projection

    def with_dem(self, dem):
        """Independent decoding view; never overwrite shared base priors/caches."""
        from copy import copy
        result = copy(self)
        from .global_dem import MATCHING_EPS
        # Contract global probabilities first. Regularize only projected
        # matcher priors, retaining all columns even after BP underflow.
        result.dem_xz = stim.DetectorErrorModel()
        for inst in dem:
            if inst.type == "error":
                result.dem_xz.append("error", float(np.clip(inst.args_copy()[0], MATCHING_EPS, .5)), inst.targets_copy())
            else:
                result.dem_xz.append(inst)
        result.H, result.obs_matrix, result.probs_xz = dem_to_parity_check(result.dem_xz)
        result.bp_prior_clipping = True
        result.bp_stage1_probs_xz = None
        result.dems_decomposed = result._decompose_dems()
        return result

    def with_bp_stage1_dem(self, dem):
        """Keep BP generation priors only in stage 1; score on physical priors.

        X/Z source order and stage-1 hypotheses stay in the posterior view's
        ordering. Stage 2 is rebuilt from the actual pre-BP probabilities,
        including its probability-dependent column order and source maps.
        """
        from copy import copy
        posterior = self.with_dem(dem)
        def key(inst):
            return tuple(sorted(inst.targets_copy(), key=str))
        physical_errors = [inst for inst in self.dem_xz if inst.type == 'error']
        physical = {key(inst): inst.args_copy()[0] for inst in physical_errors}
        projected = [inst for inst in posterior.dem_xz if inst.type == 'error']
        if (len(physical) != len(physical_errors)
                or len(projected) != len(physical)
                or {key(inst) for inst in projected} != set(physical)):
            raise ValueError('BP and physical X/Z DEM source labels do not align')
        result = copy(posterior)
        result.bp_stage1_probs_xz = posterior.probs_xz.copy()
        result.dem_xz = stim.DetectorErrorModel()
        for inst in posterior.dem_xz:
            if inst.type == 'error':
                # Never cap or otherwise transform the original physical prior.
                result.dem_xz.append('error', physical[key(inst)], inst.targets_copy())
            else:
                result.dem_xz.append(inst)
        result.H, result.obs_matrix, result.probs_xz = dem_to_parity_check(result.dem_xz)
        result.dems_decomposed = result._decompose_dems()
        for color, original in result.dems_decomposed.items():
            bp = posterior.dems_decomposed[color]
            if ((original.Hs[0] != bp.Hs[0]).nnz
                    or (original.error_map_matrices[0] != bp.error_map_matrices[0]).nnz):
                raise ValueError('Physical-prior rebuild changed BP stage-1 source ordering')
            original.probs = (bp.probs[0], original.probs[1])
            original._dems = (bp[0], original[1])
        return result
