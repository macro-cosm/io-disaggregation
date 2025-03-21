"""Module for generating planted solutions for testing disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import Optional, TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregationBlocks,
    SectorId,
    SectorInfo,
)
from disag_tools.disaggregation.solution_blocks import SolutionBlocks

logger = logging.getLogger(__name__)

# Type aliases for clarity
Array: TypeAlias = np.ndarray


@dataclass
class PlantedSolution:
    """Class for generating planted solutions for testing disaggregation problems.

    This class provides methods to generate a known solution that satisfies the disaggregation
    equations, which can be used for testing the solver's correctness.
    """

    blocks: DisaggregationBlocks
    disaggregation_dict: dict[SectorId, list[SectorId]]
    weight_dict: dict[SectorId, float]
    output: pd.Series
    reordered_matrix: pd.DataFrame
    b_n_vectors: dict[int, Array]

    @classmethod
    def from_disaggregation_blocks(
        cls,
        blocks: DisaggregationBlocks,
        disaggregation_dict: dict[SectorId, list[SectorId]],
        weight_dict: dict[SectorId, float],
    ) -> "PlantedSolution":
        """Create a planted solution from disaggregation blocks.

        Args:
            blocks: Original disaggregation blocks
            disaggregation_dict: Dictionary mapping sectors to their subsectors
            weight_dict: Dictionary mapping subsectors to their weights

        Returns:
            PlantedSolution instance with the solution matrix and vectors
        """
        # Make copies to avoid modifying originals
        matrix = blocks.reordered_matrix.copy()
        output = blocks.output.copy()

        # Store original outputs for later use
        old_outputs = {sector: output.loc[sector] for sector in blocks.to_disagg_sector_names}

        # For each sector being disaggregated
        for sector_id in blocks.to_disagg_sector_names:
            # Get the subsectors for this sector
            subsectors = disaggregation_dict[sector_id]

            # Get the original sector's output before removing it
            original_output = output[sector_id]

            # Remove the original sector's row and column
            matrix = matrix.drop(sector_id, axis=0)
            matrix = matrix.drop(sector_id, axis=1)
            output = output.drop(sector_id)

            # Add new rows and columns for subsectors
            # First add rows
            all_rows = list(matrix.index) + subsectors
            matrix = matrix.reindex(index=all_rows, fill_value=np.nan)
            # Then add columns
            all_cols = list(matrix.columns) + subsectors
            matrix = matrix.reindex(columns=all_cols, fill_value=np.nan)

            # Add output entries for subsectors using weights
            for subsector in subsectors:
                output.loc[subsector] = original_output * weight_dict.get(subsector, 0)

        # Create SectorInfo objects for the new sectors
        sectors = []
        for i, sector_id in enumerate(blocks.to_disagg_sector_names):
            subsectors = disaggregation_dict[sector_id]
            # For each subsector, create a SectorInfo with k=1 since it's already disaggregated
            for subsector in subsectors:
                name = str(subsector[1]) if isinstance(subsector, tuple) else str(subsector)
                sectors.append(
                    SectorInfo(index=len(sectors) + 1, sector_id=subsector, name=name, k=1)
                )

        # Fill in the E, F, and G blocks
        aggregated_sectors_list = list(blocks.to_disagg_sector_names)
        non_disagg_sector_names = list(blocks.non_disagg_sector_names)

        # Apply E block
        for n in range(1, len(aggregated_sectors_list) + 1):
            aggregated_sector = aggregated_sectors_list[n - 1]
            b_vector = blocks.get_B(n)
            disaggregated_sectors = disaggregation_dict[aggregated_sector]
            k_n = len(disaggregated_sectors)
            weights = np.array([weight_dict.get(sector, 0) for sector in disaggregated_sectors])
            dims = (len(non_disagg_sector_names), k_n)
            e_block = np.zeros(dims)
            for i in range(dims[0]):
                for j in range(dims[1]):
                    e_block[i, j] = b_vector[i] / (k_n * weights[j])
            matrix.loc[non_disagg_sector_names, disaggregated_sectors] = e_block

        # Apply F block
        for n in range(1, len(aggregated_sectors_list) + 1):
            aggregated_sector = aggregated_sectors_list[n - 1]
            disaggregated_sectors = disaggregation_dict[aggregated_sector]
            c_n = blocks.get_C(n)
            k_n = len(disaggregated_sectors)
            dims = (k_n, len(non_disagg_sector_names))
            f_block = np.zeros(dims)
            for i in range(dims[0]):
                for j in range(dims[1]):
                    f_block[i, j] = c_n[j] / k_n
            matrix.loc[disaggregated_sectors, non_disagg_sector_names] = f_block

        # Apply G blocks
        for n in range(1, len(aggregated_sectors_list) + 1):
            for l in range(1, len(aggregated_sectors_list) + 1):
                aggregated_sector_n = aggregated_sectors_list[n - 1]
                aggregated_sector_l = aggregated_sectors_list[l - 1]
                source_subsectors = disaggregation_dict[aggregated_sector_n]
                target_subsectors = disaggregation_dict[aggregated_sector_l]
                weights = np.array([weight_dict.get(sector, 0) for sector in target_subsectors])
                k_n = len(source_subsectors)
                k_l = len(target_subsectors)
                g_block = np.zeros((k_n, k_l))
                d_nl = blocks.get_D_nl(n, l)
                for i in range(k_n):
                    for j in range(k_l):
                        g_block[i, j] = d_nl / (k_n * k_l * weights[j])
                matrix.loc[source_subsectors, target_subsectors] = g_block

        # get all relative weights for all sectors
        relative_weights = []
        for l in range(1, len(aggregated_sectors_list) + 1):
            aggregated_sector_l = aggregated_sectors_list[l - 1]
            weights = np.array(
                [weight_dict.get(sector, 0) for sector in disaggregation_dict[aggregated_sector_l]]
            )
            relative_weights.append(weights)

        b_n_vectors = {}

        for n in range(1, len(aggregated_sectors_list) + 1):
            M4 = blocks.get_m4_block(n, relative_weights)
            M5 = blocks.get_m5_block(n)

            aggregated_sector = aggregated_sectors_list[n - 1]
            disaggregated_sectors = disaggregation_dict[aggregated_sector]
            f_block = matrix.loc[disaggregated_sectors, non_disagg_sector_names].values
            f_vector = f_block.flatten(order="F")

            g_vectors = []
            for l in range(1, len(aggregated_sectors_list) + 1):
                g_block = matrix.loc[
                    disaggregated_sectors, disaggregation_dict[aggregated_sectors_list[l - 1]]
                ].values
                g_vectors.append(g_block.flatten())

            g_vector = np.concatenate(g_vectors)

            w = np.array([weight_dict.get(sector, 0) for sector in disaggregated_sectors])

            # b_n theoretical
            b_n = w - M5 @ f_vector - M4 @ g_vector

            b_n_vectors[n] = b_n

        return cls(
            blocks=blocks,
            disaggregation_dict=disaggregation_dict,
            weight_dict=weight_dict,
            output=output,
            reordered_matrix=matrix,
            b_n_vectors=b_n_vectors,
        )

    def get_xn_vector(self, n: int) -> Array:
        """Get the x_n vector for sector n.

        Args:
            n: Sector index (1-based)

        Returns:
            Array containing the x_n vector
        """
        aggregated_sector = self.blocks.to_disagg_sector_names[n - 1]
        disaggregated_sectors = self.disaggregation_dict[aggregated_sector]
        k_n = len(disaggregated_sectors)

        # Get E block
        e_block = self.get_e_block(n)
        e_vector = e_block.flatten()

        # Get F block
        f_block = self.get_f_block(n)
        f_vector = f_block.flatten(order="F")

        # Get G blocks
        g_vectors = []
        for l in range(1, len(self.blocks.to_disagg_sector_names) + 1):
            g_block = self.get_gnl_block(n, l)
            g_vectors.append(g_block.flatten())

        # Get b_n vector
        b_n = self.get_bn_vector(n)

        # Combine all vectors
        return np.concatenate([e_vector, f_vector] + g_vectors + [b_n])

    def get_e_block(self, n: int) -> Array:
        """Get the E block for sector n."""
        aggregated_sector = self.blocks.to_disagg_sector_names[n - 1]
        disaggregated_sectors = self.disaggregation_dict[aggregated_sector]
        return self.reordered_matrix.loc[
            self.blocks.non_disagg_sector_names, disaggregated_sectors
        ].values

    def get_e_vector(self, n: int) -> Array:
        """Get the E vector for sector n (technical coefficients from undisaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from all undisaggregated sectors
            to the subsectors of sector n
        """
        e_block = self.get_e_block(n)
        return e_block.flatten()

    def get_f_block(self, n: int) -> Array:
        """Get the F block for sector n."""
        aggregated_sector = self.blocks.to_disagg_sector_names[n - 1]
        disaggregated_sectors = self.disaggregation_dict[aggregated_sector]
        return self.reordered_matrix.loc[
            disaggregated_sectors, self.blocks.non_disagg_sector_names
        ].values

    def get_f_vector(self, n: int) -> Array:
        """Get the F vector for sector n (technical coefficients to undisaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from the subsectors of sector n
            to all undisaggregated sectors
        """
        f_block = self.get_f_block(n)
        return f_block.flatten(order="F")

    def get_gnl_block(self, n: int, l: int) -> Array:
        """Get the G block for sectors n and l."""
        aggregated_sector_n = self.blocks.to_disagg_sector_names[n - 1]
        aggregated_sector_l = self.blocks.to_disagg_sector_names[l - 1]
        source_subsectors = self.disaggregation_dict[aggregated_sector_n]
        target_subsectors = self.disaggregation_dict[aggregated_sector_l]
        return self.reordered_matrix.loc[source_subsectors, target_subsectors].values

    def get_gnl_vector(self, n: int, l: int) -> Array:
        """Get the G vector for sectors n and l (technical coefficients between their subsectors).

        Args:
            n: Index of the first sector (1-based as in math notation)
            l: Index of the second sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from subsectors of sector n to
            subsectors of sector l, flattened in row-major order
        """
        g_block = self.get_gnl_block(n, l)
        return g_block.flatten()

    def get_gn_vector(self, n: int) -> Array:
        """Get the G vector for sector n (technical coefficients between subsectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from the subsectors of sector n
            to all other subsectors
        """
        gnls = [
            self.get_gnl_vector(n, l) for l in range(1, len(self.blocks.to_disagg_sector_names) + 1)
        ]
        return np.concatenate(gnls)

    def get_bn_vector(self, n: int) -> Array:
        """Get the b_n vector for sector n."""
        return self.b_n_vectors[n]
