"""Module for handling prior information in disaggregation problems."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.disaggregation_blocks import DisaggregationBlocks, SectorId

PriorInfo = tuple[SectorId, SectorId, float]
FinalDemandPriorInfo = tuple[SectorId, float]


@dataclass
class PriorBlocks:
    """Class to hold prior information about the solution.

    This class maintains a matrix of prior information where:
    - 0 indicates elements that should be sparse
    - Positive values indicate known values
    - NaN indicates no prior information

    The structure allows extracting these elements in X_n vectors that match
    the structure used in the optimization problem.

    Attributes:
        reordered_matrix: DataFrame with same structure as solution matrix
        final_demand_priors: Dictionary mapping subsectors to their final demand priors
        sector_mapping: Dictionary mapping original sectors to subsectors
        aggregated_sectors_list: List of sectors being disaggregated
        non_disaggregated_sector_names: List of sectors not being disaggregated
    """

    reordered_matrix: pd.DataFrame
    final_demand_priors: dict[SectorId, float]
    sector_mapping: dict[SectorId, list[SectorId]]
    aggregated_sectors_list: list[SectorId]
    non_disaggregated_sector_names: list[SectorId]

    @classmethod
    def from_disaggregation_blocks(
        cls,
        blocks: DisaggregationBlocks,
        disaggregation_dict: dict[SectorId, list[SectorId]],
        prior_info: list[PriorInfo],
        final_demand_prior: list[FinalDemandPriorInfo] | None = None,
    ) -> "PriorBlocks":
        """Create a PriorBlocks instance with prior information about the solution.

        Args:
            blocks: DisaggregationBlocks instance containing the original matrix
            disaggregation_dict: Dictionary mapping each sector to its list of subsectors
            prior_info: List of (source_sector, dest_sector, value) tuples where:
                - value = 0 indicates the element should be sparse
                - value > 0 indicates a known value
                - pairs not in the list have no prior information (nan)
            final_demand_prior: Optional list of (sector, value) tuples for final demand
                where value is the known/sparse final demand for that sector

        Returns:
            PriorBlocks instance with prior information
        """
        # Start with empty matrix
        matrix = blocks.reordered_matrix.copy()

        # For each sector being disaggregated
        for sector_id in blocks.to_disagg_sector_names:
            # Get the subsectors for this sector
            subsectors = disaggregation_dict[sector_id]

            # Remove the original sector's row and column
            matrix = matrix.drop(sector_id, axis=0)
            matrix = matrix.drop(sector_id, axis=1)

            # Add new rows and columns for subsectors
            all_rows = list(matrix.index) + subsectors
            matrix = matrix.reindex(index=all_rows, fill_value=np.nan)
            all_cols = list(matrix.columns) + subsectors
            matrix = matrix.reindex(columns=all_cols, fill_value=np.nan)

        # Fill in prior information
        for source, dest, value in prior_info:
            matrix.loc[source, dest] = value

        # Create final demand priors dictionary
        final_demand_priors = {}
        if final_demand_prior is not None:
            for sector, value in final_demand_prior:
                final_demand_priors[sector] = value

        return cls(
            reordered_matrix=matrix,
            final_demand_priors=final_demand_priors,
            sector_mapping=disaggregation_dict,
            aggregated_sectors_list=list(blocks.to_disagg_sector_names),
            non_disaggregated_sector_names=list(blocks.non_disagg_sector_names),
        )

    def get_prior_n_vector(self, n: int) -> np.ndarray:
        """Get the prior information vector for sector n.

        This returns a vector matching the structure of X_n where:
        - 0 indicates elements that should be sparse
        - Positive values indicate known values
        - NaN indicates no prior information

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing prior information in the same structure as X_n = [E, F, G, B]
        """
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        # Get E block (flows from undisaggregated to disaggregated)
        e_block = self.reordered_matrix.loc[
            self.non_disaggregated_sector_names, disaggregated_sectors
        ].values
        e_vector = e_block.flatten()

        # Get F block (flows from disaggregated to undisaggregated)
        f_block = self.reordered_matrix.loc[
            disaggregated_sectors, self.non_disaggregated_sector_names
        ].values
        f_vector = f_block.flatten(order="F")

        # Get G blocks (flows between disaggregated sectors)
        g_vectors = []
        for l in range(1, len(self.aggregated_sectors_list) + 1):
            dest_sector = self.aggregated_sectors_list[l - 1]
            dest_subsectors = self.sector_mapping[dest_sector]
            g_block = self.reordered_matrix.loc[disaggregated_sectors, dest_subsectors].values
            g_vectors.append(g_block.flatten())

        # Get B vector (final demand for subsectors)
        b_vector = np.full(len(disaggregated_sectors), np.nan)
        for i, sector in enumerate(disaggregated_sectors):
            if sector in self.final_demand_priors:
                b_vector[i] = self.final_demand_priors[sector]

        # Concatenate all vectors in the order [E, F, G, B]
        return np.concatenate([e_vector, f_vector] + g_vectors + [b_vector])
