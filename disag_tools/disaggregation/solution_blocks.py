from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.disaggregation_blocks import DisaggregationBlocks, SectorInfo

SectorId: TypeAlias = str | tuple[str, str]


@dataclass
class SolutionBlocks:
    sectors: list[SectorInfo]
    reordered_matrix: pd.DataFrame
    output: pd.Series
    sector_mapping: dict[SectorId, list[SectorId]]
    aggregated_sectors_list: list[SectorId]

    @classmethod
    def from_disaggregation_blocks(
        cls, blocks: DisaggregationBlocks, disaggregation_dict: dict[SectorId, list[SectorId]]
    ) -> "SolutionBlocks":
        """Create a SolutionBlocks instance from DisaggregationBlocks and a disaggregation mapping.

        This method takes a DisaggregationBlocks instance where some sectors are marked for
        disaggregation, and a mapping that specifies how each sector should be split into
        subsectors. It creates a new matrix where the original sectors are replaced by their
        subsectors, with NaN values for the new entries.

        Args:
            blocks: DisaggregationBlocks instance containing the original matrix
            disaggregation_dict: Dictionary mapping each sector to its list of subsectors
                e.g. {("ROW", "A"): [("ROW", "A01"), ("ROW", "A03")]}

        Returns:
            SolutionBlocks instance with the expanded matrix and output series
        """
        # Make copies to avoid modifying originals
        matrix = blocks.reordered_matrix.copy()
        output = blocks.output.copy()

        # For each sector being disaggregated
        for sector_id in blocks.to_disagg_sector_names:
            # Get the subsectors for this sector
            subsectors = disaggregation_dict[sector_id]

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

            # Add output entries for subsectors
            for subsector in subsectors:
                output.loc[subsector] = np.nan

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

        return cls(
            sectors=sectors,
            reordered_matrix=matrix,
            output=output,
            sector_mapping=disaggregation_dict,
            aggregated_sectors_list=list(blocks.to_disagg_sector_names),
        )
