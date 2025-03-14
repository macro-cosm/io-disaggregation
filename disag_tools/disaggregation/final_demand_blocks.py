from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.disaggregation_blocks import SectorInfo

SectorId: TypeAlias = str | tuple[str, str]


@dataclass
class FinalDemandBlocks:
    sectors: list[SectorInfo]
    disaggregated_final_demand: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    output: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    disagg_mapping: dict[SectorId, list[SectorId]] = field(default_factory=dict)
    ratios: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    aggregated_sectors_list: list[SectorId] = field(default_factory=list)

    @classmethod
    def from_disaggregation_blocks(
        cls,
        final_demand_table: pd.DataFrame,
        output: pd.Series,
        disagg_mapping: dict[SectorId, list[SectorId]],
    ) -> "FinalDemandBlocks":
        """Create a FinalDemandBlocks instance from disaggregation data.

        Args:
            final_demand_table: DataFrame containing final demand values for all sectors
            output: Series containing output values for all sectors
            disagg_mapping: Dictionary mapping aggregated sectors to their subsectors
        """
        # Create ordered list of aggregated sectors
        aggregated_sectors_list = sorted(list(disagg_mapping.keys()))

        # Start with empty matrix
        disaggregated_final_demand = final_demand_table.copy()

        # Get list of all disaggregated sectors
        all_disagg_sectors = []
        for subsectors in disagg_mapping.values():
            all_disagg_sectors.extend(subsectors)

        # For each sector being disaggregated
        for sector_id in aggregated_sectors_list:
            # Get the subsectors for this sector
            subsectors = disagg_mapping[sector_id]

            # Check if sector exists in the index
            if sector_id not in disaggregated_final_demand.index:
                continue

            # Remove the original sector's row
            disaggregated_final_demand = disaggregated_final_demand.drop(sector_id)

            # Add new rows for subsectors
            all_rows = list(disaggregated_final_demand.index) + subsectors
            disaggregated_final_demand = disaggregated_final_demand.reindex(
                index=all_rows, fill_value=np.nan
            )

        # Create sectors list with SectorInfo objects
        sectors = []
        # Only create SectorInfo objects for the sectors being disaggregated
        for i, sector_id in enumerate(aggregated_sectors_list):
            # For each subsector, create a SectorInfo with k=1 since it's already disaggregated
            subsectors = disagg_mapping[sector_id]
            for subsector in subsectors:
                name = str(subsector[1]) if isinstance(subsector, tuple) else str(subsector)
                sectors.append(
                    SectorInfo(index=len(sectors) + 1, sector_id=subsector, name=name, k=1)
                )

        # Initialize ratios DataFrame with the same index type as final_demand_table
        if isinstance(final_demand_table.index[0], tuple):
            # For MultiIndex, ensure all_disagg_sectors contains tuples
            index = pd.MultiIndex.from_tuples(all_disagg_sectors)
        else:
            index = pd.Index(all_disagg_sectors)

        ratios = pd.DataFrame(np.nan, index=index, columns=final_demand_table.columns)

        # For each aggregated sector, compute ratios for its subsectors
        for sector_id in aggregated_sectors_list:
            subsectors = disagg_mapping[sector_id]
            # Get the aggregated sector's final demand values
            if sector_id not in final_demand_table.index:
                continue

            # Get total final demand for this aggregated sector
            total_fd = final_demand_table.loc[sector_id].sum()
            if total_fd != 0:  # Avoid division by zero
                # For each subsector, set ratios based on aggregated sector's distribution
                for subsector in subsectors:
                    ratios.loc[subsector] = final_demand_table.loc[sector_id] / total_fd

        return cls(
            sectors=sectors,
            disaggregated_final_demand=disaggregated_final_demand,
            output=output,
            disagg_mapping=disagg_mapping,
            ratios=ratios,
            aggregated_sectors_list=aggregated_sectors_list,
        )

    def apply_bn_vector(self, n: int, bn_vector: np.ndarray) -> np.ndarray:
        """Apply the bn vector to the appropriate block in the disaggregated final demand.

        Args:
            n: Index of the sector (1-based as in math notation)
            bn_vector: Vector containing final demand values for subsectors of sector n

        Returns:
            The reshaped block that was applied to the disaggregated final demand
        """
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.disagg_mapping[aggregated_sector]

        # Check that the length of bn_vector matches the number of disaggregated sectors
        assert len(bn_vector) == len(disaggregated_sectors), (
            f"bn_vector length ({len(bn_vector)}) does not match "
            f"number of disaggregated sectors ({len(disaggregated_sectors)})"
        )

        # Multiply bn_vector by the aggregate output of the sector
        if aggregated_sector not in self.output.index:
            raise KeyError(f"Sector {aggregated_sector} not found in output series")

        aggregate_output = self.output.loc[aggregated_sector]
        bn_vector *= aggregate_output

        # For each subsector i and final demand category j
        # D_ij = b_i * r_j where r_j is the ratio for that category
        for i, subsector in enumerate(disaggregated_sectors):
            for col in self.disaggregated_final_demand.columns:
                ratio = self.ratios.loc[subsector, col]
                if not np.isnan(ratio):  # Only apply if we have a valid ratio
                    self.disaggregated_final_demand.loc[subsector, col] = bn_vector[i] * ratio

        return bn_vector
