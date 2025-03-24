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
        region_outputs: pd.Series | None = None,
    ) -> "FinalDemandBlocks":
        """Create a FinalDemandBlocks instance from disaggregation data.

        Args:
            final_demand_table: DataFrame containing final demand values for all sectors
            output: Series containing output values for all sectors
            disagg_mapping: Dictionary mapping aggregated sectors to their subsectors
            region_outputs: Optional Series containing the output of each region
        """

        is_regional = _check_regional(disagg_mapping)

        if is_regional and region_outputs is None:
            raise ValueError("region_outputs must be provided for regional disaggregation")

        if is_regional:
            return cls.regional_disagg(disagg_mapping, final_demand_table, output, region_outputs)
        return cls.sector_only_disagg(disagg_mapping, final_demand_table, output)

    @classmethod
    def sector_only_disagg(
        cls,
        disagg_mapping: dict[SectorId, list[SectorId]],
        final_demand_table: pd.DataFrame,
        output: pd.Series,
    ) -> "FinalDemandBlocks":
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

    @classmethod
    def regional_disagg(
        cls,
        disagg_mapping: dict[SectorId, list[SectorId]],
        final_demand_table: pd.DataFrame,
        output: pd.Series,
        region_outputs: pd.Series,
    ) -> "FinalDemandBlocks":
        # get the list of new regions
        new_regions = set()
        disaggregated_countries = set()
        for old_sector, subsectors in disagg_mapping.items():
            country = old_sector[0]
            disag_countries = set(subsector[0] for subsector in subsectors)
            new_regions.update(disag_countries - {country})
            # if we added new regions, we need to update the disaggregated_countries with the current country
            if (disag_countries - {country}) != set():
                disaggregated_countries.add(country)

        # Create the new index by replacing disaggregated countries with their regions
        new_index = []
        for idx in final_demand_table.index:
            country, sector = idx
            if country in disaggregated_countries:
                # For each region of this country, add a new index entry
                for region in new_regions:
                    if region.startswith(country):
                        new_index.append((region, sector))
            else:
                # Keep non-disaggregated countries as is
                new_index.append(idx)

        # Create the new columns by replacing disaggregated countries with their regions
        new_columns = []
        for col in final_demand_table.columns:
            country, dest = col
            if country in disaggregated_countries:
                # For each region of this country, add a new column entry
                for region in new_regions:
                    if region.startswith(country):
                        new_columns.append((region, dest))
            else:
                # Keep non-disaggregated countries as is
                new_columns.append(col)

        # Create the disaggregated final demand DataFrame with the new index and columns
        disaggregated_final_demand = (
            pd.DataFrame(
                np.nan,
                index=pd.MultiIndex.from_tuples(new_index),
                columns=pd.MultiIndex.from_tuples(new_columns),
            )
            .sort_index(axis=0, level=[0, 1])
            .sort_index(axis=1, level=[0, 1])
        )

        # Copy over values from original table for non-disaggregated countries
        for idx in final_demand_table.index:
            country, sector = idx
            if country not in disaggregated_countries:
                for col in final_demand_table.columns:
                    dest_country, dest = col
                    if dest_country not in disaggregated_countries:
                        disaggregated_final_demand.loc[idx, col] = final_demand_table.loc[idx, col]

        # Create one_step_ratio matrix by normalizing each row of the final demand table
        row_sums = final_demand_table.sum(axis=1)
        one_step_ratio = final_demand_table.div(row_sums, axis=0)
        # Replace any NaN values (from division by zero) with 0
        one_step_ratio = one_step_ratio.fillna(0)

        # Create ratios DataFrame with the same structure as disaggregated_final_demand
        ratios = pd.DataFrame(
            np.nan,
            index=disaggregated_final_demand.index,
            columns=disaggregated_final_demand.columns,
        )

        # Get the mapping of countries to their regions
        regions_dict = _get_country_regions(disagg_mapping)

        # Create inverse mapping from regions to their parent countries
        region_to_country = {}
        for country, regions in regions_dict.items():
            for region in regions:
                region_to_country[region] = country
        # Add non-disaggregated countries to the mapping
        for country in final_demand_table.index.get_level_values(0).unique():
            if country not in disaggregated_countries:
                region_to_country[country] = country

        # Calculate ratios for each row and column
        for origin_region, origin_sector in ratios.index:
            origin_country = region_to_country[origin_region]

            for dest_region, dest_fd in ratios.columns:
                dest_country = region_to_country[dest_region]

                # Get the original ratio
                orig_ratio = one_step_ratio.loc[
                    (origin_country, origin_sector), (dest_country, dest_fd)
                ]

                # Calculate output share (1 if not disaggregated)
                if dest_country in disaggregated_countries:
                    # Get region outputs for this country
                    country_regions = regions_dict[dest_country]
                    region_outputs_subset = region_outputs[country_regions]
                    output_share = region_outputs[dest_region] / region_outputs_subset.sum()
                else:
                    output_share = 1.0

                # Set the ratio
                ratios.loc[(origin_region, origin_sector), (dest_region, dest_fd)] = (
                    orig_ratio * output_share
                )

        # Create the sectors list with SectorInfo objects
        sectors = []
        for i, (region, sector) in enumerate(disaggregated_final_demand.index):
            name = str(sector)  # sector is already a string in this case
            sectors.append(SectorInfo(index=i + 1, sector_id=(region, sector), name=name, k=1))

        return cls(
            sectors=sectors,
            disaggregated_final_demand=disaggregated_final_demand,
            output=output,
            disagg_mapping=disagg_mapping,
            ratios=ratios,
            aggregated_sectors_list=sorted(list(disagg_mapping.keys())),
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


def _check_regional(disagg_mapping: dict[SectorId, list[SectorId]]) -> bool:
    is_regional = False
    for key, value in disagg_mapping.items():
        if isinstance(key, tuple):
            first_country = key[0]
            # Only check if any subsector has a different country code
            regionalised = any(
                isinstance(subsector, tuple) and subsector[0] != first_country
                for subsector in value
            )
            is_regional = is_regional or regionalised
    return is_regional


def _get_country_regions(disagg_mapping: dict[SectorId, list[SectorId]]) -> dict[str, list[str]]:
    """Extract a mapping from countries to their regions from the disaggregation mapping.

    Args:
        disagg_mapping: Dictionary mapping (country, sector) to list of (region, sector) tuples

    Returns:
        Dictionary mapping country codes to lists of region codes
    """
    country_regions: dict[str, set[str]] = {}

    for old_sector, subsectors in disagg_mapping.items():
        country = old_sector[0]
        if country not in country_regions:
            country_regions[country] = set()

        # Add all regions from subsectors
        for subsector in subsectors:
            if isinstance(subsector, tuple):
                region = subsector[0]
                if region != country:  # Only add if it's a different region
                    country_regions[country].add(region)

    # Convert sets to sorted lists for consistency
    return {country: sorted(list(regions)) for country, regions in country_regions.items()}
