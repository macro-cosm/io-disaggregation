"""Tests for the FinalDemandBlocks class."""

import numpy as np
import pytest

from disag_tools.disaggregation.final_demand_blocks import FinalDemandBlocks


@pytest.fixture
def final_demand_blocks(disaggregated_blocks, aggregated_blocks, usa_aggregated_reader):
    """Create a FinalDemandBlocks instance for testing.

    This creates a FinalDemandBlocks instance with the mapping from sector A to A01 and A03
    for all countries in the dataset.
    """
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(disaggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disagg_mapping = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disagg_mapping[sector_id] = subsector_ids

    # Create and return FinalDemandBlocks instance
    return FinalDemandBlocks.from_disaggregation_blocks(
        final_demand_table=usa_aggregated_reader.final_demand_table,
        output=aggregated_blocks.output,
        disagg_mapping=disagg_mapping,
    )


def test_final_demand_values(disaggregated_blocks, usa_reader):
    """Test that final demand values match between disaggregated blocks and ICIO reader.

    The bn_vector from disaggregated_blocks is rescaled by output, so we need to
    multiply it by the aggregate output to get the actual final demand values.
    """
    # Get bn_vector for sector 1 (first disaggregated sector)
    bn_vector = disaggregated_blocks.get_bn_vector(1)

    # Get the aggregated sector for n=1
    aggregated_sector = disaggregated_blocks.aggregated_sectors_list[0]
    disaggregated_sectors = disaggregated_blocks.sector_mapping[aggregated_sector]

    # Get the aggregate output for this sector
    aggregate_output = sum(disaggregated_blocks.output[s] for s in disaggregated_sectors)

    # Multiply bn_vector by aggregate output to get actual final demand
    actual_final_demand = bn_vector * aggregate_output

    # Get expected final demand from ICIO reader for these sectors
    expected_final_demand = usa_reader.final_demand.loc[disaggregated_sectors].values

    # Compare values
    assert np.allclose(
        actual_final_demand, expected_final_demand, rtol=1e-2
    ), "Final demand values do not match between disaggregated blocks and ICIO reader"


def test_ratios_sum_to_one(final_demand_blocks):
    """Test that ratios for each set of disaggregated sectors sum to 1 across columns."""
    # For each aggregated sector
    sums = final_demand_blocks.ratios.sum(axis=1).values
    assert np.allclose(sums, 1), "Ratios do not sum to 1 for each set of disaggregated sectors"


def test_apply_all_bn_vectors(final_demand_blocks, disaggregated_blocks, usa_reader):
    """Test applying all bn vectors to FinalDemandBlocks and verifying total demand.

    We check that the total demand (sum over all final demand columns) for each
    disaggregated sector matches the total demand from the original ICIO reader.
    """
    # Apply bn vectors for all sectors
    for n in range(1, disaggregated_blocks.m + 1):
        bn_vector = disaggregated_blocks.get_bn_vector(n)
        final_demand_blocks.apply_bn_vector(n, bn_vector)

    # Get the complete final demand table and sum over columns for total demand
    result = final_demand_blocks.disaggregated_final_demand.sum(axis=1)

    # Get expected values from ICIO reader (sum over columns)
    expected = usa_reader.final_demand_table.loc[disaggregated_blocks.to_disagg_sector_names].sum(
        axis=1
    )

    # Compare total demand values
    assert np.allclose(
        result[expected.index], expected, rtol=1e-2
    ), "Total final demand does not match expected values after applying all bn vectors"
