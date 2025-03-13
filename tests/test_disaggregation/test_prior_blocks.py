"""Tests for the PriorBlocks class."""

import numpy as np
import pytest

from disag_tools.disaggregation.prior_blocks import PriorBlocks, PriorInfo, FinalDemandPriorInfo


def test_prior_blocks_from_disaggregation(disaggregated_blocks, aggregated_blocks):
    """Test that PriorBlocks correctly reconstructs X_n vectors from DisaggregationBlocks values."""
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

    # Extract all values from the disaggregated blocks into prior info
    prior_info: list[PriorInfo] = []
    matrix = disaggregated_blocks.reordered_matrix

    # Only include non-NaN values from the matrix
    for source in matrix.index:
        for dest in matrix.columns:
            value = matrix.loc[source, dest]
            if not np.isnan(value):
                prior_info.append((source, dest, value))

    # Create some final demand priors
    final_demand_prior: list[FinalDemandPriorInfo] = []

    # Create prior blocks with the extracted values
    prior_blocks = PriorBlocks.from_disaggregation_blocks(
        aggregated_blocks,
        disaggregation_dict,
        prior_info,
        final_demand_prior,
    )

    # For each sector, verify that the X_n vectors match
    for n in range(1, disaggregated_blocks.m + 1):
        # Get X_n from disaggregated blocks (without final demand)
        x_n = disaggregated_blocks.get_xn_vector(n)

        # Get prior_n from prior blocks
        prior_n = prior_blocks.get_prior_n_vector(n)

        # They should have the same shape plus the final demand components
        assert prior_n.shape[0] == x_n.shape[0]

        # non nan prior indices
        prior_indices = np.flatnonzero(~np.isnan(prior_n))

        assert np.allclose(prior_n[prior_indices], x_n[prior_indices])


def test_prior_blocks_no_info(aggregated_blocks):
    """Test that with no prior information, we get a vector of NaN values."""
    # Create disaggregation dictionary for all sectors being disaggregated
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

    # Create prior blocks with no prior information
    prior_blocks = PriorBlocks.from_disaggregation_blocks(
        aggregated_blocks,
        disaggregation_dict,
        prior_info=[],  # No prior info
        final_demand_prior=None,  # No final demand priors
    )

    # Get prior vector for the first sector
    prior_n = prior_blocks.get_prior_n_vector(1)

    # Check that all values are NaN
    assert np.all(np.isnan(prior_n))
