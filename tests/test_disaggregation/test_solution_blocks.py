import numpy as np

from disag_tools.disaggregation.solution_blocks import SolutionBlocks


def test_solution_blocks_from_disaggregation_blocks(aggregated_blocks):
    """Test creating solution blocks from disaggregation blocks."""
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            disaggregation_dict[(country, sector)] = [(country, s) for s in subsectors]

    # Create solution blocks
    solution = SolutionBlocks.from_disaggregation_blocks(aggregated_blocks, disaggregation_dict)

    # Test that original sectors are removed
    assert "A" not in solution.reordered_matrix.index.get_level_values(0)
    assert "A" not in solution.reordered_matrix.columns.get_level_values(0)

    # Test that new subsectors are added for each country
    for country in countries:
        for subsector in ["A01", "A03"]:
            # Check matrix indices
            assert (country, subsector) in solution.reordered_matrix.index
            assert (country, subsector) in solution.reordered_matrix.columns
            # Check output series
            assert (country, subsector) in solution.output.index

    # Test that new entries are NaN
    for country in countries:
        for subsector1 in ["A01", "A03"]:
            # Check row entries
            assert solution.reordered_matrix.loc[(country, subsector1)].isna().all()
            # Check column entries
            assert solution.reordered_matrix.loc[:, (country, subsector1)].isna().all()
            # Check output
            assert np.isnan(solution.output.loc[(country, subsector1)])

    # Test that non-disaggregated sectors are preserved
    for idx in aggregated_blocks.non_disaggregated_sector_names:
        assert idx in solution.reordered_matrix.index
        assert idx in solution.reordered_matrix.columns
        assert idx in solution.output.index
        # Their values should be unchanged
        np.testing.assert_array_equal(
            solution.reordered_matrix.loc[idx, aggregated_blocks.non_disaggregated_sector_names],
            aggregated_blocks.reordered_matrix.loc[
                idx, aggregated_blocks.non_disaggregated_sector_names
            ],
        )

    # Test that sectors list is correctly constructed
    # Each country's "A" sector should be replaced by A01 and A03
    expected_sector_count = len(countries) * 2  # 2 subsectors per country
    assert len(solution.sectors) == expected_sector_count

    # Verify each sector info object
    for sector in solution.sectors:
        country, subsector = sector.sector_id
        assert country in countries
        assert subsector in ["A01", "A03"]
        assert sector.k == 1  # Each subsector has k=1 since it's already disaggregated
        assert sector.name == subsector  # Name should be the sector code
