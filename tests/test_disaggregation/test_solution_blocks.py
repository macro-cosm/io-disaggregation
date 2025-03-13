import numpy as np
import pytest

from disag_tools.disaggregation.solution_blocks import SolutionBlocks


# solution fixture
@pytest.fixture(scope="function")
def solution_blocks(aggregated_blocks, disaggregated_blocks):
    """Create a SolutionBlocks instance for testing.

    This fixture is recomputed for each test function to ensure test isolation.
    """
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(disaggregated_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = disaggregated_blocks.output[subsector_id] / total_output

    # Create solution blocks
    return SolutionBlocks.from_disaggregation_blocks(
        aggregated_blocks,
        disaggregation_dict,
        weight_dict,
    )


def test_solution_blocks_from_disaggregation_blocks(aggregated_blocks, disaggregated_blocks):
    """Test creating solution blocks from disaggregation blocks."""
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(disaggregated_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = disaggregated_blocks.output[subsector_id] / total_output

    # Create solution blocks
    solution = SolutionBlocks.from_disaggregation_blocks(
        aggregated_blocks,
        disaggregation_dict,
        weight_dict,
    )

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

    # Test that new matrix entries are NaN
    for country in countries:
        for subsector1 in ["A01", "A03"]:
            # Check row entries
            assert solution.reordered_matrix.loc[(country, subsector1)].isna().all()
            # Check column entries
            assert solution.reordered_matrix.loc[:, (country, subsector1)].isna().all()

    # Test that output values are correctly set using weights
    for country in countries:
        sector_id = (country, "A")
        original_output = aggregated_blocks.output[sector_id]
        for subsector in ["A01", "A03"]:
            subsector_id = (country, subsector)
            expected_output = original_output * weight_dict[subsector_id]
            assert np.allclose(solution.output[subsector_id], expected_output)
            # Also verify against disaggregated blocks
            assert np.allclose(
                solution.output[subsector_id], disaggregated_blocks.output[subsector_id]
            )

    # Test that non-disaggregated sectors are preserved
    for idx in aggregated_blocks.non_disagg_sector_names:
        assert idx in solution.reordered_matrix.index
        assert idx in solution.reordered_matrix.columns
        assert idx in solution.output.index
        # Their values should be unchanged
        np.testing.assert_array_equal(
            solution.reordered_matrix.loc[idx, aggregated_blocks.non_disagg_sector_names],
            aggregated_blocks.reordered_matrix.loc[idx, aggregated_blocks.non_disagg_sector_names],
        )
        assert np.allclose(solution.output[idx], aggregated_blocks.output[idx])

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


def test_solution_blocks_sector_mapping(aggregated_blocks, disaggregated_blocks):
    """Test that sector mapping and aggregated sectors list are correctly set."""
    # Create the disaggregation dictionary for all countries
    base_mapping = {"A": ["A01", "A03"]}
    countries = [
        idx[0] for idx in aggregated_blocks.reordered_matrix.index if isinstance(idx, tuple)
    ]
    countries = sorted(set(countries))

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(disaggregated_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = disaggregated_blocks.output[subsector_id] / total_output

    # Create solution blocks
    solution = SolutionBlocks.from_disaggregation_blocks(
        aggregated_blocks,
        disaggregation_dict,
        weight_dict,
    )

    # Test that sector_mapping is correctly set
    assert solution.sector_mapping == disaggregation_dict

    # Test that aggregated_sectors_list contains all the original sectors to be disaggregated
    assert set(solution.aggregated_sectors_list) == set(aggregated_blocks.to_disagg_sector_names)

    # Test that the order in aggregated_sectors_list matches the original order
    assert solution.aggregated_sectors_list == list(aggregated_blocks.to_disagg_sector_names)

    # Test that non_disaggregated_sector_names matches the original
    assert set(solution.non_disaggregated_sector_names) == set(
        aggregated_blocks.non_disagg_sector_names
    )
    assert solution.non_disaggregated_sector_names == list(
        aggregated_blocks.non_disagg_sector_names
    )

    # Verify the structure matches what we expect for each country
    for country in countries:
        # Check that each country-sector pair is in the mapping
        assert (country, "A") in solution.sector_mapping
        # Check that it maps to the correct subsectors
        assert solution.sector_mapping[(country, "A")] == [(country, "A01"), (country, "A03")]
        # Check that it's in the aggregated sectors list
        assert (country, "A") in solution.aggregated_sectors_list
        # Check that it's not in the non-disaggregated sectors list
        assert (country, "A") not in solution.non_disaggregated_sector_names


@pytest.mark.parametrize("n", [1, 2])
def test_solution_blocks_apply_e_vector(solution_blocks, disaggregated_blocks, n):
    """Test applying the e vector to a block."""
    # Generate the e vector
    e_vector = disaggregated_blocks.get_e_vector(n)

    # Apply the e vector to the block
    result = solution_blocks.apply_e_vector(n, e_vector)

    # check that product of block dimensions is equal to e_vector length
    assert result.shape[0] * result.shape[1] == e_vector.shape[0]

    # check that flattened result equals e_vector
    assert np.allclose(result.flatten(), e_vector)


@pytest.mark.parametrize("n", [1, 2])
def test_solution_blocks_apply_f_vector(solution_blocks, disaggregated_blocks, n):
    """Test applying the f vector to a block."""
    # Generate the f vector
    f_vector = disaggregated_blocks.get_f_vector(n)

    # Apply the f vector to the block
    result = solution_blocks.apply_f_vector(n, f_vector)

    # check that product of block dimensions is equal to f_vector length
    assert result.shape[0] * result.shape[1] == f_vector.shape[0]

    # check that flattened result (in column-major order) equals f_vector
    assert np.allclose(result.flatten(order="F"), f_vector)


@pytest.mark.parametrize("n,l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_solution_blocks_apply_g_vector(solution_blocks, disaggregated_blocks, n, l):
    """Test applying the g vector to a block.

    Tests both same-sector (n=l) and different-sector (n≠l) cases.
    """
    # Generate the g vector
    g_vector = disaggregated_blocks.get_gnl_vector(n, l)

    # Apply the g vector to the block
    result = solution_blocks.apply_gnl_vector(n, l, g_vector)

    # check that product of block dimensions is equal to g_vector length
    assert result.shape[0] * result.shape[1] == g_vector.shape[0]

    # check that flattened result (in column-major order) equals g_vector
    assert np.allclose(result.flatten(), g_vector)

    # Additional checks for specific cases
    if n == l:
        # For same sector blocks, verify it's a square matrix
        assert result.shape[0] == result.shape[1]
        # The shape should match the number of subsectors for that sector
        sector = solution_blocks.aggregated_sectors_list[n - 1]
        assert result.shape[0] == len(solution_blocks.sector_mapping[sector])


def test_entire_solution(disaggregated_blocks, solution_blocks):
    """Test applying all vectors to the solution blocks."""
    for n in range(1, disaggregated_blocks.m + 1):
        # Generate the vectors
        x_n = disaggregated_blocks.get_xn_vector(n)
        solution_blocks.apply_xn(n, x_n)

    # Check that the output matches the original
    assert np.allclose(
        solution_blocks.reordered_matrix.values, disaggregated_blocks.reordered_matrix.values
    )
    assert np.allclose(solution_blocks.output.values, disaggregated_blocks.output.values)
