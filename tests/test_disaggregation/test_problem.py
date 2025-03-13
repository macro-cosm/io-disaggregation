import logging

import numpy as np
import pytest

from disag_tools.disaggregation.problem import DisaggregationProblem

logger = logging.getLogger(__name__)


@pytest.fixture()
def default_problem(real_disag_config, usa_aggregated_reader):
    return DisaggregationProblem.from_configuration(
        config=real_disag_config, reader=usa_aggregated_reader
    )


def test_sizes(default_problem):
    assert len(default_problem.problems) == 2
    assert len(default_problem.weights) == 2
    assert len(default_problem.weights[0]) == 2
    assert default_problem.disaggregation_blocks is not None


@pytest.mark.parametrize("n", [1, 2])
def test_problem_compatibility(default_problem, aggregated_blocks, disaggregated_blocks, n):
    relative_output_weights_list = [
        disaggregated_blocks.get_relative_output_weights(l + 1)
        for l in range(disaggregated_blocks.m)
    ]
    large_m = aggregated_blocks.get_large_m(n, relative_output_weights_list)

    large_m_problem = default_problem.problems[n - 1].m_matrix

    assert np.allclose(large_m, large_m_problem)


@pytest.mark.parametrize("n", [1, 2])
def test_problem_solution(default_problem, n, disaggregated_blocks):
    problem = default_problem.problems[n - 1]
    m_matrix = problem.m_matrix
    y_vector = problem.y_vector

    x = disaggregated_blocks.get_xn_vector(n)

    # check that m @ x = y
    assert np.allclose(m_matrix @ x, y_vector, rtol=2e-2)


def test_solution_blocks_structure(default_problem):
    """Test that the solution_blocks attribute is properly created and structured."""
    solution = default_problem.solution_blocks
    blocks = default_problem.disaggregation_blocks

    # Test that original sectors are removed
    for sector_id in blocks.to_disagg_sector_names:
        assert sector_id not in solution.reordered_matrix.index
        assert sector_id not in solution.reordered_matrix.columns
        assert sector_id not in solution.output.index

    # Test that new subsectors are added
    for sector_id in blocks.to_disagg_sector_names:
        subsectors = solution.sector_mapping[sector_id]
        for subsector in subsectors:
            # Check matrix indices
            assert subsector in solution.reordered_matrix.index
            assert subsector in solution.reordered_matrix.columns
            # Check output series
            assert subsector in solution.output.index
            # Check that new entries are NaN
            assert solution.reordered_matrix.loc[subsector].isna().all()
            assert solution.reordered_matrix.loc[:, subsector].isna().all()
            assert np.isnan(solution.output.loc[subsector])

    # Test that non-disaggregated sectors are preserved with unchanged values
    for idx in blocks.non_disagg_sector_names:
        assert idx in solution.reordered_matrix.index
        assert idx in solution.reordered_matrix.columns
        assert idx in solution.output.index
        np.testing.assert_array_equal(
            solution.reordered_matrix.loc[idx, blocks.non_disagg_sector_names],
            blocks.reordered_matrix.loc[idx, blocks.non_disagg_sector_names],
        )

    # Test sector mapping and lists
    assert solution.aggregated_sectors_list == list(blocks.to_disagg_sector_names)
    assert solution.non_disaggregated_sector_names == list(blocks.non_disagg_sector_names)


def test_solution_blocks_apply_solution(default_problem, disaggregated_blocks):
    """Test that we can apply the solution to the solution blocks."""
    solution = default_problem.solution_blocks

    # Apply solution for each sector
    for n in range(1, disaggregated_blocks.m + 1):
        x_n = disaggregated_blocks.get_xn_vector(n)
        solution.apply_xn(n, x_n)

    # Check that the result matches the original disaggregated matrix
    assert np.allclose(
        solution.reordered_matrix.values, disaggregated_blocks.reordered_matrix.values
    )
