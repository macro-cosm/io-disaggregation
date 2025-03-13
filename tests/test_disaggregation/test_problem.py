import logging

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

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


def test_no_prior_info(default_problem):
    """Test that prior information is None when not provided."""
    # Check that problem-level prior blocks are None
    assert default_problem.prior_blocks is None

    # Check that each sector's prior vector is None
    for problem in default_problem.problems:
        assert problem.prior_vector is None


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


def test_solve_with_partial_prior(real_disag_config, usa_aggregated_reader, disaggregated_blocks):
    """Test solving with partial prior information from the real solution."""
    # Use disaggregated_blocks as our real solution
    real_solution = disaggregated_blocks.reordered_matrix
    non_disagg_sectors = set(disaggregated_blocks.non_disagg_sector_names)

    # Create prior information DataFrame from real solution
    prior_data = []

    # For each sector pair in the solution
    for row_sector in real_solution.index:
        for col_sector in real_solution.columns:
            # Only include if at least one sector is disaggregated (not in non_disagg_sectors)
            if row_sector not in non_disagg_sectors or col_sector not in non_disagg_sectors:
                value = real_solution.loc[row_sector, col_sector]
                if isinstance(row_sector, tuple):
                    prior_data.append(
                        {
                            "Country_row": row_sector[0],
                            "Sector_row": row_sector[1],
                            "Country_column": col_sector[0],
                            "Sector_column": col_sector[1],
                            "value": value,
                        }
                    )
                else:
                    prior_data.append(
                        {"Sector_row": row_sector, "Sector_column": col_sector, "value": value}
                    )

    # Convert to DataFrame
    prior_df = pd.DataFrame(prior_data)

    # Randomly set 35% of values to NaN
    np.random.seed(42)  # For reproducibility
    mask = np.random.choice(
        [True, False], size=len(prior_df), p=[0.35, 0.65]  # 35% True (will be set to NaN)
    )
    prior_df.loc[mask, "value"] = np.nan

    # Drop rows with NaN values
    prior_df = prior_df.dropna()

    # Create and solve problem with prior information
    problem_with_prior = DisaggregationProblem.from_configuration(
        real_disag_config, usa_aggregated_reader, prior_df=prior_df
    )
    problem_with_prior.solve(lambda_sparse=1.0, mu_prior=10.0)

    # Get solution with prior
    solution_with_prior = problem_with_prior.solution_blocks.reordered_matrix

    # Calculate correlation between real solution and solution with prior
    # Flatten both matrices and compute correlation
    real_flat = real_solution.values.flatten()
    prior_flat = solution_with_prior.values.flatten()

    correlation = np.corrcoef(real_flat, prior_flat)[0, 1]
    assert correlation > 0.5, f"Correlation {correlation} is too low"

    # Also verify that where we had prior information, the values are close
    for _, row in prior_df.iterrows():
        if "Country_row" in row:
            row_idx = (row["Country_row"], row["Sector_row"])
            col_idx = (row["Country_column"], row["Sector_column"])
        else:
            row_idx = row["Sector_row"]
            col_idx = row["Sector_column"]

        assert_allclose(
            solution_with_prior.loc[row_idx, col_idx],
            row["value"],
            atol=1e-2,
            err_msg=f"Solution differs from prior at {row_idx}, {col_idx}",
        )
