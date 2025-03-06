import pytest
import logging
import numpy as np

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
