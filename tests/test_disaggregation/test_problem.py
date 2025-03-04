import pytest

from disag_tools.disaggregation.problem import DisaggregationProblem


@pytest.fixture(scope="session")
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
def test_solution(default_problem, disaggregated_blocks, n):

    problem = default_problem.problems[n - 1]

    x_n = disaggregated_blocks.get_xn_vector(n)

    result1 = problem.m_matrix @ x_n
    result2 = problem.y_vector

    assert result1 == pytest.approx(result2, rel=5e-2)
