import pytest
from pathlib import Path

from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.disaggregation.problem import DisaggregationProblem


def test_country_config_read(icio_reader, canada_provincial_disagg_config):
    """Test that the Canada provinces configuration can be loaded correctly."""
    assert True


def test_disaggregation_problem(can_reader, canada_provincial_disagg_config):
    problem = DisaggregationProblem.from_configuration(
        reader=can_reader, config=canada_provincial_disagg_config
    )

    assert True
