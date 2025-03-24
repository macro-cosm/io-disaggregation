import logging
import pytest
from pathlib import Path

from disag_tools.assembler import AssembledData
from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.disaggregation.problem import DisaggregationProblem

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_disaggregation_problem(
    can_reader,
    canada_provincial_disagg_config,
    canada_technical_coeffs_prior,
):
    """Test disaggregation problem with prior information.

    We keep only the top 50% of values in the prior to simplify the problem.
    """
    # Get threshold for top 50% of values
    threshold = canada_technical_coeffs_prior["value"].quantile(0.50)
    logger.debug(f"Using threshold {threshold:.2e} for prior values")
    logger.debug(
        f"Prior value range: [{canada_technical_coeffs_prior['value'].min():.2e}, {canada_technical_coeffs_prior['value'].max():.2e}]"
    )

    # Keep only top 50% of values
    canada_technical_coeffs_prior = canada_technical_coeffs_prior[
        canada_technical_coeffs_prior["value"] >= threshold
    ].copy()

    logger.debug(f"Using {len(canada_technical_coeffs_prior)} prior constraints")

    problem = DisaggregationProblem.from_configuration(
        reader=can_reader,
        config=canada_provincial_disagg_config,
        technical_coeffs_prior_df=canada_technical_coeffs_prior,
    )

    problem.solve()

    assembled = AssembledData.from_solution(problem, can_reader)

    province_codes = [
        "CAN_AB",
        "CAN_BC",
        "CAN_MB",
        "CAN_NB",
        "CAN_NL",
        "CAN_NS",
        "CAN_ON",
        "CAN_PE",
        "CAN_QC",
        "CAN_SK",
    ]

    for code in province_codes:
        assert code in assembled.data.index.get_level_values("CountryInd").unique()
        assert code in assembled.data.columns.get_level_values("CountryInd").unique()
