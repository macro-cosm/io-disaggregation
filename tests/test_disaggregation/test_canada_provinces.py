import logging

import numpy as np
import pytest

from disag_tools.assembler import AssembledData
from disag_tools.disaggregation.problem import DisaggregationProblem
from disag_tools.readers import ICIOReader

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_disaggregation_problem(
    can_reader,
    canada_provincial_disagg_config,
    canada_technical_coeffs_prior,
    tmp_path_factory,
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

    # # Keep only top 50% of values
    # canada_technical_coeffs_prior = canada_technical_coeffs_prior[
    #     canada_technical_coeffs_prior["value"] >= threshold
    # ].copy()

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

    fd_cols = [
        col
        for col in assembled.data.columns
        if any(col[1].startswith(p) for p in can_reader.FINAL_DEMAND_PREFIXES)
    ]

    # check that all provinces have the final demand columns
    for code in province_codes:
        assert any(
            code == col[0] for col in fd_cols
        ), f"Province {code} does not have final demand columns"

    for code in province_codes:
        assert code in assembled.data.index.get_level_values("CountryInd").unique()
        assert code in assembled.data.columns.get_level_values("CountryInd").unique()

    temp_dir = tmp_path_factory.mktemp("data")

    assembled.data.to_csv(temp_dir / "canada_provinces.csv")

    reader = ICIOReader.from_csv(temp_dir / "canada_provinces.csv")

    assert np.allclose(reader.output_from_out, reader.output_from_sums, rtol=1e-5)

    reaggregated_intermediate_table = (
        reader.intermediate_demand_table.loc[province_codes, province_codes]
        .groupby(level=1)
        .sum()
        .groupby(level=1, axis=1)
        .sum()
    )

    reagg_ratio = (
        reaggregated_intermediate_table / can_reader.intermediate_demand_table.loc["CAN", "CAN"]
    ).unstack()

    # drop nans and infs from the ratio
    reagg_ratio = reagg_ratio.replace([np.inf, -np.inf], np.nan).dropna()

    assert reagg_ratio.median() == pytest.approx(1.0, rel=1e-5)

    assert True
