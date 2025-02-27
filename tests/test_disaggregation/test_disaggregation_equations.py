"""Tests for the disaggregation equations.

This module tests the individual equations that make up the disaggregation
constraints. Each equation is tested separately to ensure it holds for
known data.
"""

import numpy as np
import pytest

from disag_tools.disaggregation.constraints import (
    generate_M1_block,
    generate_M2_block,
    generate_M3_block,
    generate_M4_block,
    generate_M5_block,
)
from disag_tools.readers.disaggregation_blocks import (
    DisaggregationBlocks,
    DisaggregatedBlocks,
    unfold_countries,
)
from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.vector_blocks import (
    flatten_F_block,
    flatten_G_block,
)


@pytest.fixture(scope="session")
def aggregated_blocks(usa_aggregated_reader: ICIOReader):
    """Get the aggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    sectors_info = unfold_countries(usa_aggregated_reader.countries, sectors_mapping)
    # setup blocks
    aggregated_blocks = DisaggregationBlocks.from_technical_coefficients(
        tech_coef=usa_aggregated_reader.technical_coefficients,
        sectors_info=sectors_info,
        output=usa_aggregated_reader.output_from_out,
    )

    return aggregated_blocks


@pytest.fixture(scope="session")
def disaggregated_blocks(usa_reader: ICIOReader):
    """Get the disaggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    # setup blocks
    disaggregated_blocks = DisaggregatedBlocks.from_reader(
        reader=usa_reader, sector_mapping=sectors_mapping
    )

    return disaggregated_blocks


def test_E_block_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the equation for the E block (flows from undisaggregated to disaggregated sectors).

    The equation states that:
    E_{ij} = B_i w_j^n

    where:
    - E_{ij} is the flow from undisaggregated sector i to subsector j
    - B_i is the original flow from sector i to the aggregated sector
    - w_j^n is the weight for subsector j
    """
    sectors_mapping = {"A": ["A01", "A03"]}
    sectors_info = unfold_countries(usa_reader.countries, sectors_mapping)
    # setup blocks
    aggregated_blocks = DisaggregationBlocks.from_technical_coefficients(
        tech_coef=usa_aggregated_reader.technical_coefficients,
        sectors_info=sectors_info,
        output=usa_aggregated_reader.output_from_out,
    )
    disaggregated_blocks = DisaggregatedBlocks.from_reader(
        reader=usa_reader, sector_mapping=sectors_mapping
    )

    E = disaggregated_blocks.get_e_vector(1)
    weights = disaggregated_blocks.get_relative_output_weights(1)

    M1 = aggregated_blocks.get_m1_block(1, weights)

    B = aggregated_blocks.get_B(1)

    # test M1 E = B
    result = M1 @ E
    assert result == pytest.approx(B, rel=1e-2), "M1E ≠ B: Equation does not hold"


def test_F_block_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the equation M₂^n F^n = C^n for the F block.

    According to the disaggregation plan:
    - F^n has shape (k_n × N_K) where:
        * k_n is the number of subsectors for sector n (2 in our case: A01, A03)
        * N_K is the number of undisaggregated sectors across all countries
    - M₂^n has shape (N_K × N_K*k_n)
    - C^n has shape (N_K × 1)

    The equation ensures that when we sum the flows from subsectors (F)
    using the constraint matrix (M₂), we get the original aggregated flows (C).
    """

    sectors_mapping = {"A": ["A01", "A03"]}
    sectors_info = unfold_countries(usa_reader.countries, sectors_mapping)
    # setup blocks
    aggregated_blocks = DisaggregationBlocks.from_technical_coefficients(
        tech_coef=usa_aggregated_reader.technical_coefficients,
        sectors_info=sectors_info,
        output=usa_aggregated_reader.output_from_out,
    )
    disaggregated_blocks = DisaggregatedBlocks.from_reader(
        reader=usa_reader, sector_mapping=sectors_mapping
    )

    F = disaggregated_blocks.get_f_vector(1)
    M2 = aggregated_blocks.get_m2_block(1)
    C = aggregated_blocks.get_C(1)

    # test M2 F = C
    result = M2 @ F
    assert result == pytest.approx(C, rel=1e-2), "M2F ≠ C: Equation does not hold"


# @pytest.mark.parametrize("n", [1, 2])
# make n, l take values in [1,2]
@pytest.mark.parametrize("n, l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_Gnl_block_equation(aggregated_blocks, disaggregated_blocks, n, l):
    """Test the equation M₃^n G^n = D^n for the G block.

    The equation ensures that when we apply the weight constraints (M₃)
    to the subsector flows (G), we get the original aggregated flows (D).
    """
    g_nl = disaggregated_blocks.get_gnl_vector(n, l)
    weights_l = disaggregated_blocks.get_relative_output_weights(l)
    M3 = aggregated_blocks.get_m3_nl_block(n, weights_l)
    d_nl = aggregated_blocks.get_D_nl(n, l)

    # test M3 G = D
    result = M3 @ g_nl
    assert result == pytest.approx(
        d_nl, rel=1e-2
    ), f"M3G ≠ D: Equation does not hold for n={n}, l={l}"


@pytest.mark.parametrize("n", [1, 2])
def test_G_block_equation(aggregated_blocks, disaggregated_blocks, n):
    g = disaggregated_blocks.get_gn_vector(n)
    relative_weights = [
        disaggregated_blocks.get_relative_output_weights(l) for l in range(disaggregated_blocks.m)
    ]
    M3 = aggregated_blocks.get_m3_block(n, relative_weights)
    D = aggregated_blocks.get_D(n)

    col_index = 0
    for l, weights_l in enumerate(relative_weights):
        # check that first row of M3 is equal to M3_11
        M3_nl = aggregated_blocks.get_m3_nl_block(n, weights_l)
        n_cols = len(weights_l) * aggregated_blocks.sectors[n - 1].k  # number of columns in M3_11
        assert np.all(
            M3[l, col_index : col_index + n_cols] == M3_nl
        ), "First row of M3 is not equal to M3_11"
        col_index += n_cols

    # test M3 G = D
    result = M3 @ g
    assert result == pytest.approx(D, rel=1e-1), f"M3G ≠ D: Equation does not hold for n={n}"
