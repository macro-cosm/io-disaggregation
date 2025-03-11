"""Tests for the disaggregation equations.

This module tests the individual equations that make up the disaggregation
constraints. Each equation is tested separately to ensure it holds for
known data.
"""

import numpy as np
import pytest

from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregatedBlocks,
    DisaggregationBlocks,
    unfold_countries,
)
from disag_tools.readers.icio_reader import ICIOReader


def test_m1_block_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
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


@pytest.mark.parametrize("n", [1, 2])
def test_m5_eqn(aggregated_blocks, disaggregated_blocks, n):
    # generate F BLOCK
    aggregated_sectors = disaggregated_blocks.aggregated_sectors_list[n - 1]
    disaggregated_sectors = disaggregated_blocks.sector_mapping[aggregated_sectors]

    indices = [
        disaggregated_blocks.disaggregated_sector_names.index(sector)
        for sector in disaggregated_sectors
    ]
    disaggregated_sectors = [disaggregated_blocks.disaggregated_sector_names[i] for i in indices]

    F = disaggregated_blocks.reordered_matrix.loc[
        disaggregated_sectors, disaggregated_blocks.non_disaggregated_sector_names
    ].values

    # get the x_j/z_n vector
    sector_n = aggregated_blocks.disaggregated_sector_names[n - 1]
    x_j = aggregated_blocks.output.loc[aggregated_blocks.non_disaggregated_sector_names].values
    z_n = aggregated_blocks.output.loc[sector_n]
    weights_vector = x_j / z_n

    result = F @ weights_vector

    M5 = aggregated_blocks.get_m5_block(n)
    F_flat = disaggregated_blocks.get_f_vector(n)
    result2 = M5 @ F_flat

    assert result == pytest.approx(result2, rel=1e-2), f"M5F ≠ Fw: Equation does not hold for n={n}"


@pytest.mark.parametrize("n, l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_m4_nl_eqn(aggregated_blocks, disaggregated_blocks, n, l):
    # generate G BLOCK
    aggregated_sector_n = disaggregated_blocks.aggregated_sectors_list[n - 1]
    aggregated_sector_l = disaggregated_blocks.aggregated_sectors_list[l - 1]

    disaggregated_sectors_n = disaggregated_blocks.sector_mapping[aggregated_sector_n]
    disaggregated_sectors_l = disaggregated_blocks.sector_mapping[aggregated_sector_l]

    indices_n = [
        disaggregated_blocks.disaggregated_sector_names.index(sector)
        for sector in disaggregated_sectors_n
    ]

    indices_l = [
        disaggregated_blocks.disaggregated_sector_names.index(sector)
        for sector in disaggregated_sectors_l
    ]

    disaggregated_sectors_n = [
        disaggregated_blocks.disaggregated_sector_names[i] for i in indices_n
    ]
    disaggregated_sectors_l = [
        disaggregated_blocks.disaggregated_sector_names[i] for i in indices_l
    ]

    gnl = disaggregated_blocks.reordered_matrix.loc[
        disaggregated_sectors_n, disaggregated_sectors_l
    ].values

    weights = disaggregated_blocks.get_relative_output_weights(l)

    output_n = aggregated_blocks.output.loc[aggregated_sector_n]
    output_l = aggregated_blocks.output.loc[aggregated_sector_l]
    ratio = output_l / output_n

    result = ratio * gnl @ weights

    M4_nl = aggregated_blocks.get_m4_nl_block(n, l, weights)
    gnl_vector = disaggregated_blocks.get_gnl_vector(n, l)

    result2 = M4_nl @ gnl_vector

    assert result == pytest.approx(
        result2, rel=1e-2
    ), f"M4G ≠ Gw: Equation does not hold for n={n}, l={l}"


@pytest.mark.parametrize("n", [1, 2])
def test_g_vector_sum(aggregated_blocks, disaggregated_blocks, n):
    results = [
        aggregated_blocks.get_m4_nl_block(n, l, disaggregated_blocks.get_relative_output_weights(l))
        @ disaggregated_blocks.get_gnl_vector(n, l)
        for l in range(1, disaggregated_blocks.m + 1)
    ]
    result = sum(results)

    relative_output_weights = [
        disaggregated_blocks.get_relative_output_weights(l + 1)
        for l in range(disaggregated_blocks.m)
    ]

    M4 = aggregated_blocks.get_m4_block(n, relative_output_weights)
    g = disaggregated_blocks.get_gn_vector(n)

    assert result == pytest.approx(
        M4 @ g, rel=1e-2
    ), f"Sum of M4G ≠ M4G: Equation does not hold for n={n}"


@pytest.mark.parametrize("n", [1, 2])
def test_final_demand_block_equation(aggregated_blocks, disaggregated_blocks, n):
    """Test the equation for the final demand block.

    The equation states that:
    M_5 F + M_4 G + b = w

    where b is the (rescaled) final demand vector, and w is the relative outputs vector.
    """
    f = disaggregated_blocks.get_f_vector(n)
    g = disaggregated_blocks.get_gn_vector(n)
    b = disaggregated_blocks.get_bn_vector(n)
    w = disaggregated_blocks.get_relative_output_weights(n)

    relative_output_weights = [
        disaggregated_blocks.get_relative_output_weights(l + 1)
        for l in range(disaggregated_blocks.m)
    ]

    M4 = aggregated_blocks.get_m4_block(n, relative_output_weights)
    M5 = aggregated_blocks.get_m5_block(n)

    result = M5 @ f + M4 @ g + b

    assert result == pytest.approx(
        w, rel=1e-2
    ), f"M5F + M4G + b ≠ w: Equation does not hold for n={n}"


@pytest.mark.parametrize("n", [1, 2])
def test_large_equation(aggregated_blocks, disaggregated_blocks, n):
    relative_output_weights_list = [
        disaggregated_blocks.get_relative_output_weights(l + 1)
        for l in range(disaggregated_blocks.m)
    ]
    large_m = aggregated_blocks.get_large_m(n, relative_output_weights_list)

    x_n = disaggregated_blocks.get_xn_vector(n)

    result = large_m @ x_n

    result2 = aggregated_blocks.get_y_vector(n, disaggregated_blocks.get_relative_output_weights(n))

    assert result == pytest.approx(result2, rel=1e-2), f"Large equation does not hold for n={n}"
