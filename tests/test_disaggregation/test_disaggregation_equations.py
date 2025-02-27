"""Tests for the disaggregation equations.

This module tests the individual equations that make up the disaggregation
constraints. Each equation is tested separately to ensure it holds for
known data.
"""

import numpy as np
import pandas as pd
import pytest

from disag_tools.disaggregation.constraints import (
    generate_M1_block,
    generate_M2_block,
    generate_M3_block,
    generate_M4_block,
    generate_M5_block,
)
from disag_tools.readers.disaggregation_blocks import (
    SectorId,
    DisaggregationBlocks,
    get_sectors_info,
    DisaggregatedBlocks,
    unfold_countries,
)
from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.vector_blocks import (
    extract_E_block,
    extract_F_block,
    extract_G_block,
    flatten_E_block,
    flatten_F_block,
    flatten_G_block,
    reshape_E_block,
    reshape_F_block,
    reshape_G_block,
)


@pytest.fixture(scope="session")
def aggregated_blocks(usa_aggregated_reader: ICIOReader):
    """Get the aggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    sectors_info = unfold_countries(usa_aggregated_reader.countries, sectors_mapping)
    # setup blocks
    aggregated_blocks = DisaggregationBlocks.from_technical_coefficients(
        tech_coef=usa_aggregated_reader.technical_coefficients, sectors_info=sectors_info
    )

    return aggregated_blocks


@pytest.fixture(scope="session")
def disaggregated_blocks(usa_reader: ICIOReader):
    """Get the disaggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    # setup blocks
    disaggregated_blocks = DisaggregatedBlocks.from_technical_coefficients(
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
        tech_coef=usa_aggregated_reader.technical_coefficients, sectors_info=sectors_info
    )
    disaggregated_blocks = DisaggregatedBlocks.from_technical_coefficients(
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
        tech_coef=usa_aggregated_reader.technical_coefficients, sectors_info=sectors_info
    )
    disaggregated_blocks = DisaggregatedBlocks.from_technical_coefficients(
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


def test_final_demand_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the final demand consistency equation M₅^n F^n + M₄^n G^n + b^n = w^n.

    According to the disaggregation plan:
    - F^n has shape (k_n × N_K): flows from subsectors to undisaggregated sectors
    - G^n has shape (k_n × sum(k_ℓ)): flows between subsectors
    - M₅^n has shape (k_n × k_n*N_K): scales undisaggregated sector outputs by 1/z_n
    - M₄^n has shape (k_n × k_n*sum(k_ℓ)): scales sector weights by output ratios z_ℓ/z_n
    - b^n has shape (k_n): final demand for each subsector
    - w^n has shape (k_n): target weights vector

    where:
    - k_n is the number of subsectors for sector n (2 in our case: A01, A03)
    - N_K = N_u * N_c where:
        * N_u is the number of undisaggregated sectors
        * N_c is the number of countries (USA + ROW in our case)
    - k_ℓ is the number of subsectors for each sector ℓ
    """
    # 1. Setup: Define sectors and get dimensions
    disaggregated_sectors = ["A01", "A03"]  # Subsectors of sector A
    k_n = len(disaggregated_sectors)  # Should be 2 (A01 and A03)
    assert k_n == 2, f"Expected 2 subsectors, got {k_n}"

    # Get undisaggregated sectors (all except A01 and A03)
    undisaggregated = [s for s in usa_reader.industries if s not in disaggregated_sectors]
    N_u = len(undisaggregated)  # Number of undisaggregated sectors
    N_c = len(usa_reader.countries)  # Should be 2 (USA + ROW)
    N_K = N_u * N_c  # Total number of undisaggregated sector-country pairs

    # For this test, we're disaggregating both A01 and A03 into subsectors
    sectors_l = [disaggregated_sectors]  # List of lists of subsectors for each sector ℓ
    K = len(sectors_l)  # Number of sectors being disaggregated
    k_l = [len(sectors) for sectors in sectors_l]  # Number of subsectors for each sector
    total_subsectors = sum(k_l)  # Total number of subsectors across all sectors

    # 2. Extract blocks and verify their shapes
    F = usa_reader.get_F_block(undisaggregated, disaggregated_sectors)
    G = usa_reader.get_G_block(disaggregated_sectors, sectors_l)

    # 3. Get outputs and calculate weights
    z_1 = usa_reader.output_from_sums.loc[("USA", "A01")]
    z_2 = usa_reader.output_from_sums.loc[("USA", "A03")]
    z_n = z_1 + z_2  # Total output of sector A

    # Calculate weights w_1 and w_2
    w_1 = z_1 / z_n
    w_2 = z_2 / z_n
    weights_n = np.array([w_1, w_2])

    # For M₄, we need weights and outputs for the subsectors
    weights_l = [weights_n]  # List of weights for sector A
    z_l = np.array([z_n])  # Output of sector A

    # Get x (outputs of undisaggregated sectors)
    x = np.array(
        [
            usa_reader.output_from_sums.loc[(c, s)]
            for c in usa_reader.countries
            for s in undisaggregated
        ]
    )

    # 4. Generate M₄ and M₅ matrices
    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)
    M5 = generate_M5_block(k_n, x, z_n)

    # Get final demand for subsectors and normalize by z_n
    b = np.array([usa_reader.final_demand[("USA", s)] for s in disaggregated_sectors]) / z_n

    # 5. Test equation M₅F + M₄G + b = w^n
    F_flat = flatten_F_block(F, N_K, k_n)
    G_flat = flatten_G_block(G, k_n, [k_n])
    result = M5 @ F_flat + M4 @ G_flat + b
    assert result == pytest.approx(weights_n, rel=2.5e-2), (
        f"Final demand consistency equation does not hold.\n"
        f"Expected: {weights_n}\n"
        f"Got: {result}\n"
        f"Relative differences: {np.abs(result - weights_n) / weights_n}"
    )


def test_entire_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the entire disaggregation equation."""
    # 1. Setup: Define sectors and get dimensions
    disaggregated_sectors = ["A01", "A03"]  # Subsectors of sector A
    k_n = len(disaggregated_sectors)  # Should be 2 (A01 and A03)
    assert k_n == 2, f"Expected 2 subsectors, got {k_n}"

    # Get undisaggregated sectors (all except A01 and A03)
    undisaggregated = [s for s in usa_reader.industries if s not in disaggregated_sectors]
    N_u = len(undisaggregated)  # Number of undisaggregated sectors
    N_c = len(usa_reader.countries)  # Should be 2 (USA + ROW)
    N_K = N_u * N_c  # Total number of undisaggregated sector-country pairs

    # For this test, we're disaggregating both A01 and A03 into subsectors
    sectors_l = [disaggregated_sectors]  # List of lists of subsectors for each sector ℓ
    K = len(sectors_l)  # Number of sectors being disaggregated
    k_l = [len(sectors) for sectors in sectors_l]  # Number of subsectors for each sector
    total_subsectors = sum(k_l)  # Total number of subsectors across all sectors

    # we must get the weights
    total_output_A01 = usa_reader.output_from_sums.loc[("USA", "A01")]
    total_output_A03 = usa_reader.output_from_sums.loc[("USA", "A03")]
    total_output = total_output_A01 + total_output_A03
    weights = np.array([total_output_A01 / total_output, total_output_A03 / total_output])

    # data needed for final demand
    z_1 = usa_reader.output_from_sums.loc[("USA", "A01")]
    z_2 = usa_reader.output_from_sums.loc[("USA", "A03")]
    z_n = z_1 + z_2  # Total output of sector A

    # get E block
    E = usa_reader.get_E_block(undisaggregated, disaggregated_sectors)
    assert E.shape == (N_K, k_n), f"E block should have shape ({N_K}, {k_n}), got {E.shape}"

    # get F block
    F = usa_reader.get_F_block(undisaggregated, disaggregated_sectors)
    assert F.shape == (k_n, N_K), f"F block should have shape ({k_n}, {N_K}), got {F.shape}"

    # get G block
    G = usa_reader.get_G_block(disaggregated_sectors, sectors_l)
    assert G.shape == (
        k_n,
        total_subsectors,
    ), f"G block should have shape ({k_n}, {total_subsectors}), got {G.shape}"

    # Get final demand for subsectors and normalize by z_n
    b = np.array([usa_reader.final_demand[("USA", s)] for s in disaggregated_sectors]) / z_n

    # define weights_l and x
    weights_l = [weights]
    z_l = np.array([z_n])
    x = np.array(
        [
            usa_reader.output_from_out.loc[(c, s)]
            for c in usa_reader.countries
            for s in undisaggregated
        ]
    )

    # get M1 block
    M1 = generate_M1_block(N_K, k_n, weights)
    assert M1.shape == (
        N_K,
        N_K * k_n,
    ), f"M1 block should have shape ({N_K}, {N_K * k_n}), got {M1.shape}"

    # get M2 block
    M2 = generate_M2_block(N_K, k_n)
    assert M2.shape == (
        N_K,
        N_K * k_n,
    ), f"M2 block should have shape ({N_K}, {N_K * k_n}), got {M2.shape}"

    # get M3 block
    M3 = generate_M3_block(k_n, [weights])
    assert M3.shape == (
        k_n,
        total_subsectors,
    ), f"M3 block should have shape ({k_n}, {total_subsectors}), got {M3.shape}"

    # get M4 block
    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)
    assert M4.shape == (
        k_n,
        k_n * N_K,
    ), f"M4 block should have shape ({k_n}, {k_n * N_K}), got {M4.shape}"

    # get M5 block
    M5 = generate_M5_block(k_n, x, z_n)
    assert M5.shape == (
        k_n,
        k_n * N_K,
    ), f"M5 block should have shape ({k_n}, {k_n * N_K}), got {M5.shape}"
