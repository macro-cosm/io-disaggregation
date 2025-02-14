"""Tests for the disaggregation equations.

This module tests the individual equations that make up the disaggregation
constraints. Each equation is tested separately to ensure it holds for
known data.
"""

import logging

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
from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.vector_blocks import (
    flatten_E_block,
    flatten_F_block,
    flatten_G_block,
    reshape_E_block,
    reshape_F_block,
    reshape_G_block,
    extract_F_block,
    extract_G_block,
)

logger = logging.getLogger(__name__)


def test_E_block_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the equation M₁^n E^n = B^n for the E block.

    According to the disaggregation plan:
    - E^n has shape (N_K × k_n) where:
        * N_K is the number of undisaggregated sectors across all countries
        * k_n is the number of subsectors for sector n (2 in our case: A01, A03)
    - M₁^n has shape (N_K × (N_K * k_n))
    - B^n has shape (N_K × 1)

    The equation ensures that when we multiply the disaggregated flows (E)
    by the weights matrix (M₁), we get the aggregated flows (B).
    """
    logger.info("Testing E block equation M₁E = B")

    # 1. Setup: Define sectors and get dimensions
    disaggregated_sectors = ["A01", "A03"]  # Sector n's subsectors
    k_n = len(disaggregated_sectors)  # Should be 2
    assert k_n == 2, f"Expected 2 subsectors, got {k_n}"

    # Get undisaggregated sectors (all except A01 and A03)
    undisaggregated = [s for s in usa_reader.industries if s not in disaggregated_sectors]
    N_K = len(undisaggregated) * len(usa_reader.countries)  # Number of undisaggregated sectors across all countries
    logger.debug(f"N_K = {N_K} (sectors: {len(undisaggregated)} × countries: {len(usa_reader.countries)})")
    logger.debug(f"Undisaggregated sectors: {undisaggregated}")
    logger.debug(f"Countries: {usa_reader.countries}")

    # 2. Extract E block from disaggregated data
    E = usa_reader.get_E_block(undisaggregated, disaggregated_sectors)
    logger.debug(f"E block shape: {E.shape}")
    logger.debug(f"E block index: {E.index}")
    logger.debug(f"E block columns: {E.columns}")
    # Verify E block shape: (N_K × k_n)
    assert E.shape == (N_K, k_n), f"E block should have shape ({N_K}, {k_n}), got {E.shape}"

    # 3. Extract B block from aggregated data
    B = usa_aggregated_reader.get_E_block(undisaggregated, ["A"])
    logger.debug(f"B block shape: {B.shape}")
    logger.debug(f"B block index: {B.index}")
    logger.debug(f"B block columns: {B.columns}")
    # Verify B block shape: (N_K × 1)
    assert B.shape == (N_K, 1), f"B block should have shape ({N_K}, 1), got {B.shape}"

    # 4. Calculate weights based on output shares
    total_output_A01 = usa_reader.output_from_sums.loc[("USA", "A01")]
    total_output_A03 = usa_reader.output_from_sums.loc[("USA", "A03")]
    total_output = total_output_A01 + total_output_A03
    weights = np.array([total_output_A01 / total_output, total_output_A03 / total_output])
    logger.debug(f"Weights: A01={weights[0]:.3f}, A03={weights[1]:.3f}")
    # Verify weights sum to 1
    assert weights.sum() == pytest.approx(1.0, rel=1e-2), f"Weights should sum to 1, got {weights.sum()}"

    # 5. Generate M₁ matrix
    M1 = generate_M1_block(N_K, k_n, weights)
    logger.debug(f"M1 matrix shape: {M1.shape}")
    logger.debug(f"M1 matrix first row: {M1[0]}")
    # Verify M₁ shape: (N_K × N_K*k_n)
    assert M1.shape == (
        N_K,
        N_K * k_n,
    ), f"M1 should have shape ({N_K}, {N_K * k_n}), got {M1.shape}"

    # 6. Test M₁E = B
    # First verify matrix multiplication is possible
    logger.debug(f"M1 shape for multiplication: {M1.shape}")
    logger.debug(f"E values shape for multiplication: {E.values.shape}")
    logger.debug(f"Expected shape after multiplication: ({M1.shape[0]}, {E.values.shape[1]})")
    # For M₁E to work:
    # - M₁ is (N_K × N_K*k_n)
    # - E flattened is (N_K*k_n × 1)
    E_flat = E.values.flatten(order="C")
    assert len(E_flat) == N_K * k_n, "Matrix multiplication dimensions mismatch"

    # Compute M₁E
    result = M1 @ E_flat
    logger.debug(f"Result shape: {result.shape}")
    # Verify result shape matches B
    assert result.shape == (N_K,), f"Result should have shape ({N_K},), got {result.shape}"

    # 7. Verify equation holds
    # Check that M₁E = B with a relative error of 1e-2
    assert result == pytest.approx(B.values.flatten(), rel=1e-2), "M₁E ≠ B: Equation does not hold"

    # 8. Additional sanity checks
    # Check that all flows are non-negative
    assert np.all(E.values >= 0), "Found negative flows in E block"
    assert np.all(B.values >= 0), "Found negative flows in B block"

    # Check that M₁ preserves flow magnitudes
    assert np.sum(B.values) == pytest.approx(np.sum(E.values @ weights), rel=1e-2), "M₁ does not preserve total flows"

    logger.info("✓ E block equation verified: M₁E = B")


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
    logger.info("Testing F block equation M₂F = C")

    # 1. Setup: Define sectors and get dimensions
    disaggregated_sectors = ["A01", "A03"]  # Sector n's subsectors
    k_n = len(disaggregated_sectors)  # Should be 2
    assert k_n == 2, f"Expected 2 subsectors, got {k_n}"

    # Get undisaggregated sectors (all except A01 and A03)
    undisaggregated = [s for s in usa_reader.industries if s not in disaggregated_sectors]
    N_K = len(undisaggregated) * len(usa_reader.countries)  # Number of undisaggregated sectors across all countries
    logger.debug(f"N_K = {N_K} (sectors: {len(undisaggregated)} × countries: {len(usa_reader.countries)})")
    logger.debug(f"Undisaggregated sectors: {undisaggregated}")
    logger.debug(f"Countries: {usa_reader.countries}")

    # 2. Extract F block from disaggregated data
    F = usa_reader.get_F_block(undisaggregated, disaggregated_sectors)
    logger.debug(f"F block shape: {F.shape}")
    logger.debug(f"F block index: {F.index}")
    logger.debug(f"F block columns: {F.columns}")

    # Get aggregated F block for comparison
    F_agg = usa_aggregated_reader.get_F_block(undisaggregated, ["A"])
    logger.debug(f"Aggregated F block shape: {F_agg.shape}")
    logger.debug(f"Aggregated F block index: {F_agg.index}")
    logger.debug(f"Aggregated F block columns: {F_agg.columns}")

    # 4. Generate M₂ matrix
    M2 = generate_M2_block(N_K, k_n)
    logger.debug(f"M2 matrix shape: {M2.shape}")
    logger.debug(f"M2 matrix first row: {M2[0]}")
    # Verify M₂ shape: (N_K × N_K*k_n)
    assert M2.shape == (
        N_K,
        N_K * k_n,
    ), f"M2 should have shape ({N_K}, {N_K * k_n}), got {M2.shape}"

    # 5. Test M₂F = C
    # First verify matrix multiplication is possible
    logger.debug(f"M2 shape for multiplication: {M2.shape}")
    logger.debug(f"F values shape for multiplication: {F.values.shape}")
    logger.debug(f"Expected shape after multiplication: ({M2.shape[0]}, {F.values.shape[1]})")
    # For M₂F to work:
    # - M₂ is (N_K × N_K*k_n)
    # - F flattened is (N_K*k_n × 1)
    F_flat = F.values.flatten(order="F")  # Note: Using F-order to match column-major flattening
    assert len(F_flat) == N_K * k_n, "Matrix multiplication dimensions mismatch"

    # Compute M₂F
    result = M2 @ F_flat
    logger.debug(f"Result shape: {result.shape}")
    # Verify result shape matches C
    assert result.shape == (N_K,), f"Result should have shape ({N_K},), got {result.shape}"

    # 6. Verify equation holds
    # Check that M₂F = C with a relative error of 1e-2
    assert result == pytest.approx(F_agg.values.flatten(), rel=1e-2), "M₂F ≠ C: Equation does not hold"

    # 7. Additional sanity checks
    # Check that all flows are non-negative
    assert np.all(F.values >= 0), "Found negative flows in F block"
    assert np.all(F_agg.values >= 0), "Found negative flows in aggregated F block"

    # Check that M₂ preserves flow magnitudes
    assert np.sum(F_agg.values) == pytest.approx(np.sum(F.values), rel=1e-2), "M₂ does not preserve total flows"

    logger.info("✓ F block equation verified: M₂F = C")


def test_G_block_equation(usa_reader: ICIOReader, usa_aggregated_reader: ICIOReader):
    """Test the equation M₃^n G^n = D^n for the G block.

    According to the disaggregation plan:
    - G^n has shape (k_n × sum(k_ℓ)) where:
        * k_n is the number of subsectors for sector n
        * k_ℓ are the numbers of subsectors for each sector ℓ
        * sum(k_ℓ) is the total number of subsectors across all sectors
    - M₃^n has shape (K × k_n*sum(k_ℓ)) where:
        * K is the number of sectors being disaggregated
    - D^n has shape (K × 1)

    The equation ensures that when we apply the weight constraints (M₃)
    to the subsector flows (G), we get the original aggregated flows (D).
    """
    logger.info("Testing G block equation M₃G = D")

    # 1. Setup: Define sectors and get dimensions
    disaggregated_sectors = ["A01", "A03"]  # Sector n's subsectors
    k_n = len(disaggregated_sectors)  # Should be 2
    assert k_n == 2, f"Expected 2 subsectors, got {k_n}"

    # For this test, we're disaggregating both A01 and A03 into subsectors
    sectors_l = [disaggregated_sectors]  # List of lists of subsectors for each sector ℓ
    K = len(sectors_l)  # Number of sectors being disaggregated
    k_l = [len(sectors) for sectors in sectors_l]  # Number of subsectors for each sector
    total_subsectors = sum(k_l)  # Total number of subsectors across all sectors
    logger.debug(f"K = {K} sectors being disaggregated")
    logger.debug(f"k_l = {k_l} subsectors per sector")
    logger.debug(f"Total subsectors = {total_subsectors}")

    # 2. Extract G block from disaggregated data
    # G block: flows between subsectors and other A sectors
    # Note: Each sector in sectors_l needs to be in its own list since they're separate sectors
    G = usa_reader.get_G_block(disaggregated_sectors, sectors_l)
    logger.debug(f"G block shape: {G.shape}")
    logger.debug(f"G block index: {G.index}")
    logger.debug(f"G block columns: {G.columns}")
    # Verify G block shape: (k_n × sum(k_ℓ))
    assert G.shape == (
        k_n,
        total_subsectors,
    ), f"G block should have shape ({k_n}, {total_subsectors}), got {G.shape}"

    # 3. Extract D block from aggregated data
    D = usa_aggregated_reader.get_G_block(["A"], [["A"]])
    logger.debug(f"D block shape: {D.shape}")
    logger.debug(f"D block index: {D.index}")
    logger.debug(f"D block columns: {D.columns}")
    # Verify D block shape: (1 × K)
    assert D.shape == (1, K), f"D block should have shape (1, {K}), got {D.shape}"

    # 4. Calculate weights based on output shares
    total_output_A01 = usa_reader.output_from_sums.loc[("USA", "A01")]
    total_output_A03 = usa_reader.output_from_sums.loc[("USA", "A03")]
    total_output = total_output_A01 + total_output_A03
    weights = np.array([total_output_A01 / total_output, total_output_A03 / total_output])
    logger.debug(f"Weights: A01={weights[0]:.3f}, A03={weights[1]:.3f}")
    # Verify weights sum to 1
    assert weights.sum() == pytest.approx(1.0, rel=1e-2), f"Weights should sum to 1, got {weights.sum()}"

    # 5. Generate M₃ matrix
    M3 = generate_M3_block(k_n, [weights])  # Pass list of weights for each sector ℓ
    logger.debug(f"M3 matrix shape: {M3.shape}")
    logger.debug(f"M3 matrix first row: {M3[0]}")
    # Verify M₃ shape: (K × k_n*sum(k_ℓ))
    expected_cols = k_n * total_subsectors
    assert M3.shape == (
        K,
        expected_cols,
    ), f"M3 should have shape ({K}, {expected_cols}), got {M3.shape}"

    # 6. Test M₃G = D
    # First verify matrix multiplication is possible
    logger.debug(f"M3 shape for multiplication: {M3.shape}")
    logger.debug(f"G values shape for multiplication: {G.values.shape}")
    logger.debug(f"Expected shape after multiplication: ({M3.shape[0]}, {G.values.shape[1]})")
    # For M₃G to work:
    # - M₃ is (K × k_n*sum(k_ℓ))
    # - G flattened is (k_n*sum(k_ℓ) × 1)
    G_flat = G.values.flatten(order="C")  # Use C-order as we're working with rows
    assert len(G_flat) == k_n * total_subsectors, "Matrix multiplication dimensions mismatch"

    # Compute M₃G
    result = M3 @ G_flat
    logger.debug(f"Result shape: {result.shape}")
    # Verify result shape matches D
    assert result.shape == (K,), f"Result should have shape ({K},), got {result.shape}"

    # 7. Verify equation holds
    # Check that M₃G = D with a relative error of 1e-2
    assert result == pytest.approx(D.values.flatten(), rel=1e-2), "M₃G ≠ D: Equation does not hold"

    # 8. Additional sanity checks
    # Check that all flows are non-negative
    assert np.all(G.values >= 0), "Found negative flows in G block"
    assert np.all(D.values >= 0), "Found negative flows in D block"

    # Check that M₃ preserves flow magnitudes (weighted sum)
    weighted_G_sum = np.sum(G.values @ weights)
    assert np.sum(D.values) == pytest.approx(weighted_G_sum, rel=1e-2), "M₃ does not preserve weighted flows"

    logger.info("✓ G block equation verified: M₃G = D")


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
    logger.info("Testing final demand consistency equation...")

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

    # Log all dimensions for debugging
    logger.debug(f"k_n = {k_n} (subsectors: {disaggregated_sectors})")
    logger.debug(f"N_u = {N_u} (undisaggregated sectors)")
    logger.debug(f"N_c = {N_c} (countries: {usa_reader.countries})")
    logger.debug(f"N_K = {N_K} (total undisaggregated sector-country pairs)")
    logger.debug(f"K = {K} sectors being disaggregated")
    logger.debug(f"k_l = {k_l} subsectors per sector")
    logger.debug(f"Total subsectors = {total_subsectors}")

    # 2. Extract blocks and verify their shapes
    # F block: flows from subsectors to undisaggregated sectors
    F = usa_reader.get_F_block(undisaggregated, disaggregated_sectors)
    logger.debug(f"F block shape: {F.shape}")
    logger.debug(f"Expected F shape: ({k_n}, {N_K})")
    assert F.shape == (k_n, N_K), f"F block should have shape ({k_n}, {N_K}), got {F.shape}"

    # G block: flows between subsectors
    G = usa_reader.get_G_block(disaggregated_sectors, sectors_l)
    logger.debug(f"G block shape: {G.shape}")
    logger.debug(f"Expected G shape: ({k_n}, {total_subsectors})")
    assert G.shape == (k_n, total_subsectors), f"G block should have shape ({k_n}, {total_subsectors}), got {G.shape}"

    # 3. Get outputs and calculate weights
    # Get z_1 and z_2 (outputs of A01 and A03)
    z_1 = usa_reader.output_from_sums.loc[("USA", "A01")]
    z_2 = usa_reader.output_from_sums.loc[("USA", "A03")]
    z_n = z_1 + z_2  # Total output of sector A
    logger.debug(f"z_1 (output of A01): {z_1}")
    logger.debug(f"z_2 (output of A03): {z_2}")
    logger.debug(f"z_n (total output of sector A): {z_n}")

    # Calculate weights w_1 and w_2
    w_1 = z_1 / z_n
    w_2 = z_2 / z_n
    weights_n = np.array([w_1, w_2])
    logger.debug(f"z_1: {z_1}, z_2: {z_2}, z_n: {z_n}")
    logger.debug(f"w_1 = z_1/z_n = {w_1:.6f}")
    logger.debug(f"w_2 = z_2/z_n = {w_2:.6f}")
    logger.debug(f"Weights: w_1 (A01)={w_1:.6f}, w_2 (A03)={w_2:.6f}")
    assert weights_n.sum() == pytest.approx(1.0, rel=1e-2), f"Weights should sum to 1, got {weights_n.sum()}"

    # For M₄, we need weights and outputs for the subsectors
    weights_l = [weights_n]  # List of weights for sector A
    z_l = np.array([z_n])  # Output of sector A
    logger.debug(f"z_l (output of sector A): {z_l}")

    # Get x (outputs of undisaggregated sectors)
    x = np.array([usa_reader.output_from_sums.loc[(c, s)] for c in usa_reader.countries for s in undisaggregated])
    logger.debug(f"x shape: {x.shape}")
    logger.debug(f"Expected x shape: ({N_K},)")
    assert x.shape == (N_K,), f"x should have shape ({N_K},), got {x.shape}"

    # 4. Generate M₄ and M₅ matrices
    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)
    logger.debug(f"M4 shape: {M4.shape}")
    expected_M4_shape = (k_n, k_n * total_subsectors)
    logger.debug(f"Expected M4 shape: {expected_M4_shape}")
    assert M4.shape == expected_M4_shape, f"M4 should have shape {expected_M4_shape}, got {M4.shape}"

    M5 = generate_M5_block(k_n, x, z_n)
    logger.debug(f"M5 shape: {M5.shape}")
    expected_M5_shape = (k_n, k_n * N_K)
    logger.debug(f"Expected M5 shape: {expected_M5_shape}")
    assert M5.shape == expected_M5_shape, f"M5 should have shape {expected_M5_shape}, got {M5.shape}"

    # Get final demand for subsectors and normalize by z_n
    b = np.array([usa_reader.final_demand[("USA", s)] for s in disaggregated_sectors]) / z_n
    logger.debug(f"b shape: {b.shape}")
    logger.debug(f"Expected b shape: ({k_n},)")
    assert b.shape == (k_n,), f"b should have shape ({k_n},), got {b.shape}"

    # 5. Test equation M₅F + M₄G + b = w^n
    # Flatten F and G blocks using vector_blocks functions
    F_flat = flatten_F_block(F, N_K, k_n)
    logger.debug(f"F_flat shape: {F_flat.shape}")
    logger.debug(f"Expected F_flat shape: ({k_n * N_K},)")
    logger.debug(f"Original F block:\n{F.values}")
    logger.debug(f"F_flat:\n{F_flat}")
    assert F_flat.shape == (k_n * N_K,), f"F_flat should have shape ({k_n * N_K},), got {F_flat.shape}"

    G_flat = flatten_G_block(G, k_n, [k_n])
    logger.debug(f"G_flat shape: {G_flat.shape}")
    logger.debug(f"Expected G_flat shape: ({k_n * total_subsectors},)")
    assert G_flat.shape == (
        k_n * total_subsectors,
    ), f"G_flat should have shape ({k_n * total_subsectors},), got {G_flat.shape}"

    # Compute left side of equation
    logger.debug("Computing M₅F + M₄G + b...")
    M5F = M5 @ F_flat
    logger.debug(f"M5 @ F_flat shape: {M5F.shape}")
    logger.debug(f"M5 @ F_flat:\n{M5F}")
    logger.debug(f"M5 block:\n{M5}")
    logger.debug(f"F_flat:\n{F_flat}")

    M4G = M4 @ G_flat
    logger.debug(f"M4 @ G_flat shape: {M4G.shape}")
    logger.debug(f"M4 @ G_flat:\n{M4G}")
    logger.debug(f"M4 block:\n{M4}")
    logger.debug(f"G_flat:\n{G_flat}")

    logger.debug(f"b vector:\n{b}")
    result = M5F + M4G + b
    logger.debug(f"M5F contribution: {M5F}")
    logger.debug(f"M4G contribution: {M4G}")
    logger.debug(f"b contribution: {b}")
    logger.debug(f"Result shape: {result.shape}")
    logger.debug(f"Expected result shape: {weights_n.shape}")
    logger.debug(f"Result: {result}")
    logger.debug(f"Expected weights: {weights_n}")

    # Verify equation holds with a more appropriate tolerance for numerical computations
    assert result == pytest.approx(weights_n, rel=2.5e-2), (
        f"Final demand consistency equation does not hold.\n"
        f"Expected: {weights_n}\n"
        f"Got: {result}\n"
        f"Relative differences: {np.abs(result - weights_n) / weights_n}"
    )

    # Log the actual differences for transparency
    logger.debug("Final demand consistency equation check:")
    logger.debug(f"Expected weights: {weights_n}")
    logger.debug(f"Obtained result: {result}")
    logger.debug(f"Absolute differences: {np.abs(result - weights_n)}")
    logger.debug(f"Relative differences: {np.abs(result - weights_n) / weights_n}")

    logger.info("✓ Final demand consistency equation verified: M₅F + M₄G + b = w^n")
