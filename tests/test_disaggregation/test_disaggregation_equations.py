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
)
from disag_tools.readers.icio_reader import ICIOReader

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
    N_K = len(undisaggregated) * len(
        usa_reader.countries
    )  # Number of undisaggregated sectors across all countries
    logger.debug(
        f"N_K = {N_K} (sectors: {len(undisaggregated)} × countries: {len(usa_reader.countries)})"
    )
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
    assert weights.sum() == pytest.approx(
        1.0, rel=1e-2
    ), f"Weights should sum to 1, got {weights.sum()}"

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
    assert np.sum(B.values) == pytest.approx(
        np.sum(E.values @ weights), rel=1e-2
    ), "M₁ does not preserve total flows"

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
    N_K = len(undisaggregated) * len(
        usa_reader.countries
    )  # Number of undisaggregated sectors across all countries
    logger.debug(
        f"N_K = {N_K} (sectors: {len(undisaggregated)} × countries: {len(usa_reader.countries)})"
    )
    logger.debug(f"Undisaggregated sectors: {undisaggregated}")
    logger.debug(f"Countries: {usa_reader.countries}")

    # 2. Extract F block from disaggregated data
    F = usa_reader.get_F_block(undisaggregated, disaggregated_sectors)
    logger.debug(f"F block shape: {F.shape}")
    logger.debug(f"F block index: {F.index}")
    logger.debug(f"F block columns: {F.columns}")
    # Verify F block shape: (k_n × N_K)
    assert F.shape == (k_n, N_K), f"F block should have shape ({k_n}, {N_K}), got {F.shape}"

    # 3. Extract C block from aggregated data
    C = usa_aggregated_reader.get_F_block(undisaggregated, ["A"])
    logger.debug(f"C block shape: {C.shape}")
    logger.debug(f"C block index: {C.index}")
    logger.debug(f"C block columns: {C.columns}")
    # Verify C block shape: (1 × N_K)
    assert C.shape == (1, N_K), f"C block should have shape (1, {N_K}), got {C.shape}"

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
    assert result == pytest.approx(C.values.flatten(), rel=1e-2), "M₂F ≠ C: Equation does not hold"

    # 7. Additional sanity checks
    # Check that all flows are non-negative
    assert np.all(F.values >= 0), "Found negative flows in F block"
    assert np.all(C.values >= 0), "Found negative flows in C block"

    # Check that M₂ preserves flow magnitudes
    assert np.sum(C.values) == pytest.approx(
        np.sum(F.values), rel=1e-2
    ), "M₂ does not preserve total flows"

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
    assert weights.sum() == pytest.approx(
        1.0, rel=1e-2
    ), f"Weights should sum to 1, got {weights.sum()}"

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
    assert np.sum(D.values) == pytest.approx(
        weighted_G_sum, rel=1e-2
    ), "M₃ does not preserve weighted flows"

    logger.info("✓ G block equation verified: M₃G = D")
