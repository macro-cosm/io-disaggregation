"""Tests for the constraint matrix generation functions."""

import logging

import numpy as np
import pytest

from disag_tools.disaggregation.constraints import (
    generate_M1_block,
    generate_M2_block,
    generate_M3_block,
    generate_M4_block,
    generate_M5_block,
    generate_M_n_matrix,
)

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_M1_block_shape():
    """Test that M1 block has the correct shape."""
    N_K = 3  # 3 undisaggregated sectors
    k = 2  # Splitting into 2 subsectors
    weights = np.array([0.6, 0.4])  # Weights sum to 1

    M1 = generate_M1_block(N_K, k, weights)
    assert M1.shape == (N_K, N_K * k)


def test_M1_block_values():
    """Test that M1 block has the correct values."""
    N_K = 2  # 2 undisaggregated sectors
    k = 3  # Splitting into 3 subsectors
    weights = np.array([0.5, 0.3, 0.2])  # Weights sum to 1

    M1 = generate_M1_block(N_K, k, weights)

    # Expected structure:
    # [w1 w2 w3  0  0  0 ]
    # [ 0  0  0 w1 w2 w3 ]
    expected = np.array(
        [
            [0.5, 0.3, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.3, 0.2],
        ]
    )

    assert np.allclose(M1, expected)


def test_M1_invalid_weights():
    """Test that M1 block raises error for invalid weights."""
    N_K = 2
    k = 3
    weights = np.array([0.5, 0.5])  # Wrong number of weights

    with pytest.raises(ValueError, match="Expected 3 weights, got 2"):
        generate_M1_block(N_K, k, weights)


def test_M2_block_shape():
    """Test that M2 block has the correct shape."""
    N_K = 3  # 3 undisaggregated sectors
    k = 2  # Splitting into 2 subsectors

    M2 = generate_M2_block(N_K, k)
    assert M2.shape == (N_K, N_K * k)


def test_M2_block_values():
    """Test that M2 block has the correct values."""
    N_K = 2  # 2 undisaggregated sectors
    k = 3  # Splitting into 3 subsectors

    M2 = generate_M2_block(N_K, k)

    # Expected structure:
    # [1 1 1 0 0 0]  # First row: 3 ones, then zeros
    # [0 0 0 1 1 1]  # Second row: zeros, then 3 ones
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(M2, expected)


def test_M2_sums_to_one():
    """Test that M2 block correctly sums subsector flows."""
    N_K = 2
    k = 4

    M2 = generate_M2_block(N_K, k)

    # Create a sample subsector flow vector
    F = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Flows to first undisaggregated sector
    )  # Flows to second undisaggregated sector

    # When we multiply M2 @ F, each element should be the sum of k flows
    result = M2 @ F
    assert np.allclose(result, [1.0, 2.6])  # 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    # 0.5 + 0.6 + 0.7 + 0.8 = 2.6


def test_M3_block_shape():
    """Test that M3 block has the correct shape."""
    k_n = 2  # Sector n splits into 2 subsectors
    weights_l = [
        np.array([0.6, 0.4]),  # First sector l splits into 2
        np.array([0.3, 0.3, 0.4]),  # Second sector l splits into 3
    ]

    M3 = generate_M3_block(k_n, weights_l)

    K = len(weights_l)  # Number of sectors being disaggregated
    total_cols = sum(len(w_l) * k_n for w_l in weights_l)
    assert M3.shape == (K, total_cols)


def test_M3_block_values():
    """Test that M3 block has the correct values."""
    k_n = 2  # Sector n splits into 2 subsectors
    weights_l = [
        np.array([0.7, 0.3]),  # First sector l splits into 2
        np.array([0.5, 0.5]),  # Second sector l splits into 2
    ]

    M3 = generate_M3_block(k_n, weights_l)

    # Expected structure for k_n = 2, two sectors with 2 subsectors each:
    # For first row (first sector l):
    # [0.7 0.3 0.7 0.3 0 0 0 0]  # Weights repeated k_n times, then zeros
    # For second row (second sector l):
    # [0 0 0 0 0.5 0.5 0.5 0.5]  # Zeros, then weights repeated k_n times
    expected = np.array(
        [
            [0.7, 0.3, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        ]
    )

    assert np.allclose(M3, expected)


def test_M3_block_different_sizes():
    """Test M3 block with sectors split into different numbers of subsectors."""
    k_n = 3  # Sector n splits into 3 subsectors
    weights_l = [
        np.array([0.6, 0.4]),  # First sector l splits into 2
        np.array([0.3, 0.3, 0.4]),  # Second sector l splits into 3
    ]

    M3 = generate_M3_block(k_n, weights_l)

    # Expected shape: 2 rows (K sectors),
    # Columns = (2 subsectors × 3) + (3 subsectors × 3) = 6 + 9 = 15
    assert M3.shape == (2, 15)

    # Check first row (first sector l with 2 subsectors)
    # Pattern should repeat 3 times (k_n = 3)
    first_pattern = np.array([0.6, 0.4])
    for i in range(3):
        start = i * 2
        assert np.allclose(M3[0, start : start + 2], first_pattern)

    # Check second row (second sector l with 3 subsectors)
    # Pattern should repeat 3 times (k_n = 3)
    second_pattern = np.array([0.3, 0.3, 0.4])
    for i in range(3):
        start = 6 + i * 3  # Start after first sector's columns
        assert np.allclose(M3[1, start : start + 3], second_pattern)


def test_M4_block_shape():
    """Test that M4 block has the correct shape."""
    k_n = 2  # Sector n splits into 2 subsectors
    weights_l = [
        np.array([0.6, 0.4]),  # First sector l splits into 2
        np.array([0.3, 0.3, 0.4]),  # Second sector l splits into 3
    ]
    z_l = [100.0, 200.0]  # Outputs for sectors l
    z_n = 150.0  # Output for sector n

    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)

    # Expected shape: k_n rows, sum(k_ℓ * k_n) columns
    total_cols = sum(len(w_l) * k_n for w_l in weights_l)
    assert M4.shape == (k_n, total_cols)


def test_M4_block_values():
    """Test that M4 block has the correct values."""
    k_n = 2  # Sector n splits into 2 subsectors
    weights_l = [
        np.array([0.7, 0.3]),  # First sector l splits into 2
        np.array([0.5, 0.5]),  # Second sector l splits into 2
    ]
    z_l = [150.0, 300.0]  # Outputs for sectors l
    z_n = 100.0  # Output for sector n

    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)

    # Expected structure for k_n = 2, two sectors with 2 subsectors each:
    # First row: scaled weights for first position of each sector
    # [1.05 0.45 0.0 0.0 1.5 1.5 0.0 0.0]
    # Second row: scaled weights for second position of each sector
    # [0.0 0.0 1.05 0.45 0.0 0.0 1.5 1.5]
    expected = np.array(
        [
            [1.05, 0.45, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.05, 0.45, 0.0, 0.0, 1.5, 1.5],
        ]
    )

    assert np.allclose(M4, expected)


def test_M4_block_different_sizes():
    """Test M4 block with sectors split into different numbers of subsectors."""
    k_n = 3  # Sector n splits into 3 subsectors
    weights_l = [
        np.array([0.6, 0.4]),  # First sector l splits into 2
        np.array([0.3, 0.3, 0.4]),  # Second sector l splits into 3
    ]
    z_l = [200.0, 300.0]  # Outputs for sectors l
    z_n = 100.0  # Output for sector n

    M4 = generate_M4_block(k_n, weights_l, z_l, z_n)

    # Expected shape: 3 rows (k_n),
    # Columns = (2 subsectors × 3) + (3 subsectors × 3) = 6 + 9 = 15
    assert M4.shape == (3, 15)

    # Check first row (first position in each sector)
    first_pattern = np.array([1.2, 0.8])  # 0.6 * 2, 0.4 * 2
    second_pattern = np.array([0.9, 0.9, 1.2])  # 0.3 * 3, 0.3 * 3, 0.4 * 3

    # First row: patterns in first position
    assert np.allclose(M4[0, 0:2], first_pattern)
    assert np.allclose(M4[0, 6:9], second_pattern)

    # Second row: patterns in second position
    assert np.allclose(M4[1, 2:4], first_pattern)
    assert np.allclose(M4[1, 9:12], second_pattern)

    # Third row: patterns in third position
    assert np.allclose(M4[2, 4:6], first_pattern)
    assert np.allclose(M4[2, 12:15], second_pattern)


def test_M4_invalid_sector_outputs():
    """Test that M4 block raises error for invalid sector outputs."""
    k_n = 2
    weights_l = [np.array([0.6, 0.4]), np.array([0.5, 0.5])]
    z_l = [100.0]  # Missing one sector output
    z_n = 150.0

    with pytest.raises(ValueError, match="Expected 2 sector outputs, got 1"):
        generate_M4_block(k_n, weights_l, z_l, z_n)


def test_M5_block_shape():
    """Test that M5 block has the correct shape."""
    k_n = 2  # Sector n splits into 2 subsectors
    x = np.array([100.0, 150.0, 200.0])  # 3 undisaggregated sectors
    z_n = 300.0  # Output for sector n

    M5 = generate_M5_block(k_n, x, z_n)

    # Expected shape: k_n rows, N_K * k_n columns
    N_K = len(x)
    assert M5.shape == (k_n, N_K * k_n)


def test_M5_block_values():
    """Test that M5 block has the correct values."""
    k_n = 2  # Sector n splits into 2 subsectors
    x = np.array([150.0, 300.0])  # 2 undisaggregated sectors
    z_n = 100.0  # Output for sector n

    M5 = generate_M5_block(k_n, x, z_n)

    # Expected structure for k_n = 2, two undisaggregated sectors:
    # First row: scaled outputs in first position of each sector block
    # [1.5 0.0 3.0 0.0]
    # Second row: scaled outputs in second position of each sector block
    # [0.0 1.5 0.0 3.0]
    expected = np.array(
        [
            [1.5, 0.0, 3.0, 0.0],
            [0.0, 1.5, 0.0, 3.0],
        ]
    )

    assert np.allclose(M5, expected)


def test_M5_block_different_sizes():
    """Test M5 block with different numbers of subsectors and undisaggregated sectors."""
    k_n = 3  # Sector n splits into 3 subsectors
    x = np.array([100.0, 200.0, 300.0, 400.0])  # 4 undisaggregated sectors
    z_n = 200.0  # Output for sector n

    M5 = generate_M5_block(k_n, x, z_n)

    # Expected shape: 3 rows (k_n), 12 columns (4 sectors × 3 subsectors)
    assert M5.shape == (3, 12)

    # Check that scaled outputs appear in correct positions
    scaled_outputs = x / z_n  # [0.5, 1.0, 1.5, 2.0]

    # Check first row (first position in each sector block)
    for j, scaled_output in enumerate(scaled_outputs):
        assert np.isclose(M5[0, j * k_n], scaled_output)
        assert np.allclose(M5[0, j * k_n + 1 : j * k_n + 3], 0.0)

    # Check second row (second position in each sector block)
    for j, scaled_output in enumerate(scaled_outputs):
        assert np.isclose(M5[1, j * k_n + 1], scaled_output)
        assert np.isclose(M5[1, j * k_n], 0.0)
        assert np.isclose(M5[1, j * k_n + 2], 0.0)

    # Check third row (third position in each sector block)
    for j, scaled_output in enumerate(scaled_outputs):
        assert np.isclose(M5[2, j * k_n + 2], scaled_output)
        assert np.allclose(M5[2, j * k_n : j * k_n + 2], 0.0)


def test_M5_empty_outputs():
    """Test that M5 block raises error for empty outputs array."""
    k_n = 2
    x = np.array([])  # Empty outputs array
    z_n = 100.0

    with pytest.raises(ValueError, match="At least one undisaggregated sector output is required"):
        generate_M5_block(k_n, x, z_n)


def test_M_n_matrix_shape():
    """Test that complete M^n matrix has the correct shape."""
    # Setup test parameters
    k_n = 2  # Sector n splits into 2 subsectors
    N_K = 3  # 3 undisaggregated sectors
    n = 1  # First disaggregated sector
    weights_n = np.array([0.6, 0.4])  # Weights for sector n
    weights_l = [
        np.array([0.7, 0.3]),  # First sector l splits into 2
        np.array([0.5, 0.5]),  # Second sector l splits into 2
    ]
    x = np.array([100.0, 150.0, 200.0])  # Outputs for undisaggregated sectors
    z_l = np.array([300.0, 400.0])  # Outputs for sectors l

    M_n = generate_M_n_matrix(k_n, N_K, n, weights_n, weights_l, x, z_l)

    # Calculate expected dimensions
    K = len(weights_l)  # Number of sectors being disaggregated
    total_cols_G = sum(len(w_l) * k_n for w_l in weights_l)
    total_cols = (N_K * k_n) + (N_K * k_n) + total_cols_G + k_n
    total_rows = N_K + N_K + K + k_n

    assert M_n.shape == (total_rows, total_cols)


def test_M_n_matrix_block_structure():
    """Test that M^n matrix has the correct block structure."""
    # Setup simple test case
    k_n = 2  # Sector n splits into 2 subsectors
    N_K = 2  # 2 undisaggregated sectors
    n = 1  # First disaggregated sector
    weights_n = np.array([0.6, 0.4])  # Weights for sector n
    weights_l = [
        np.array([0.7, 0.3]),  # First sector l splits into 2
    ]
    x = np.array([100.0, 150.0])  # Outputs for undisaggregated sectors
    z_l = np.array([100.0, 200.0])  # Outputs for sectors l

    M_n = generate_M_n_matrix(k_n, N_K, n, weights_n, weights_l, x, z_l)

    # Expected shape: (7, 10)
    # - 7 rows = 2 (M₁) + 2 (M₂) + 1 (M₃) + 2 (M₄/M₅/I)
    # - 10 cols = 4 (E) + 4 (F) + 4 (G) + 2 (b)
    assert M_n.shape == (7, 14)

    # Check M₁ block (top-left)
    M1_expected = np.array(
        [
            [0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.4],
        ]
    )
    assert np.allclose(M_n[0:2, 0:4], M1_expected)

    # Check M₂ block (second block)
    M2_expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(M_n[2:4, 4:8], M2_expected)

    # Check M₃ block (third block)
    M3_expected = np.array([[0.7, 0.3, 0.7, 0.3]])
    assert np.allclose(M_n[4:5, 8:12], M3_expected)

    # Check bottom row blocks
    # M₅ block
    M5_expected = np.array(
        [
            [1.0, 0.0, 1.5, 0.0],
            [0.0, 1.0, 0.0, 1.5],
        ]
    )
    assert np.allclose(M_n[5:7, 4:8], M5_expected)

    # M₄ block
    M4_expected = np.array(
        [
            [1.4, 0.6, 0.0, 0.0],
            [0.0, 0.0, 1.4, 0.6],
        ]
    )
    assert np.allclose(M_n[5:7, 8:12], M4_expected)

    # Identity block
    I_expected = np.eye(2)
    assert np.allclose(M_n[5:7, 12:14], I_expected)


def test_M_n_matrix_different_sizes():
    """Test M^n matrix with different numbers of sectors and subsectors."""
    k_n = 3  # Sector n splits into 3 subsectors
    N_K = 2  # 2 undisaggregated sectors
    n = 1  # First disaggregated sector
    weights_n = np.array([0.4, 0.3, 0.3])  # Weights for sector n
    weights_l = [
        np.array([0.6, 0.4]),  # First sector l splits into 2
        np.array([0.3, 0.3, 0.4]),  # Second sector l splits into 3
    ]
    x = np.array([100.0, 150.0])  # Outputs for undisaggregated sectors
    z_l = np.array([100.0, 200.0, 300.0])  # Outputs for sectors l

    M_n = generate_M_n_matrix(k_n, N_K, n, weights_n, weights_l, x, z_l)

    # Calculate expected dimensions
    K = len(weights_l)  # Number of sectors being disaggregated
    total_cols_G = sum(len(w_l) * k_n for w_l in weights_l)  # 15 = (2*3) + (3*3)
    total_cols = (N_K * k_n) + (N_K * k_n) + total_cols_G + k_n  # 6 + 6 + 15 + 3 = 30
    total_rows = N_K + N_K + K + k_n  # 2 + 2 + 2 + 3 = 9

    assert M_n.shape == (total_rows, total_cols)

    # Verify that blocks have correct shapes
    # M₁ block
    assert np.count_nonzero(M_n[0:2, 0:6]) == 6  # Each row has 3 weights

    # M₂ block
    assert np.count_nonzero(M_n[2:4, 6:12]) == 6  # Each row has 3 ones

    # M₃ block
    assert np.count_nonzero(M_n[4:6, 12:27]) > 0  # Contains repeated weights

    # Bottom row blocks (M₅, M₄, I)
    assert np.count_nonzero(M_n[6:9, 6:12]) > 0  # M₅ block
    assert np.count_nonzero(M_n[6:9, 12:27]) > 0  # M₄ block
    assert np.array_equal(M_n[6:9, 27:30], np.eye(3))  # Identity block


def test_M_n_matrix_invalid_sector_index():
    """Test that M^n matrix raises error for invalid sector index."""
    k_n = 2
    N_K = 2
    weights_n = np.array([0.6, 0.4])
    weights_l = [np.array([0.7, 0.3])]
    x = np.array([100.0, 150.0])
    z_l = np.array([100.0])  # Only one sector

    # Test index too small
    with pytest.raises(ValueError, match="Sector index 0 out of range"):
        generate_M_n_matrix(k_n, N_K, 0, weights_n, weights_l, x, z_l)

    # Test index too large
    with pytest.raises(ValueError, match="Sector index 2 out of range"):
        generate_M_n_matrix(k_n, N_K, 2, weights_n, weights_l, x, z_l)
