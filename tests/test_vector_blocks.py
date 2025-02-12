"""Tests for the vector_blocks module."""

import logging

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.vector_blocks import (
    blocks_to_vector,
    extract_E_block,
    extract_F_block,
    extract_G_block,
    flatten_E_block,
    flatten_F_block,
    flatten_G_block,
    reshape_E_block,
    reshape_F_block,
    reshape_G_block,
    vector_to_blocks,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Type alias for readability
Array = NDArray[np.float64]


@pytest.fixture
def sample_E_block() -> tuple[pd.DataFrame, Array]:
    """Create a sample E block and its expected flattened form."""
    # Create 3x2 block (3 undisaggregated sectors, 2 subsectors)
    E = pd.DataFrame(
        [
            [0.1, 0.2],  # First undisaggregated sector to subsectors
            [0.3, 0.4],  # Second undisaggregated sector to subsectors
            [0.5, 0.6],  # Third undisaggregated sector to subsectors
        ]
    )
    # Expected flattened form (row by row)
    E_flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return E, E_flat


@pytest.fixture
def sample_F_block() -> tuple[pd.DataFrame, Array]:
    """Create a sample F block and its expected flattened form."""
    # Create 2x3 block (2 subsectors, 3 undisaggregated sectors)
    F = pd.DataFrame(
        [
            [0.1, 0.3, 0.5],  # First subsector to undisaggregated sectors
            [0.2, 0.4, 0.6],  # Second subsector to undisaggregated sectors
        ]
    )
    # Expected flattened form (column by column)
    F_flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return F, F_flat


@pytest.fixture
def sample_G_block() -> tuple[pd.DataFrame, Array]:
    """Create a sample G block and its expected flattened form."""
    # Create 2x4 block (2 subsectors of n, 4 total subsectors of l)
    G = pd.DataFrame(
        [
            [0.1, 0.2, 0.3, 0.4],  # First subsector of n to all subsectors of l
            [0.5, 0.6, 0.7, 0.8],  # Second subsector of n to all subsectors of l
        ]
    )
    # Expected flattened form (row by row)
    G_flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    return G, G_flat


def test_flatten_E_block(sample_E_block):
    """Test flattening of E block."""
    E, E_flat_expected = sample_E_block
    N_K, k_n = E.shape
    E_flat = flatten_E_block(E, N_K, k_n)
    assert np.allclose(E_flat, E_flat_expected)


def test_flatten_F_block(sample_F_block):
    """Test flattening of F block."""
    F, F_flat_expected = sample_F_block
    k_n, N_K = F.shape
    F_flat = flatten_F_block(F, N_K, k_n)
    assert np.allclose(F_flat, F_flat_expected)


def test_flatten_G_block(sample_G_block):
    """Test flattening of G block."""
    G, G_flat_expected = sample_G_block
    k_n, total_cols = G.shape
    k_l = [2, 2]  # Two sectors l, each with 2 subsectors
    G_flat = flatten_G_block(G, k_n, k_l)
    assert np.allclose(G_flat, G_flat_expected)


def test_reshape_E_block(sample_E_block):
    """Test reshaping of E block."""
    E_expected, E_flat = sample_E_block
    N_K, k_n = E_expected.shape
    E = reshape_E_block(E_flat, N_K, k_n)
    assert np.allclose(E, E_expected.values)


def test_reshape_F_block(sample_F_block):
    """Test reshaping of F block."""
    F_expected, F_flat = sample_F_block
    k_n, N_K = F_expected.shape
    F = reshape_F_block(F_flat, N_K, k_n)
    assert np.allclose(F, F_expected.values)


def test_reshape_G_block(sample_G_block):
    """Test reshaping of G block."""
    G_expected, G_flat = sample_G_block
    k_n, total_cols = G_expected.shape
    k_l = [2, 2]  # Two sectors l, each with 2 subsectors
    G = reshape_G_block(G_flat, k_n, k_l)
    assert np.allclose(G, G_expected.values)


def test_invalid_E_block_shape():
    """Test error handling for invalid E block shape."""
    E = pd.DataFrame([[0.1, 0.2]])  # 1x2 matrix
    N_K, k_n = 2, 2  # Claiming it's 2x2
    with pytest.raises(ValueError, match="E block should have shape"):
        flatten_E_block(E, N_K, k_n)


def test_invalid_F_block_shape():
    """Test error handling for invalid F block shape."""
    F = pd.DataFrame([[0.1, 0.2]])  # 1x2 matrix
    N_K, k_n = 2, 2  # Claiming it's 2x2
    with pytest.raises(ValueError, match="F block should have shape"):
        flatten_F_block(F, N_K, k_n)


def test_invalid_G_block_shape():
    """Test error handling for invalid G block shape."""
    G = pd.DataFrame([[0.1, 0.2]])  # 1x2 matrix
    k_n, k_l = 2, [2, 2]  # Claiming it's 2x4
    with pytest.raises(ValueError, match="G block should have shape"):
        flatten_G_block(G, k_n, k_l)


def test_invalid_E_flat_length():
    """Test error handling for invalid flattened E vector length."""
    E_flat = np.array([0.1, 0.2])  # Length 2
    N_K, k_n = 2, 2  # Should be length 4
    with pytest.raises(ValueError, match="E_flat should have length"):
        reshape_E_block(E_flat, N_K, k_n)


def test_invalid_F_flat_length():
    """Test error handling for invalid flattened F vector length."""
    F_flat = np.array([0.1, 0.2])  # Length 2
    N_K, k_n = 2, 2  # Should be length 4
    with pytest.raises(ValueError, match="F_flat should have length"):
        reshape_F_block(F_flat, N_K, k_n)


def test_invalid_G_flat_length():
    """Test error handling for invalid flattened G vector length."""
    G_flat = np.array([0.1, 0.2])  # Length 2
    k_n, k_l = 2, [2, 2]  # Should be length 8
    with pytest.raises(ValueError, match="G_flat should have length"):
        reshape_G_block(G_flat, k_n, k_l)


def test_blocks_to_vector_and_back(sample_reader: ICIOReader):
    """Test full cycle of converting blocks to vector and back."""
    # Get dimensions from sample data
    sectors_to_disagg = ["AGR"]  # Disaggregate agriculture
    all_sectors = sample_reader.industries
    undisaggregated = [s for s in all_sectors if s not in sectors_to_disagg]
    N_K = len(undisaggregated) * len(sample_reader.countries)
    k_n = len(sectors_to_disagg)
    k_l = [k_n]  # Only one sector being disaggregated

    # Convert blocks to vector
    X_n = blocks_to_vector(sample_reader, sectors_to_disagg)

    # Convert vector back to blocks
    E, F, G, b = vector_to_blocks(X_n, N_K, k_n, k_l)

    # Extract original blocks for comparison
    E_orig = extract_E_block(sample_reader, undisaggregated, sectors_to_disagg)
    F_orig = extract_F_block(sample_reader, undisaggregated, sectors_to_disagg)
    G_orig = extract_G_block(sample_reader, sectors_to_disagg, [sectors_to_disagg])
    b_orig = sample_reader.final_demand.loc[
        pd.MultiIndex.from_product(
            [sample_reader.countries, sectors_to_disagg], names=["CountryInd", "industryInd"]
        )
    ].values

    # Check that blocks match
    assert np.allclose(E, E_orig.values)
    assert np.allclose(F, F_orig.values)
    assert np.allclose(G, G_orig.values)
    assert np.allclose(b, b_orig)


def test_extract_blocks_shapes(sample_reader: ICIOReader):
    """Test that extracted blocks have the correct shapes."""
    # Create test data
    sectors_to_disaggregate = ["AGR"]
    undisaggregated = ["MFG"]

    # Extract blocks
    E = extract_E_block(sample_reader, undisaggregated, sectors_to_disaggregate)
    F = extract_F_block(sample_reader, undisaggregated, sectors_to_disaggregate)
    G = extract_G_block(sample_reader, sectors_to_disaggregate, [sectors_to_disaggregate])

    # Check shapes
    N_K = len(undisaggregated) * len(sample_reader.countries)  # 3 countries * 1 sector = 3
    k_n = len(sectors_to_disaggregate)  # 1 sector
    k_l = [k_n]  # Only one sector being disaggregated
    total_cols = sum(k_l)  # Total number of subsectors across all sectors l

    assert E.shape == (N_K, k_n)  # (3, 1) - flows from all countries to USA
    assert F.shape == (k_n, N_K)  # (1, 3) - flows from USA to all countries
    assert G.shape == (
        k_n,
        total_cols,
    )  # (1, 1) - flows from USA subsectors to all subsectors, summed across countries


def test_extract_blocks_values(sample_reader: ICIOReader):
    """Test specific values in extracted blocks."""
    # Setup
    sectors_to_disagg = ["AGR"]
    all_sectors = sample_reader.industries
    undisaggregated = [s for s in all_sectors if s not in sectors_to_disagg]

    # Extract blocks
    E = extract_E_block(sample_reader, undisaggregated, sectors_to_disagg)
    F = extract_F_block(sample_reader, undisaggregated, sectors_to_disagg)
    G = extract_G_block(sample_reader, sectors_to_disagg, [sectors_to_disagg])

    # Check specific values from sample data
    # These values should match the known structure of sample_reader
    assert E.loc[("USA", "MFG"), ("USA", "AGR")] > 0
    assert F.loc[("USA", "AGR"), ("USA", "MFG")] > 0
    assert G.loc[("USA", "AGR"), "AGR"] > 0  # G block is summed across countries
