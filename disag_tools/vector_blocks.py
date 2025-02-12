"""Module for transforming between block matrices and vectors in disaggregation problems.

This module provides functionality to:
1. Extract blocks from technical coefficient matrices
2. Convert blocks to flattened vectors (for use in optimization)
3. Convert vectors back to block form (for reconstruction)
"""

import logging
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from disag_tools.readers.icio_reader import ICIOReader

# Type alias for float arrays
Array: TypeAlias = NDArray[np.float64]

logger = logging.getLogger(__name__)


# Block → Vector (Flattening) Operations


def flatten_E_block(E: pd.DataFrame, N_K: int, k_n: int) -> Array:
    """
    Flatten the E block (flows from undisaggregated to disaggregated sectors).

    Following Eq. (X), E is flattened as:
    E_{11}^n, E_{12}^n, ..., E_{1k_n}^n,
    E_{21}^n, E_{22}^n, ..., E_{2k_n}^n,
    ...,
    E_{(N-K)1}^n, E_{(N-K)2}^n, ..., E_{(N-K)k_n}^n

    Args:
        E: DataFrame with flows from undisaggregated to disaggregated sectors
        N_K: Number of undisaggregated sectors
        k_n: Number of subsectors for the disaggregated sector

    Returns:
        Flattened array of shape (N_K * k_n,)
    """
    if E.shape != (N_K, k_n):
        raise ValueError(f"E block should have shape ({N_K}, {k_n}), got {E.shape}")

    # Flatten row by row (C-order)
    return E.values.flatten(order="C")


def flatten_F_block(F: pd.DataFrame, N_K: int, k_n: int) -> Array:
    """
    Flatten the F block (flows from disaggregated to undisaggregated sectors).

    Following Eq. (X), F is flattened as:
    F_{11}^n, F_{21}^n, ..., F_{k_n1}^n,
    F_{12}^n, F_{22}^n, ..., F_{k_n2}^n,
    ...,
    F_{1(N-K)}^n, F_{2(N-K)}^n, ..., F_{k_n(N-K)}^n

    Args:
        F: DataFrame with flows from disaggregated to undisaggregated sectors
        N_K: Number of undisaggregated sectors
        k_n: Number of subsectors for the disaggregated sector

    Returns:
        Flattened array of shape (N_K * k_n,)
    """
    if F.shape != (k_n, N_K):
        raise ValueError(f"F block should have shape ({k_n}, {N_K}), got {F.shape}")

    # Flatten column by column (F-order)
    return F.values.flatten(order="F")


def flatten_G_block(G: pd.DataFrame, k_n: int, k_l: list[int]) -> Array:
    """
    Flatten the G block (flows between disaggregated sectors).

    Following Eq. (X), G is flattened as:
    G_{11}^{n1}, G_{12}^{n1}, ..., G_{1k_1}^{n1},
    G_{21}^{n1}, G_{22}^{n1}, ..., G_{2k_1}^{n1},
    ...,
    G_{k_n1}^{n1}, G_{k_n2}^{n1}, ..., G_{k_nk_1}^{n1},
    ...,
    G_{11}^{nK}, G_{12}^{nK}, ..., G_{1k_K}^{nK},
    ...,
    G_{k_n1}^{nK}, G_{k_n2}^{nK}, ..., G_{k_nk_K}^{nK}

    Args:
        G: DataFrame with flows between disaggregated sectors
        k_n: Number of subsectors for sector n
        k_l: List of numbers of subsectors for each sector l

    Returns:
        Flattened array of shape (k_n * sum(k_l),)
    """
    # For each sector l, we have k_l subsectors in each country
    # So the total number of columns should be sum(k_l) * num_countries
    expected_cols = sum(k_l)
    if G.shape[0] != k_n:
        raise ValueError(f"G block should have shape ({k_n}, N), got {G.shape}")
    if G.shape[1] % expected_cols != 0:
        raise ValueError(
            f"G block should have number of columns divisible by {expected_cols}, got {G.shape[1]}"
        )

    num_countries = G.shape[1] // expected_cols
    logger.debug(f"G block has {num_countries} countries")
    logger.debug(f"G block shape: {G.shape}")
    logger.debug(f"k_n: {k_n}, k_l: {k_l}")

    # Reshape to combine countries and subsectors
    # This gives us the shape we need for the flattening equation
    G_reshaped = G.values.reshape((k_n, expected_cols))

    # Flatten row by row within each sector block
    return G_reshaped.flatten(order="C")


# Vector → Block (Reshaping) Operations


def reshape_E_block(E_flat: Array, N_K: int, k_n: int) -> Array:
    """
    Reshape a flattened E vector back into block form.

    Args:
        E_flat: Flattened array from flatten_E_block
        N_K: Number of undisaggregated sectors
        k_n: Number of subsectors for the disaggregated sector

    Returns:
        Array of shape (N_K, k_n)
    """
    if len(E_flat) != N_K * k_n:
        raise ValueError(f"E_flat should have length {N_K * k_n}, got {len(E_flat)}")

    # Reshape row by row (C-order)
    return E_flat.reshape((N_K, k_n), order="C")


def reshape_F_block(F_flat: Array, N_K: int, k_n: int) -> Array:
    """
    Reshape a flattened F vector back into block form.

    Args:
        F_flat: Flattened array from flatten_F_block
        N_K: Number of undisaggregated sectors
        k_n: Number of subsectors for the disaggregated sector

    Returns:
        Array of shape (k_n, N_K)
    """
    if len(F_flat) != N_K * k_n:
        raise ValueError(f"F_flat should have length {N_K * k_n}, got {len(F_flat)}")

    # Reshape column by column (F-order)
    return F_flat.reshape((k_n, N_K), order="F")


def reshape_G_block(G_flat: Array, k_n: int, k_l: list[int]) -> Array:
    """
    Reshape a flattened G vector back into block form.

    Args:
        G_flat: Flattened array from flatten_G_block
        k_n: Number of subsectors for sector n
        k_l: List of numbers of subsectors for each sector l

    Returns:
        Array of shape (k_n, sum(k_l))
    """
    expected_cols = sum(k_l)
    expected_len = k_n * expected_cols
    if len(G_flat) != expected_len:
        raise ValueError(f"G_flat should have length {expected_len}, got {len(G_flat)}")

    # Reshape row by row within each sector block
    return G_flat.reshape((k_n, expected_cols), order="C")


# Block Extraction Operations


def extract_E_block(
    reader: ICIOReader,
    undisaggregated_sectors: list[str],
    disaggregated_sectors: list[str],
) -> pd.DataFrame:
    """Extract the E block from a disaggregated table.

    This block contains flows FROM all undisaggregated sectors (in all countries)
    TO the disaggregated sector in the target country.
    """
    tech_coef = reader.technical_coefficients

    # Create indices for undisaggregated sectors (all countries)
    row_idx = pd.MultiIndex.from_product(
        [reader.countries, undisaggregated_sectors], names=["CountryInd", "industryInd"]
    )
    # For disaggregated sectors, we only want the target country
    col_idx = pd.MultiIndex.from_product(
        [["USA"], disaggregated_sectors], names=["CountryInd", "industryInd"]
    )

    # Extract block
    E = tech_coef.loc[row_idx, col_idx]
    logger.debug(f"Extracted E block with shape {E.shape}")
    return E


def extract_F_block(
    reader: ICIOReader,
    undisaggregated_sectors: list[str],
    disaggregated_sectors: list[str],
) -> pd.DataFrame:
    """Extract the F block from a disaggregated table.

    This block contains flows FROM the disaggregated sector in the target country
    TO all undisaggregated sectors (in all countries).
    """
    tech_coef = reader.technical_coefficients

    # For disaggregated sectors, we only want the target country
    row_idx = pd.MultiIndex.from_product(
        [["USA"], disaggregated_sectors], names=["CountryInd", "industryInd"]
    )
    # Create indices for undisaggregated sectors (all countries)
    col_idx = pd.MultiIndex.from_product(
        [reader.countries, undisaggregated_sectors], names=["CountryInd", "industryInd"]
    )

    # Extract block
    F = tech_coef.loc[row_idx, col_idx]
    logger.debug(f"Extracted F block with shape {F.shape}")
    return F


def extract_G_block(
    reader: ICIOReader,
    sector_n: list[str],
    sectors_l: list[list[str]],
) -> pd.DataFrame:
    """Extract the G block from a disaggregated table.

    This block contains flows FROM the disaggregated sector n's subsectors
    TO all subsectors of all sectors ℓ. Each G^{nℓ} represents flows from
    sector n's subsectors to sector ℓ's subsectors, summed across all countries.

    Args:
        reader: ICIOReader with the disaggregated data
        sector_n: List of subsectors for sector n being disaggregated
        sectors_l: List of lists of subsectors for each sector l that n interacts with

    Returns:
        DataFrame with flows between disaggregated sectors, summed across countries
    """
    tech_coef = reader.technical_coefficients

    # For row indices, we want the target country's subsectors
    row_idx = pd.MultiIndex.from_product([["USA"], sector_n], names=["CountryInd", "industryInd"])

    # For column indices, we want all countries' subsectors for each sector l
    col_sectors = [s for sector in sectors_l for s in sector]
    col_idx = pd.MultiIndex.from_product(
        [reader.countries, col_sectors], names=["CountryInd", "industryInd"]
    )

    # Extract block and sum across destination countries
    G = tech_coef.loc[row_idx, col_idx].T.groupby("industryInd").sum().T
    logger.debug(f"Extracted G block with shape {G.shape}")
    logger.debug(f"G block sums flows across {len(reader.countries)} countries")
    return G


def blocks_to_vector(
    reader: ICIOReader,
    sectors_to_disaggregate: list[str],
) -> Array:
    """
    Convert technical coefficient blocks to a flattened vector.

    This is the main function for converting from block form to vector form.
    It extracts all blocks (E, F, G) and final demand (b), flattens them,
    and concatenates them into the X^n vector.

    Args:
        reader: ICIOReader with the disaggregated data
        sectors_to_disaggregate: List of sector codes being disaggregated

    Returns:
        The X^n vector containing [E^n, F^n, G^n, b^n] in order
    """
    # Get all sectors
    all_sectors = reader.industries
    undisaggregated = [s for s in all_sectors if s not in sectors_to_disaggregate]
    logger.info(f"Converting blocks to vector for sectors: {sectors_to_disaggregate}")
    logger.info(f"Undisaggregated sectors: {undisaggregated}")
    logger.info(f"Countries in reader: {reader.countries}")

    # Extract blocks
    E = extract_E_block(reader, undisaggregated, sectors_to_disaggregate)
    F = extract_F_block(reader, undisaggregated, sectors_to_disaggregate)
    G = extract_G_block(reader, sectors_to_disaggregate, [sectors_to_disaggregate])

    # Get dimensions
    N_K = len(undisaggregated) * len(reader.countries)
    k_n = len(sectors_to_disaggregate)
    k_l = [k_n]  # For this case, only one sector is being disaggregated
    logger.info(f"Dimensions: N_K={N_K}, k_n={k_n}, k_l={k_l}")
    logger.info(f"Block shapes: E={E.shape}, F={F.shape}, G={G.shape}")

    # Flatten blocks
    E_flat = flatten_E_block(E, N_K, k_n)
    F_flat = flatten_F_block(F, N_K, k_n)
    G_flat = flatten_G_block(G, k_n, k_l)

    # Get total final demand across all countries for the disaggregated sectors
    fd_idx = pd.MultiIndex.from_product(
        [reader.countries, sectors_to_disaggregate], names=["CountryInd", "industryInd"]
    )
    logger.info(f"Final demand index: {fd_idx}")
    b = reader.final_demand.loc[fd_idx].groupby(level="industryInd").sum().values
    logger.info(f"Final demand values (summed across countries): {b}")
    logger.info(f"Final demand shape: {b.shape}")

    # Concatenate all components
    X_n = np.concatenate([E_flat, F_flat, G_flat, b])

    logger.info(f"Created X^n vector of length {len(X_n)}")
    logger.info(f"Components: E={len(E_flat)}, F={len(F_flat)}, G={len(G_flat)}, b={len(b)}")

    return X_n


def vector_to_blocks(
    X_n: Array,
    N_K: int,
    k_n: int,
    k_l: list[int],
) -> tuple[Array, Array, Array, Array]:
    """
    Convert a flattened vector back to technical coefficient blocks.

    This is the main function for converting from vector form to block form.
    It takes the X^n vector and splits it back into its component blocks
    (E, F, G) and final demand (b).

    Args:
        X_n: The flattened vector from blocks_to_vector
        N_K: Number of undisaggregated sectors
        k_n: Number of subsectors for sector n
        k_l: List of numbers of subsectors for each sector l

    Returns:
        Tuple of (E, F, G, b) arrays in their original block forms
    """
    # Calculate component sizes
    E_size = N_K * k_n
    F_size = N_K * k_n
    G_size = k_n * sum(k_l)
    b_size = k_n
    expected_len = E_size + F_size + G_size + b_size

    logger.info(f"Converting vector of length {len(X_n)} back to blocks")
    logger.info(f"Input dimensions: N_K={N_K}, k_n={k_n}, k_l={k_l}")
    logger.info(f"Expected component sizes: E={E_size}, F={F_size}, G={G_size}, b={b_size}")
    logger.info(f"Total expected length: {expected_len}")

    # Verify total length
    if len(X_n) != expected_len:
        raise ValueError(f"X_n should have length {expected_len}, got {len(X_n)}")

    # Split vector into components
    E_flat = X_n[:E_size]
    F_flat = X_n[E_size : E_size + F_size]
    G_flat = X_n[E_size + F_size : E_size + F_size + G_size]
    b = X_n[E_size + F_size + G_size :]

    # Reshape blocks
    E = reshape_E_block(E_flat, N_K, k_n)
    F = reshape_F_block(F_flat, N_K, k_n)
    G = reshape_G_block(G_flat, k_n, k_l)

    logger.info(f"Reshaped blocks: E={E.shape}, F={F.shape}, G={G.shape}, b={b.shape}")
    return E, F, G, b
