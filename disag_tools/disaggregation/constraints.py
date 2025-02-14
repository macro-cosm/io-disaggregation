"""Module for generating constraint matrices in disaggregation problems."""

import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Type alias for float arrays
Array: TypeAlias = NDArray[np.float64]

logger = logging.getLogger(__name__)


def generate_M1_block(N_K: int, k: int, weights: Array) -> Array:
    """
    Generate the M₁ block for aggregation constraints on B blocks.

    This block ensures that flows from undisaggregated sectors to subsectors
    respect the weights w^n. For each undisaggregated sector i, we have:
    E_{ij} = B_i w_j^n

    Args:
        N_K: Number of undisaggregated sectors
        k: Number of subsectors for the disaggregated sector
        weights: Array of length k containing the weights w_j^n

    Returns:
        Array of shape (N_K, N_K * k) representing M₁
        Each row i of M₁ contains the weights in positions (i*k) to ((i+1)*k-1)
    """
    if len(weights) != k:
        raise ValueError(f"Expected {k} weights, got {len(weights)}")

    # Initialize zero matrix of shape (N_K, N_K * k)
    M1 = np.zeros((N_K, N_K * k))

    # For each undisaggregated sector i
    for i in range(N_K):
        # Place weights in positions (i*k) to ((i+1)*k-1) of row i
        M1[i, i * k : (i + 1) * k] = weights

    logger.debug(f"Generated M1 block of shape {M1.shape}")
    logger.debug(f"M1 block has {np.count_nonzero(M1)} nonzero elements")
    logger.debug(f"M1 block row sums: {M1.sum(axis=1)}")

    return M1


def generate_M2_block(N_K: int, k: int) -> Array:
    """Generate the M2 block for aggregation constraints.

    The M2 block ensures that flows from subsectors to undisaggregated sectors
    sum to the original flows. For each undisaggregated sector i, it contains
    k ones in positions (i*k) to ((i+1)*k), with zeros elsewhere.

    Args:
        N_K: Number of undisaggregated sectors
        k: Number of subsectors for the disaggregated sector

    Returns:
        NDArray: M2 block matrix of shape (N_K, N_K * k)
    """
    # Initialize zero matrix
    M2 = np.zeros((N_K, N_K * k))

    # For each undisaggregated sector i
    for i in range(N_K):
        # Fill k ones in positions (i*k) to ((i+1)*k)
        M2[i, i * k : (i + 1) * k] = 1.0

    logger.debug(f"Generated M2 block with shape {M2.shape}")
    logger.debug(f"M2 block nonzero elements: {np.count_nonzero(M2)}")
    logger.debug(f"M2 block row sums: {M2.sum(axis=1)}")
    return M2


def generate_M3_block(k_n: int, weights_l: list[Array]) -> Array:
    """Generate the M3 block for cross-sector constraints.

    The M3 block ensures that flows between subsectors respect both sectors' weights.
    For sector n being disaggregated, M3^n is a block diagonal matrix where each block
    M3^{nℓ} is constructed by repeating the weights vector w^ℓ k_n times.

    Args:
        k_n: Number of subsectors for sector n (the sector being disaggregated)
        weights_l: List of weight arrays for each sector ℓ that n interacts with.
                  Each array contains the weights w_j^ℓ for sector ℓ's subsectors.

    Returns:
        Array: M3 block matrix of shape (K, sum(k_ℓ * k_n) for all sectors ℓ)
        where K is the number of sectors being disaggregated
    """
    K = len(weights_l)  # Number of sectors being disaggregated

    # For each sector ℓ, create its row in the block matrix
    rows = []
    for l, w_l in enumerate(weights_l):
        k_l = len(w_l)  # Number of subsectors for sector ℓ

        # Create blocks for this row:
        # - Zero blocks for sectors before ℓ
        # - The M3^{nℓ} block with repeated weights
        # - Zero blocks for sectors after ℓ
        row_blocks = []

        # Add zero blocks for sectors before ℓ
        for j in range(l):
            k_j = len(weights_l[j])
            row_blocks.append(np.zeros((1, k_j * k_n)))

        # Add the M3^{nℓ} block: stack the weights k_n times horizontally
        M3_nl = np.tile(w_l, k_n).reshape(1, -1)
        row_blocks.append(M3_nl)

        # Add zero blocks for sectors after ℓ
        for j in range(l + 1, K):
            k_j = len(weights_l[j])
            row_blocks.append(np.zeros((1, k_j * k_n)))

        # Combine all blocks for this row
        rows.append(np.hstack(row_blocks))

    # Stack all rows vertically
    M3 = np.vstack(rows)

    logger.debug(f"Generated M3 block with shape {M3.shape}")
    return M3


def generate_M4_block(k_n: int, weights_l: list[Array], z_l: Array, z_n: float) -> Array:
    """Generate the M4 block for final demand consistency constraints.

    The M4 block ensures that flows between subsectors are consistent with final demands.
    For sector n being disaggregated, M4^n is constructed by scaling each sector ℓ's weights
    by the ratio z_ℓ/z_n and repeating this k_n times.

    Args:
        k_n: Number of subsectors for sector n (the sector being disaggregated)
        weights_l: List of weight arrays for each sector ℓ that n interacts with.
                  Each array contains the weights w_j^ℓ for sector ℓ's subsectors.
        z_l: Array of outputs z_ℓ for each sector ℓ
        z_n: Output z_n of sector n being disaggregated

    Returns:
        Array: M4 block matrix of shape (k_n, sum(k_ℓ * k_n) for all sectors ℓ)
    """
    K = len(weights_l)  # Number of sectors being disaggregated

    if len(z_l) != K:
        raise ValueError(f"Expected {K} sector outputs, got {len(z_l)}")

    # For each row i in k_n rows, create blocks for all sectors ℓ
    rows = []
    for i in range(k_n):
        row_blocks = []

        # For each sector ℓ
        for l, (w_l, z_ell) in enumerate(zip(weights_l, z_l)):
            k_l = len(w_l)  # Number of subsectors for sector ℓ

            # Scale weights by z_ℓ/z_n
            scaled_weights = w_l * (z_ell / z_n)

            # Create the block for this sector:
            # - Zeros for positions before i
            # - Scaled weights at position i
            # - Zeros for positions after i
            block = np.zeros((1, k_l * k_n))
            block[0, i * k_l : (i + 1) * k_l] = scaled_weights
            row_blocks.append(block)

        # Combine all blocks for this row
        rows.append(np.hstack(row_blocks))

    # Stack all rows vertically
    M4 = np.vstack(rows)

    logger.debug(f"Generated M4 block with shape {M4.shape}")
    return M4


def generate_M5_block(k_n: int, x: Array, z_n: float) -> Array:
    """Generate the M₅ block for final demand consistency constraints.

    The M₅ block scales undisaggregated sector outputs by 1/z_n. According to eq:m5
    in the disaggregation plan, it has the form:
    M₅^n = [x₁/z_n I_{k_n} | x₂/z_n I_{k_n} | ... | x_{N-K}/z_n I_{k_n}]
    where I_{k_n} is the k_n × k_n identity matrix.

    Args:
        k_n: Number of subsectors for sector n (being disaggregated).
        x: Array of outputs for undisaggregated sectors.
        z_n: Total output of sector n.

    Returns:
        Array: The M₅ block matrix of shape (k_n × k_n*N_K).

    Raises:
        ValueError: If no undisaggregated sector outputs are provided.
    """
    N_K = len(x)
    if N_K == 0:
        raise ValueError("At least one undisaggregated sector output is required")

    # Initialize zero matrix of shape (k_n × k_n*N_K)
    M5 = np.zeros((k_n, k_n * N_K))

    # For each undisaggregated sector j
    for j in range(N_K):
        # Create x_j/z_n times identity matrix
        block = (x[j] / z_n) * np.eye(k_n)
        # Place it in the correct position
        M5[:, j * k_n : (j + 1) * k_n] = block

    logger.debug(f"Generated M5 block with shape {M5.shape}")
    logger.debug(f"M5 block has {np.count_nonzero(M5)} nonzero elements")
    logger.debug(f"M5 block row sums: {M5.sum(axis=1)}")
    return M5


def generate_M_n_matrix(
    k_n: int,
    N_K: int,
    n: int,
    weights_n: Array,
    weights_l: list[Array],
    x: Array,
    z_l: Array,
) -> Array:
    """Generate the complete M^n constraint matrix for sector n.

    This function combines all constraint blocks (M₁, M₂, M₃, M₄, M₅) into a single
    block matrix that enforces all disaggregation constraints for sector n:

    M^n = [
        [ M₁^n    0       0         0      ]
        [ 0       M₂^n    0         0      ]
        [ 0       0       M₃^n      0      ]
        [ 0       M₅^n    M₄^n      1_{k_n} ]
    ]

    Args:
        k_n: Number of subsectors for sector n (being disaggregated).
        N_K: Number of undisaggregated sectors.
        n: 1-based index of sector n being disaggregated.
        weights_n: Array of length k_n containing weights for sector n's subsectors.
        weights_l: List of weight arrays for each sector ℓ that n interacts with.
        x: Array of outputs for undisaggregated sectors.
        z_l: Array of outputs for all disaggregated sectors.

    Returns:
        Array: The complete M^n constraint matrix.

    Raises:
        ValueError: If sector index n is out of range (not between 1 and len(z_l)).
    """
    # Validate sector index
    if n < 1 or n > len(z_l):
        raise ValueError(f"Sector index {n} out of range")

    logger.info(f"Generating M^n matrix for sector {n} with {k_n} subsectors")
    logger.info(f"System has {N_K} undisaggregated sectors and {len(weights_l)} disaggregated sectors")

    # Get z_n from z_l using 1-based indexing
    z_n = z_l[n - 1]
    logger.debug(f"Output z_n for sector {n}: {z_n}")

    # Calculate dimensions
    K = len(weights_l)  # Number of sectors being disaggregated
    total_cols_G = sum(len(w_l) * k_n for w_l in weights_l)

    logger.debug(f"Weights for sector {n}: {weights_n}")
    logger.debug(f"Weights for interacting sectors: {[w.tolist() for w in weights_l]}")
    logger.debug(f"Undisaggregated sector outputs: {x}")
    logger.debug(f"Disaggregated sector outputs: {z_l}")

    # Generate all component blocks
    logger.info("Generating component blocks...")

    M1 = generate_M1_block(N_K, k_n, weights_n)
    logger.debug(f"M₁ block shape: {M1.shape}, nonzero elements: {np.count_nonzero(M1)}")
    logger.debug(f"M₁ block row sums: {M1.sum(axis=1)}")

    M2 = generate_M2_block(N_K, k_n)
    logger.debug(f"M₂ block shape: {M2.shape}, nonzero elements: {np.count_nonzero(M2)}")
    logger.debug(f"M₂ block row sums: {M2.sum(axis=1)}")

    M3 = generate_M3_block(k_n, weights_l)
    logger.debug(f"M₃ block shape: {M3.shape}, nonzero elements: {np.count_nonzero(M3)}")
    logger.debug(f"M₃ block row sums: {M3.sum(axis=1)}")

    # Extract outputs for sectors in weights_l
    z_l_subset = []
    for i in range(len(z_l)):
        if i + 1 != n:  # Convert to 1-based indexing
            z_l_subset.append(z_l[i])
    z_l_subset = np.array(z_l_subset)
    if len(z_l_subset) < K:
        # If we don't have enough outputs, pad with the last output
        z_l_subset = np.pad(z_l_subset, (0, K - len(z_l_subset)), mode="edge")
    else:
        # Take only the first K outputs
        z_l_subset = z_l_subset[:K]

    logger.debug(f"Subset of sector outputs for M₄: {z_l_subset}")

    M4 = generate_M4_block(k_n, weights_l, z_l_subset, z_n)
    logger.debug(f"M₄ block shape: {M4.shape}, nonzero elements: {np.count_nonzero(M4)}")
    logger.debug(f"M₄ block row sums: {M4.sum(axis=1)}")

    M5 = generate_M5_block(k_n, x, z_n)
    logger.debug(f"M₅ block shape: {M5.shape}, nonzero elements: {np.count_nonzero(M5)}")
    logger.debug(f"M₅ block row sums: {M5.sum(axis=1)}")

    I_kn = np.eye(k_n)
    logger.debug(f"Identity block shape: {I_kn.shape}")

    # Create zero blocks for padding
    logger.info("Creating zero blocks for padding...")

    # Calculate total columns needed
    total_cols = 2 * N_K * k_n + total_cols_G + k_n
    logger.debug(f"Total columns needed: {total_cols}")

    # First row: [M1 | 0]
    Z1 = np.zeros((N_K, total_cols - M1.shape[1]))  # Zeros after M1
    row1 = np.hstack([M1, Z1])
    logger.debug(f"Row 1 shape: {row1.shape}")

    # Second row: [0 | M2 | 0]
    Z2a = np.zeros((N_K, N_K * k_n))  # Zeros before M2
    Z2b = np.zeros((N_K, total_cols - Z2a.shape[1] - M2.shape[1]))  # Zeros after M2
    row2 = np.hstack([Z2a, M2, Z2b])
    logger.debug(f"Row 2 shape: {row2.shape}")

    # Third row: [0 | 0 | M3 | 0]
    Z3a = np.zeros((K, 2 * N_K * k_n))  # Zeros before M3
    Z3b = np.zeros((K, total_cols - Z3a.shape[1] - M3.shape[1]))  # Zeros after M3
    row3 = np.hstack([Z3a, M3, Z3b])
    logger.debug(f"Row 3 shape: {row3.shape}")

    # Fourth row: [0 | M5 | M4 | I]
    Z4a = np.zeros((k_n, N_K * k_n))  # Zeros before M5
    Z4b = np.zeros((k_n, total_cols - Z4a.shape[1] - M5.shape[1] - M4.shape[1] - I_kn.shape[1]))  # Zeros after I
    row4 = np.hstack([Z4a, M5, M4, I_kn, Z4b])
    logger.debug(f"Row 4 shape: {row4.shape}")

    # Stack all rows vertically
    M_n = np.vstack([row1, row2, row3, row4])

    logger.info(f"Generated M^n matrix with shape {M_n.shape}")
    logger.debug(f"M^n matrix nonzero elements: {np.count_nonzero(M_n)}")
    logger.debug(f"M^n matrix row sums: {M_n.sum(axis=1)}")
    logger.debug(f"M^n matrix column sums: {M_n.sum(axis=0)}")

    return M_n
