"""Module for handling block structure in disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd

# Type alias for sector identifiers - can be just str for single-region tables
# or tuple[str, str] for multi-region tables (country, sector)
SectorId: TypeAlias = str | tuple[str, str]

logger = logging.getLogger(__name__)


class SectorInfo(NamedTuple):
    """
    Information about a sector in the disaggregation problem.

    For single-region IO tables, sector_id is just the sector code (e.g., "A01").
    For multi-region tables like ICIO, sector_id is a (country, sector) tuple.
    """

    index: int  # The index n in the mathematical formulation
    sector_id: SectorId  # Sector identifier (code or country-code pair)
    name: str  # Human readable name
    k: int  # Number of subsectors this splits into

    @property
    def is_multi_region(self) -> bool:
        """Whether this is a multi-region (e.g., ICIO) sector."""
        return isinstance(self.sector_id, tuple)

    @property
    def country(self) -> str | None:
        """Get the country code if this is a multi-region sector."""
        return self.sector_id[0] if self.is_multi_region else None

    @property
    def sector(self) -> str:
        """Get the sector code (regardless of whether multi-region or not)."""
        return self.sector_id[1] if self.is_multi_region else self.sector_id


@dataclass
class DisaggregationBlocks:
    """
    Class to handle the block structure of a disaggregation problem.

    This class maintains the mapping between sector indices (n) and their
    codes/names, and provides methods to access the various blocks (A₀, B^n, etc.)
    in the mathematical formulation.

    The class is generic and works with both:
    - Single-region IO tables (sectors identified by sector code)
    - Multi-region IO tables (sectors identified by country-sector pairs)

    Attributes:
        sectors: List of sectors being disaggregated, with their indices
        N: Total number of sectors
        K: Number of sectors being disaggregated
        M: Total number of subsectors after disaggregation
        reordered_matrix: The reordered technical coefficients matrix
    """

    sectors: list[SectorInfo]
    reordered_matrix: pd.DataFrame

    @property
    def is_multi_region(self) -> bool:
        """Whether this is a multi-region (e.g., ICIO) disaggregation."""
        return self.sectors[0].is_multi_region if self.sectors else False

    @property
    def N(self) -> int:
        """Total number of sectors."""
        return len(self.reordered_matrix)

    @property
    def K(self) -> int:
        """Number of sectors being disaggregated."""
        # Each entry in self.sectors represents a unique disaggregation
        # For multi-region tables, (USA, A03) and (CAN, A03) are separate disaggregations
        return len(self.sectors)

    @property
    def M(self) -> int:
        """Total number of subsectors after disaggregation."""
        return sum(s.k for s in self.sectors)

    def get_A0(self) -> np.ndarray:
        """Get the A₀ block (undisaggregated sectors)."""
        N_K = self.N - self.K
        return self.reordered_matrix.iloc[:N_K, :N_K].values

    def get_B(self, n: int) -> np.ndarray:
        """
        Get the B^n block (undisaggregated to sector n).

        Args:
            n: Index of the sector (1-based as in math notation)
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.K}]")

        N_K = self.N - self.K
        # Find the column index for sector n
        sector_id = self.sectors[n - 1].sector_id
        try:
            col_idx = self.reordered_matrix.columns.get_loc(sector_id)
        except KeyError:
            logger.error(f"Sector {sector_id} not found in matrix columns")
            raise

        logger.debug(f"Getting B block for sector {n}")
        logger.debug(f"Matrix shape: {self.reordered_matrix.shape}")
        logger.debug(f"N_K={N_K}, col_idx={col_idx}")
        logger.debug(f"Matrix columns: {self.reordered_matrix.columns.tolist()}")
        logger.debug(f"Sector being processed: {self.sectors[n-1]}")

        # Get the block and repeat it for each subsector
        B = self.reordered_matrix.iloc[:N_K, col_idx : col_idx + 1].values
        logger.debug(f"Initial B shape: {B.shape}")

        # Always repeat to match the number of subsectors
        B = np.repeat(B, self.sectors[n - 1].k, axis=1)
        logger.debug(f"Final B shape after repeat: {B.shape}")

        return B

    def get_C(self, n: int) -> np.ndarray:
        """
        Get the C^n block (sector n to undisaggregated).

        Args:
            n: Index of the sector (1-based as in math notation)
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.K}]")

        N_K = self.N - self.K
        # Find the row index for sector n
        sector_id = self.sectors[n - 1].sector_id
        try:
            row_idx = self.reordered_matrix.index.get_loc(sector_id)
        except KeyError:
            logger.error(f"Sector {sector_id} not found in matrix index")
            raise

        # Get the block and repeat it for each subsector
        C = self.reordered_matrix.iloc[row_idx : row_idx + 1, :N_K].values
        C = np.repeat(C, self.sectors[n - 1].k, axis=0)
        return C

    def get_D(self, n: int, l: int) -> np.ndarray:
        """
        Get the D^{nℓ} block (sector n to sector ℓ).

        Args:
            n: Index of the first sector (1-based as in math notation)
            l: Index of the second sector (1-based as in math notation)
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"First sector index {n} out of range [1, {self.K}]")
        if not 1 <= l <= self.K:
            raise ValueError(f"Second sector index {l} out of range [1, {self.K}]")

        # Find the row and column indices
        sector_n = self.sectors[n - 1].sector_id
        sector_l = self.sectors[l - 1].sector_id
        try:
            row_idx = self.reordered_matrix.index.get_loc(sector_n)
            col_idx = self.reordered_matrix.columns.get_loc(sector_l)
        except KeyError as e:
            logger.error(f"Sector not found in matrix: {e}")
            raise

        # Get the block and repeat it for both dimensions
        D = self.reordered_matrix.iloc[row_idx : row_idx + 1, col_idx : col_idx + 1].values
        D = np.repeat(D, self.sectors[n - 1].k, axis=0)  # Repeat for rows
        D = np.repeat(D, self.sectors[l - 1].k, axis=1)  # Repeat for columns
        return D

    def get_sector_info(self, n: int) -> SectorInfo:
        """
        Get information about a sector by its index.

        Args:
            n: Index of the sector (1-based as in math notation)
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.K}]")
        return self.sectors[n - 1]

    @classmethod
    def from_technical_coefficients(
        cls,
        tech_coef: pd.DataFrame,
        sectors_to_disaggregate: list[tuple[SectorId, str, int]],  # (id, name, k)
    ) -> "DisaggregationBlocks":
        """
        Create DisaggregationBlocks from a technical coefficients matrix.

        Works with both single-region and multi-region IO tables:
        - For single-region: sector_id is just the sector code (str)
        - For multi-region: sector_id is a (country, sector) tuple

        Args:
            tech_coef: Technical coefficients matrix
            sectors_to_disaggregate: List of (sector_id, name, k) tuples for sectors
                to disaggregate, where k is the number of subsectors to split into.
                The order of sectors in this list determines their indices in the
                mathematical formulation.

        Returns:
            DisaggregationBlocks instance with reordered matrix and sector info
        """
        # Create sector info objects
        sectors = [
            SectorInfo(i + 1, sector_id, name, k)
            for i, (sector_id, name, k) in enumerate(sectors_to_disaggregate)
        ]

        # Determine if we're working with a multi-region table
        is_multi_region = sectors[0].is_multi_region if sectors else False

        # Get set of sector IDs to disaggregate
        sector_ids = {s.sector_id for s in sectors}

        logger.debug(
            f"Creating blocks from technical coefficients matrix of shape {tech_coef.shape}"
        )
        logger.debug(f"Sectors to disaggregate: {sector_ids}")
        logger.debug(f"Is multi-region: {is_multi_region}")

        # Split sectors into disaggregated and undisaggregated
        if is_multi_region:
            # For multi-region tables, we need to handle country-sector pairs
            # First get all sectors that aren't being disaggregated
            # For each sector_id in sector_ids, we need to find all country-sector pairs
            # that match the sector code
            sector_codes = {s.sector for s in sectors}
            undisaggregated = [
                idx
                for idx in tech_coef.index
                if isinstance(idx, tuple) and idx[1] not in sector_codes
            ]

            # Sort undisaggregated sectors by country, then sector
            def sort_key(x):
                country, sector = x
                return (country, sector)

            logger.debug("Before sorting undisaggregated:")
            logger.debug(f"Undisaggregated: {undisaggregated}")

            undisaggregated.sort(key=sort_key)

            logger.debug("After sorting undisaggregated:")
            logger.debug(f"Undisaggregated: {undisaggregated}")

            # For disaggregated sectors, we need to find all country-sector pairs
            # that match each sector code, in the order they appear in sectors
            disaggregated = []
            seen_pairs = set()  # Track which pairs we've already seen

            # Add sectors in the order they appear in sectors list
            for sector in sectors:
                # Find all pairs that match this sector's code
                matching_pairs = [
                    idx
                    for idx in tech_coef.index
                    if isinstance(idx, tuple) and idx[1] == sector.sector
                ]
                # Add the sector's own pair first
                if sector.sector_id not in seen_pairs and sector.sector_id in matching_pairs:
                    disaggregated.append(sector.sector_id)
                    seen_pairs.add(sector.sector_id)
                # Then add any remaining pairs for this sector
                for pair in sorted(matching_pairs):
                    if pair not in seen_pairs:
                        disaggregated.append(pair)
                        seen_pairs.add(pair)

            logger.debug(f"Disaggregated (keeping original order): {disaggregated}")

        else:
            # For single-region tables, index is just sector codes
            undisaggregated = sorted((idx for idx in tech_coef.index if idx not in sector_ids))
            disaggregated = [s.sector_id for s in sectors]

        # Create new order and reindex
        new_order = undisaggregated + disaggregated
        logger.debug(f"Final order: {new_order}")

        reordered = tech_coef.reindex(index=new_order, columns=new_order)
        return cls(sectors=sectors, reordered_matrix=reordered)
