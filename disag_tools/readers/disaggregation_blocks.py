"""Module for handling block structure in disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from disag_tools.readers.icio_reader import ICIOReader

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
        disaggregated_sector_names: List of sector codes/names being disaggregated
        non_disaggregated_sector_names: List of sector codes/names not being disaggregated
        reordered_matrix: The reordered technical coefficients matrix
    """

    sectors: list[SectorInfo]
    disaggregated_sector_names: list[str | tuple[str, str]]
    non_disaggregated_sector_names: list[str | tuple[str, str]]
    reordered_matrix: pd.DataFrame

    @classmethod
    def from_technical_coefficients(
        cls,
        tech_coef: pd.DataFrame,
        sectors_info: list[tuple[SectorId, str, int]],
    ) -> "DisaggregationBlocks":
        """Create DisaggregationBlocks from a technical coefficients matrix.

        Args:
            tech_coef: Technical coefficients matrix
            sectors_info: List of tuples (sector_id, name, k) where:
                - sector_id is a sector identifier (str for single region, tuple for multi)
                - name is a human-readable name
                - k is the number of subsectors to split into

        Returns:
            DisaggregationBlocks instance with reordered matrix and sector info
        """
        # Convert sectors_info to list of SectorInfo objects with 1-based indices
        sectors = [
            SectorInfo(index=i + 1, sector_id=s_id, name=name, k=k)
            for i, (s_id, name, k) in enumerate(sectors_info)
        ]

        if isinstance(sectors_info[0][0], tuple):
            k_dict = {s.sector_id[1]: s.k for s in sectors}
            name_dict = {s.sector_id[1]: s.name for s in sectors}
        else:
            k_dict = {s.sector_id: s.k for s in sectors}
            name_dict = {s.sector_id: s.name for s in sectors}

        sector_ids = [s.sector_id for s in sectors]

        # Check if we're dealing with a multi-region table
        is_multi_region = isinstance(tech_coef.index[0], tuple)
        logger.debug(f"Processing {'multi-region' if is_multi_region else 'single-region'} table")

        if is_multi_region:
            # For multi-region tables, we need to handle country-sector pairs
            # First get all sectors that aren't being disaggregated
            sector_codes = {s.sector for s in sectors}
            special_rows = ["OUT", "OUTPUT", "VA", "TLS"]

            # Split into undisaggregated and disaggregated
            undisaggregated = []
            for idx in tech_coef.index:
                if not isinstance(idx, tuple):
                    continue
                country, sector = idx
                # Skip special rows and columns
                if country in special_rows or sector in special_rows:
                    continue
                if sector not in sector_codes:
                    undisaggregated.append(idx)

            # Sort undisaggregated sectors by country, then sector
            def sort_key(x):
                country, sector = x
                return (country, sector)

            undisaggregated.sort(key=sort_key)

            # For disaggregated sectors, we need to find all country-sector pairs
            # that match each sector code, in the order they appear in sectors
            disaggregated = []
            seen_pairs = set()

            # Add sectors in the order they appear in sectors list
            for sector in sectors:
                # Find all pairs that match this sector's code
                matching_pairs = [
                    idx
                    for idx in tech_coef.index
                    if isinstance(idx, tuple)
                    and idx[0] not in special_rows
                    and idx[1] == sector.sector
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

        else:
            # For single-region tables, index is just sector codes
            special_rows = ["OUT", "OUTPUT", "VA", "TLS"]
            undisaggregated = sorted(
                idx for idx in tech_coef.index if idx not in sector_ids and idx not in special_rows
            )
            disaggregated = [s.sector_id for s in sectors]

        disaggregated = sorted(disaggregated)
        # Create new order and reindex
        new_order = undisaggregated + disaggregated
        logger.debug(f"Final order: {new_order}")

        if isinstance(sectors_info[0][0], tuple):
            sectors = [
                SectorInfo(index=i + 1, sector_id=s_id, name=name_dict[s_id[1]], k=k_dict[s_id[1]])
                for i, s_id in enumerate(disaggregated)
            ]
        else:
            sectors = [
                SectorInfo(index=i + 1, sector_id=s_id, name=name_dict[s_id], k=k_dict[s_id])
                for i, s_id in enumerate(disaggregated)
            ]

        # Filter out special rows/columns from tech_coef before reindexing
        filtered_tech_coef = tech_coef.loc[new_order, new_order]
        return cls(
            sectors=sectors,
            reordered_matrix=filtered_tech_coef,
            disaggregated_sector_names=disaggregated,
            non_disaggregated_sector_names=undisaggregated,
        )

    @property
    def is_multi_region(self) -> bool:
        """Whether this is a multi-region (e.g., ICIO) disaggregation."""
        return self.sectors[0].is_multi_region if self.sectors else False

    @property
    def N(self) -> int:
        """Total number of sectors (excluding special rows like OUT)."""
        # Count rows that aren't special rows
        special_rows = ["OUT", "OUTPUT", "VA", "TLS"]
        if self.is_multi_region:
            # For multi-region, exclude rows where the country is a special row
            return sum(
                1
                for idx in self.reordered_matrix.index
                if isinstance(idx, tuple) and idx[0] not in special_rows
            )
        else:
            # For single-region, exclude rows that are special rows
            return sum(1 for idx in self.reordered_matrix.index if idx not in special_rows)

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
        logger.debug(f"Sector being processed: {self.sectors[n - 1]}")

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


class DisaggregatedBlocks(DisaggregationBlocks):
    """Class to handle the block structure of a disaggregated problem.

    This class maintains the mapping between aggregated and disaggregated sectors,
    and provides methods to access the various blocks (E, F, G) in the
    mathematical formulation.

    For example, if sector A is disaggregated into A01 and A03:
    - The input technical coefficients matrix will have A01 and A03 (not A)
    - We specify the mapping A -> [A01, A03]
    - The class handles creating this mapping for all countries in the data

    Attributes:
        sector_mapping: Dictionary mapping aggregated sectors to their disaggregated subsectors
            e.g. {("USA", "A"): [("USA", "A01"), ("USA", "A03")]}
        aggregated_sectors_list: List of aggregated sectors in order
    """

    sector_mapping: dict[tuple[str, str], list[tuple[str, str]]]
    aggregated_sectors_list: list[tuple[str, str]]
    m: int

    def __init__(
        self,
        sectors: list[SectorInfo],
        reordered_matrix: pd.DataFrame,
        disaggregated_sector_names: list[str],
        non_disaggregated_sector_names: list[str],
        sector_mapping: dict[tuple[str, str], list[tuple[str, str]]],
        aggregated_sectors_list: list[tuple[str, str]],
        m: int,
        final_demand: pd.Series,
    ):
        super().__init__(
            sectors=sectors,
            reordered_matrix=reordered_matrix,
            disaggregated_sector_names=disaggregated_sector_names,
            non_disaggregated_sector_names=non_disaggregated_sector_names,
        )
        self.sector_mapping = sector_mapping
        self.aggregated_sectors_list = aggregated_sectors_list
        self.m = m
        self.final_demand = final_demand

    @classmethod
    def from_technical_coefficients(
        cls,
        reader: ICIOReader,
        sector_mapping: dict[str, list[str]],
    ) -> "DisaggregatedBlocks":
        """Create DisaggregatedBlocks from an ICIOReader and sector mapping.

        Args:
            reader: ICIOReader instance containing the technical coefficients
            sector_mapping: Dictionary mapping aggregated sectors to their disaggregated subsectors
                e.g. {"A": ["A01", "A03"]} - this will be applied to all countries

        Returns:
            DisaggregatedBlocks instance with reordered matrix and sector info
        """
        # Get technical coefficients from reader
        tech_coef = reader.technical_coefficients

        # Create full mapping including all countries
        full_mapping: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for country in reader.countries:
            for agg_sector, disagg_sectors in sector_mapping.items():
                # Create country-sector pairs
                agg_pair = (country, agg_sector)
                disagg_pairs = [(country, s) for s in disagg_sectors]
                full_mapping[agg_pair] = disagg_pairs

        # Create list of (sector_id, name, k) tuples for each aggregated sector
        sectors_info = []
        for agg_pair, disagg_pairs in full_mapping.items():
            # Use sector code as name and number of subsectors as k
            for pair in disagg_pairs:
                sectors_info.append((pair, pair[1], 1))

        # Create base DisaggregationBlocks instance
        blocks = DisaggregationBlocks.from_technical_coefficients(tech_coef, sectors_info)

        # Create ordered list of aggregated sectors
        aggregated_sectors_list = sorted(list(full_mapping.keys()))

        # Get final demand and output for disaggregated sectors
        final_demand = reader.final_demand.loc[blocks.disaggregated_sector_names]
        output = reader.output_from_out.loc[blocks.disaggregated_sector_names]

        # Create a Series with the same index as final_demand but containing total output of aggregated sectors
        total_output = pd.Series(index=final_demand.index, dtype=float)
        for agg_sector, disagg_sectors in full_mapping.items():
            # Get total output for this aggregated sector
            sector_output = output.loc[disagg_sectors].sum()
            # Assign this total to each disaggregated sector in the group
            for sector in disagg_sectors:
                total_output.loc[sector] = sector_output

        # Rescale final demand by total output
        final_demand = final_demand / total_output

        # Return DisaggregatedBlocks instance
        return cls(
            sectors=blocks.sectors,
            reordered_matrix=blocks.reordered_matrix,
            disaggregated_sector_names=blocks.disaggregated_sector_names,
            non_disaggregated_sector_names=blocks.non_disaggregated_sector_names,
            sector_mapping=full_mapping,
            aggregated_sectors_list=aggregated_sectors_list,
            m=len(aggregated_sectors_list),
            final_demand=final_demand,
        )

    def get_e_vector(self, n: int) -> np.ndarray:
        # get the sector name
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        # get the disaggregated sectors
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        # get the aggregated sectors in disaggregated sectors keeping the ordering
        indices = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors
        ]
        disaggregated_sectors = [self.disaggregated_sector_names[i] for i in indices]

        # get the block
        E = self.reordered_matrix.loc[
            self.non_disaggregated_sector_names, disaggregated_sectors
        ].values

        # flatten in order 11, 12, 13..., 21, 22, 23...
        E = E.flatten()
        return E

    def get_f_vector(self, n: int) -> np.ndarray:
        # get the sector name
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        # get the disaggregated sectors
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        # get the aggregated sectors in disaggregated sectors keeping the ordering
        indices = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors
        ]
        disaggregated_sectors = [self.disaggregated_sector_names[i] for i in indices]

        # get the block
        F = self.reordered_matrix.loc[
            disaggregated_sectors, self.non_disaggregated_sector_names
        ].values

        # flatten in order 11, 12, 13..., 21, 22, 23...
        F = F.flatten(order="F")
        return F

    def get_gnl_vector(self, n: int, l: int) -> np.ndarray:
        # get the sector name
        aggregated_sector_n = self.aggregated_sectors_list[n - 1]
        aggregated_sector_l = self.aggregated_sectors_list[l - 1]
        # get the disaggregated sectors
        disaggregated_sectors_n = self.sector_mapping[aggregated_sector_n]
        disaggregated_sectors_l = self.sector_mapping[aggregated_sector_l]

        # get the aggregated sectors in disaggregated sectors keeping the ordering
        indices_n = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors_n
        ]
        indices_l = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors_l
        ]
        disaggregated_sectors_n = [self.disaggregated_sector_names[i] for i in indices_n]
        disaggregated_sectors_l = [self.disaggregated_sector_names[i] for i in indices_l]

        # get the block
        gnl = self.reordered_matrix.loc[disaggregated_sectors_n, disaggregated_sectors_l].values

        return gnl.flatten()

    def get_gn_vector(self, n: int):
        gnls = [self.get_gnl_vector(n, l) for l in range(1, self.m + 1)]

        return np.concatenate(gnls)

    def get_bn_vector(self, n: int) -> np.ndarray:
        # final demand for the disaggregated sectors
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        # get the disaggregated sectors
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        indices = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors
        ]
        disaggregated_sectors = [self.disaggregated_sector_names[i] for i in indices]

        bn = self.final_demand.loc[disaggregated_sectors].values
        return bn

    def get_xn_vector(self, n: int) -> np.ndarray:
        # get E, F, G, bn
        E = self.get_e_vector(n)
        F = self.get_f_vector(n)
        G = self.get_gn_vector(n)
        bn = self.get_bn_vector(n)

        return np.concatenate([E, F, G, bn])
