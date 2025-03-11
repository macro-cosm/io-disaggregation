"""Module for handling block structure in disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.constraints import (
    generate_M1_block,
    generate_M2_block,
    generate_M4_block,
    generate_M5_block,
)
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
    """Class to handle the block structure of a disaggregation problem.

    This class works with sectors that are ABOUT TO BE disaggregated. It takes a technical
    coefficients matrix where sectors are still in their aggregated form and uses sectors_info
    to specify how each sector should be split into subsectors.

    For example, if you want to split sector AGR into 3 subsectors:
    ```python
    sectors_info = [("AGR", "Agriculture", 3)]
    blocks = DisaggregationBlocks.from_technical_coefficients(tech_coef, sectors_info, output)
    ```

    The matrix is organized into blocks that represent different relationships:
    ```
    [A₀    B¹  B²  ... Bᵏ ]
    [C¹    D¹¹ D¹² ... D¹ᵏ]
    [C²    D²¹ D²² ... D²ᵏ]
    [...   ... ... ... ...]
    [Cᵏ    Dᵏ¹ ... ... Dᵏᵏ]
    ```

    Where:
    - A₀: Technical coefficients between undisaggregated sectors
    - Bⁿ: Technical coefficients from undisaggregated to sector n
    - Cⁿ: Technical coefficients from sector n to undisaggregated
    - Dⁿˡ: Technical coefficients between sectors n and l

    Attributes:
        sectors: List of sectors being disaggregated, with their indices
        disaggregated_sector_names: List of sector codes/names being disaggregated
        non_disaggregated_sector_names: List of sector codes/names not being disaggregated
        reordered_matrix: The reordered technical coefficients matrix
        output: Output values for all sectors
    """

    sectors: list[SectorInfo]
    disaggregated_sector_names: list[str | tuple[str, str]]
    non_disaggregated_sector_names: list[str | tuple[str, str]]
    reordered_matrix: pd.DataFrame
    output: pd.Series

    @classmethod
    def from_technical_coefficients(
        cls,
        tech_coef: pd.DataFrame,
        sectors_info: list[tuple[SectorId, str, int]],
        output: pd.Series,
    ) -> "DisaggregationBlocks":
        """Create DisaggregationBlocks from a technical coefficients matrix.

        Args:
            tech_coef: Technical coefficients matrix
            sectors_info: List of tuples (sector_id, name, k) where:
                - sector_id is a sector identifier (str for single region, tuple for multi)
                - name is a human-readable name
                - k is the number of subsectors to split into
            output: Output vector for all  sectors

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

        if isinstance(disaggregated[0], tuple):
            name_dict.update({s: name_dict[s[1]] for s in disaggregated})
            k_dict.update({s: k_dict[s[1]] for s in disaggregated})

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

        reordered_outputs = output.loc[new_order]

        # Filter out special rows/columns from tech_coef before reindexing
        filtered_tech_coef = tech_coef.loc[new_order, new_order]
        return cls(
            sectors=sectors,
            reordered_matrix=filtered_tech_coef,
            disaggregated_sector_names=disaggregated,
            non_disaggregated_sector_names=undisaggregated,
            output=reordered_outputs,
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
        """Get the A₀ block (technical coefficients between undisaggregated sectors).

        Returns:
            Array of shape (N-K, N-K) where:
            - N is the total number of sectors
            - K is the number of sectors being disaggregated
        """
        N_K = self.N - self.K
        return self.reordered_matrix.iloc[:N_K, :N_K].values

    def get_B(self, n: int) -> np.ndarray:
        """Get the B^n block (technical coefficients from undisaggregated to sector n).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array of shape (N-K,) representing input requirements from undisaggregated
            sectors to sector n
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.K}]")

        # Find the column index for sector n
        sector_id = self.sectors[n - 1].sector_id
        try:
            col_idx = self.reordered_matrix.columns.get_loc(sector_id)
        except KeyError:
            logger.error(f"Sector {sector_id} not found in matrix columns")
            raise

        # Get the block and repeat it for each subsector
        B = self.reordered_matrix.loc[self.non_disaggregated_sector_names, sector_id]

        return B.values

    def get_C(self, n: int) -> np.ndarray:
        """Get the C^n block (technical coefficients from sector n to undisaggregated).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array of shape (N-K,) representing input requirements from sector n
            to undisaggregated sectors
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

        C = self.reordered_matrix.loc[sector_id, self.non_disaggregated_sector_names].values

        return C

    def get_D_nl(self, n: int, l: int) -> float:
        """Get the D^{nℓ} block (technical coefficients between sectors n and ℓ).

        Args:
            n: Index of the first sector (1-based as in math notation)
            l: Index of the second sector (1-based as in math notation)

        Returns:
            Technical coefficient representing input requirements from sector n to sector ℓ
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"First sector index {n} out of range [1, {self.K}]")
        if not 1 <= l <= self.K:
            raise ValueError(f"Second sector index {l} out of range [1, {self.K}]")

        # Find the row and column indices
        sector_n = self.sectors[n - 1].sector_id
        sector_l = self.sectors[l - 1].sector_id

        return self.reordered_matrix.loc[sector_n, sector_l]  # noqa

    def get_D(self, n: int) -> np.ndarray:
        """Get all D blocks for sector n (technical coefficients to all other disaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from sector n to all other disaggregated sectors
        """
        return np.array([self.get_D_nl(n, l) for l in range(1, self.K + 1)])

    def get_y_vector(self, n: int, relative_weights: np.ndarray) -> np.ndarray:
        """Get the y vector for sector n, combining B, C, D vectors and relative weights.

        Args:
            n: Index of the sector (1-based as in math notation)
            relative_weights: Array of relative weights for the subsectors

        Returns:
            Array containing concatenated [B, C, D, relative_weights] vectors
        """
        # get B, C, D vectors
        B = self.get_B(n)
        C = self.get_C(n)
        D = self.get_D(n)
        return np.concatenate([B, C, D, relative_weights])

    def get_sector_info(self, n: int) -> SectorInfo:
        """
        Get information about a sector by its index.

        Args:
            n: Index of the sector (1-based as in math notation)
        """
        if not 1 <= n <= self.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.K}]")
        return self.sectors[n - 1]

    def get_m1_block(self, n: int, relative_weights: np.ndarray) -> np.ndarray:
        """Get the M₁ block for sector n (weights for technical coefficients from undisaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)
            relative_weights: Array of relative weights for the subsectors

        Returns:
            Array representing the M₁ constraint matrix block
        """
        k = self.sectors[n - 1].k
        return generate_M1_block(len(self.non_disaggregated_sector_names), k, relative_weights)

    def get_m2_block(self, n: int) -> np.ndarray:
        """Get the M₂ block for sector n (sum preservation for technical coefficients).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array representing the M₂ constraint matrix block
        """
        k = self.sectors[n - 1].k
        return generate_M2_block(len(self.non_disaggregated_sector_names), k)

    def get_m3_nl_block(self, n: int, weights_l: np.ndarray) -> np.ndarray:
        """Get the M₃ block for sectors n and l (cross-sector weight constraints).

        Args:
            n: Index of the first sector (1-based as in math notation)
            weights_l: Array of weights for sector l's subsectors

        Returns:
            Array representing the M₃ constraint matrix block for sectors n and l
        """
        # repeat weights_l k_n times
        k_n = self.sectors[n - 1].k
        return np.array(list(weights_l) * k_n)

    def get_m3_block(self, n: int, relative_weights: list[np.ndarray]) -> np.ndarray:
        """Get the complete M₃ block for sector n (all cross-sector weight constraints).

        Args:
            n: Index of the sector (1-based as in math notation)
            relative_weights: List of weight arrays for each sector's subsectors

        Returns:
            Array representing the complete M₃ constraint matrix block
        """
        k = self.sectors[n - 1].k
        n_cols = sum(len(w) * k for w in relative_weights)
        m3 = np.zeros((k, n_cols))
        col_index = 0
        for i, weights in enumerate(relative_weights):
            block = self.get_m3_nl_block(n, weights)
            m3[i, col_index : col_index + len(block)] = block
            col_index += len(block)
        return m3

    def get_m4_nl_block(self, n: int, l: int, weights_l: np.ndarray) -> np.ndarray:
        """Get the M₄ block for sectors n and l (output-scaled weight constraints).

        Args:
            n: Index of the first sector (1-based as in math notation)
            l: Index of the second sector (1-based as in math notation)
            weights_l: Array of weights for sector l's subsectors

        Returns:
            Array representing the M₄ constraint matrix block for sectors n and l,
            scaled by the ratio of sector outputs
        """
        n_l = len(weights_l)
        output_n = self.output.loc[self.disaggregated_sector_names[n - 1]]
        output_l = self.output.loc[self.disaggregated_sector_names[l - 1]]
        output_ratio = output_l / output_n
        m4 = np.zeros((self.sectors[n - 1].k, n_l * self.sectors[n - 1].k))
        for i in range(self.sectors[n - 1].k):
            m4[i, i * n_l : (i + 1) * n_l] = weights_l
        return m4 * output_ratio

    def get_m4_block(self, n: int, relative_weights: list[np.ndarray]) -> np.ndarray:
        """Get the complete M₄ block for sector n (all output-scaled weight constraints).

        Args:
            n: Index of the sector (1-based as in math notation)
            relative_weights: List of weight arrays for each sector's subsectors

        Returns:
            Array representing the complete M₄ constraint matrix block
        """
        m4_list = [
            self.get_m4_nl_block(n, l + 1, weights_l)
            for l, weights_l in enumerate(relative_weights)
        ]
        return np.concatenate(m4_list, axis=1)

    def get_m5_block(self, n: int) -> np.ndarray:
        """Get the M₅ block for sector n (final demand consistency constraints).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array representing the M₅ constraint matrix block
        """
        m5 = generate_M5_block(
            k_n=self.sectors[n - 1].k,
            x=self.output.loc[self.non_disaggregated_sector_names].values,
            z_n=self.output.loc[self.disaggregated_sector_names[n - 1]],
        )
        return m5

    def get_large_m(self, n: int, relative_weights: list[np.ndarray]) -> np.ndarray:
        """Get the complete constraint matrix M for sector n.

        This matrix combines all constraint blocks (M₁ through M₅) into a single large
        constraint matrix used in the optimization problem.

        Args:
            n: Index of the sector (1-based as in math notation)
            relative_weights: List of weight arrays for each sector's subsectors

        Returns:
            Array representing the complete constraint matrix M
        """
        m1 = self.get_m1_block(n, relative_weights[n - 1])
        m2 = self.get_m2_block(n)
        m3 = self.get_m3_block(n, relative_weights)
        m4 = self.get_m4_block(n, relative_weights)
        m5 = self.get_m5_block(n)
        id_block = np.eye(self.sectors[n - 1].k)

        total_columns = m1.shape[1] + m2.shape[1] + m3.shape[1] + self.sectors[n - 1].k
        total_rows = m1.shape[0] + m2.shape[0] + m3.shape[0] + self.sectors[n - 1].k

        large_m = np.zeros((total_rows, total_columns))

        # Assemble the blocks into the large matrix
        col_counter, row_counter = 0, 0
        large_m[
            row_counter : row_counter + m1.shape[0], col_counter : col_counter + m1.shape[1]
        ] = m1
        col_counter += m1.shape[1]
        row_counter += m1.shape[0]
        large_m[
            row_counter : row_counter + m2.shape[0], col_counter : col_counter + m2.shape[1]
        ] = m2
        col_counter += m2.shape[1]
        row_counter += m2.shape[0]
        large_m[
            row_counter : row_counter + m3.shape[0], col_counter : col_counter + m3.shape[1]
        ] = m3

        row_counter += m3.shape[0]

        col_counter = m1.shape[1]
        large_m[
            row_counter : row_counter + self.sectors[n - 1].k,
            col_counter : col_counter + m2.shape[1],
        ] = m5
        col_counter += m2.shape[1]
        large_m[
            row_counter : row_counter + self.sectors[n - 1].k,
            col_counter : col_counter + m3.shape[1],
        ] = m4
        col_counter += m3.shape[1]

        large_m[
            row_counter : row_counter + self.sectors[n - 1].k,
            col_counter : col_counter + self.sectors[n - 1].k,
        ] = id_block

        return large_m


class DisaggregatedBlocks(DisaggregationBlocks):
    """Class to handle the block structure of an already disaggregated problem.

    This class works with sectors that are ALREADY disaggregated in the input data.
    It takes a technical coefficients matrix that already contains the disaggregated sectors
    and uses sector_mapping to specify which subsectors belong to which aggregated sector.

    For example, if A01 and A03 are already separate sectors in your data:
    ```python
    sector_mapping = {"A": ["A01", "A03"]}
    blocks = DisaggregatedBlocks.from_reader(reader, sector_mapping)
    ```

    The class provides methods to access the E, F, G blocks which represent:
    - E block: Technical coefficients from undisaggregated to disaggregated sectors
    - F block: Technical coefficients from disaggregated to undisaggregated sectors
    - G block: Technical coefficients between disaggregated sectors
    - b block: Final demand for disaggregated sectors

    Attributes:
        sector_mapping: Dictionary mapping aggregated sectors to their disaggregated subsectors
            e.g. {("USA", "A"): [("USA", "A01"), ("USA", "A03")]}
        aggregated_sectors_list: List of aggregated sectors in order
        m: Total number of subsectors
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
        output: pd.Series,
    ):
        super().__init__(
            sectors=sectors,
            reordered_matrix=reordered_matrix,
            disaggregated_sector_names=disaggregated_sector_names,
            non_disaggregated_sector_names=non_disaggregated_sector_names,
            output=output,
        )
        self.sector_mapping = sector_mapping
        self.aggregated_sectors_list = aggregated_sectors_list
        self.m = m
        self.final_demand = final_demand

    @classmethod
    def from_reader(
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

        full_mapping, sectors_info = get_sectors_info(reader.countries, sector_mapping)

        # Create base DisaggregationBlocks instance
        blocks = DisaggregationBlocks.from_technical_coefficients(
            tech_coef, sectors_info, output=reader.output_from_out
        )

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

        output = reader.output_from_out.loc[blocks.reordered_matrix.index]

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
            output=output,
        )

    def get_e_vector(self, n: int) -> np.ndarray:
        """Get the E block for sector n (technical coefficients from undisaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from all undisaggregated sectors
            to the subsectors of sector n
        """
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
        """Get the F block for sector n (technical coefficients to undisaggregated sectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from the subsectors of sector n
            to all undisaggregated sectors
        """
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
        """Get the G block for sectors n and l (technical coefficients between their subsectors).

        Args:
            n: Index of the first sector (1-based as in math notation)
            l: Index of the second sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from subsectors of sector n to
            subsectors of sector l, flattened in row-major order
        """
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
        """Get the G block for sector n (technical coefficients between subsectors).

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing technical coefficients from the subsectors of sector n
            to all other subsectors
        """
        gnls = [self.get_gnl_vector(n, l) for l in range(1, self.m + 1)]

        return np.concatenate(gnls)

    def get_bn_vector(self, n: int) -> np.ndarray:
        """Get the final demand block for sector n.

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing final demand values for the subsectors of sector n
        """
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
        """Get the complete solution vector for sector n.

        This combines all components of the solution vector:
        - E block: Technical coefficients from undisaggregated to subsectors of n
        - F block: Technical coefficients from subsectors of n to undisaggregated
        - G block: Technical coefficients from subsectors of n to all other subsectors
        - b block: Final demand for subsectors of n

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array containing the concatenated [E, F, G, b] vectors for sector n
        """
        # get E, F, G, bn
        E = self.get_e_vector(n)
        F = self.get_f_vector(n)
        G = self.get_gn_vector(n)
        bn = self.get_bn_vector(n)

        return np.concatenate([E, F, G, bn])

    def get_relative_output_weights(self, n: int) -> np.ndarray:
        """Get relative output weights for subsectors of sector n.

        These weights represent each subsector's share of the total output of sector n.
        They sum to 1 and are used to ensure technical coefficients are proportional
        to output shares.

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            Array of relative weights (output shares) for each subsector
        """
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        # get the disaggregated sectors
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        indices = [
            self.disaggregated_sector_names.index(sector) for sector in disaggregated_sectors
        ]
        disaggregated_sectors = [self.disaggregated_sector_names[i] for i in indices]

        output_vals = self.output.loc[disaggregated_sectors].values
        return output_vals / output_vals.sum()


def get_sectors_info(countries: list[str], sector_mapping: dict[str, list[str]]):
    """Create sector information for all countries based on sector mapping.

    For multi-region tables, this function creates sector information tuples for each
    country-sector combination, preserving the mapping between aggregated and
    disaggregated sectors.

    Args:
        countries: List of country codes in the table
        sector_mapping: Dictionary mapping aggregated sector codes to lists of
            disaggregated sector codes

    Returns:
        List of tuples (sector_id, name, k) where:
        - sector_id is a tuple (country, sector) for each country-sector pair
        - name is the sector name (same as sector code for now)
        - k is the number of subsectors in the mapping
    """
    # Create full mapping including all countries
    full_mapping: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for country in countries:
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
    return full_mapping, sectors_info


def unfold_countries(
    countries: list[str], sector_mapping: dict[str, list[str]]
) -> list[tuple[SectorId, str, int]]:
    """Create sector information tuples for all country-sector combinations.

    This function expands the sector mapping across all countries, creating
    sector information tuples that include country codes. For example, if we have
    countries ["USA", "CAN"] and sector "A" maps to ["A01", "A03"], this will create
    entries for both ("USA", "A") and ("CAN", "A").

    Args:
        countries: List of country codes
        sector_mapping: Dictionary mapping aggregated sector codes to lists of
            disaggregated sector codes

    Returns:
        List of tuples (sector_id, name, k) where:
        - sector_id is a tuple (country, sector) for each country-sector pair
        - name is the sector name (same as sector code for now)
        - k is the number of subsectors in the mapping
    """
    sectors = []
    for sector, disagg_sectors in sector_mapping.items():
        for country in countries:
            sectors.append(((country, sector), sector, len(disagg_sectors)))
    return sectors
