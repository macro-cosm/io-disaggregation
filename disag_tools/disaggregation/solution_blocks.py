from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.disaggregation_blocks import DisaggregationBlocks, SectorInfo

SectorId: TypeAlias = str | tuple[str, str]


@dataclass
class SolutionBlocks:
    sectors: list[SectorInfo]
    reordered_matrix: pd.DataFrame
    output: pd.Series
    sector_mapping: dict[SectorId, list[SectorId]]
    aggregated_sectors_list: list[SectorId]
    non_disaggregated_sector_names: list[SectorId]

    @classmethod
    def from_disaggregation_blocks(
        cls,
        blocks: DisaggregationBlocks,
        disaggregation_dict: dict[SectorId, list[SectorId]],
        weight_dict: dict[SectorId, float],
    ) -> "SolutionBlocks":
        """Create a SolutionBlocks instance from DisaggregationBlocks and a disaggregation mapping.

        This method takes a DisaggregationBlocks instance where some sectors are marked for
        disaggregation, and a mapping that specifies how each sector should be split into
        subsectors. It creates a new matrix where the original sectors are replaced by their
        subsectors, with NaN values for the new entries.

        Args:
            blocks: DisaggregationBlocks instance containing the original matrix
            disaggregation_dict: Dictionary mapping each sector to its list of subsectors
                e.g. {("ROW", "A"): [("ROW", "A01"), ("ROW", "A03")]}
            weight_dict: Dictionary mapping each subsector to its relative output weight
                e.g. {("ROW", "A01"): 0.6, ("ROW", "A03"): 0.4}

        Returns:
            SolutionBlocks instance with the expanded matrix and output series
        """
        # Make copies to avoid modifying originals
        matrix = blocks.reordered_matrix.copy()
        output = blocks.output.copy()

        # For each sector being disaggregated
        for sector_id in blocks.to_disagg_sector_names:
            # Get the subsectors for this sector
            subsectors = disaggregation_dict[sector_id]

            # Get the original sector's output before removing it
            original_output = output[sector_id]

            # Remove the original sector's row and column
            matrix = matrix.drop(sector_id, axis=0)
            matrix = matrix.drop(sector_id, axis=1)
            output = output.drop(sector_id)

            # Add new rows and columns for subsectors
            # First add rows
            all_rows = list(matrix.index) + subsectors
            matrix = matrix.reindex(index=all_rows, fill_value=np.nan)
            # Then add columns
            all_cols = list(matrix.columns) + subsectors
            matrix = matrix.reindex(columns=all_cols, fill_value=np.nan)

            # Add output entries for subsectors using weights
            for subsector in subsectors:
                output.loc[subsector] = original_output * weight_dict.get(subsector, 0)

        # Create SectorInfo objects for the new sectors
        sectors = []
        for i, sector_id in enumerate(blocks.to_disagg_sector_names):
            subsectors = disaggregation_dict[sector_id]
            # For each subsector, create a SectorInfo with k=1 since it's already disaggregated
            for subsector in subsectors:
                name = str(subsector[1]) if isinstance(subsector, tuple) else str(subsector)
                sectors.append(
                    SectorInfo(index=len(sectors) + 1, sector_id=subsector, name=name, k=1)
                )

        return cls(
            sectors=sectors,
            reordered_matrix=matrix,
            output=output,
            sector_mapping=disaggregation_dict,
            aggregated_sectors_list=list(blocks.to_disagg_sector_names),
            non_disaggregated_sector_names=list(blocks.non_disagg_sector_names),
        )

    def apply_e_vector(self, n: int, e_vector: np.ndarray) -> np.ndarray:
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        block = self.reordered_matrix.loc[
            self.non_disaggregated_sector_names, disaggregated_sectors
        ].values

        # check that product of block dimensions is equal to e_vector length
        assert block.shape[0] * block.shape[1] == e_vector.shape[0]

        # e_vector has order 11, 12, 13...
        # put it in the block by reshaping it into the block's shape
        e_block = e_vector.reshape(block.shape)

        # apply the e_vector to the block
        self.reordered_matrix.loc[self.non_disaggregated_sector_names, disaggregated_sectors] = (
            e_block
        )

        return e_block

    def apply_e_block_guess(self, n: int, b_n: np.ndarray, weights: np.ndarray):
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        k_n = len(disaggregated_sectors)

        dims = len(self.non_disaggregated_sector_names), k_n

        e_block = np.zeros(dims)
        # e_block[i,j] is b_n[i] / (k_n * weights[j])
        for i in range(dims[0]):
            for j in range(dims[1]):
                e_block[i, j] = b_n[i] / (k_n * weights[j])

        self.reordered_matrix.loc[self.non_disaggregated_sector_names, disaggregated_sectors] = (
            e_block
        )

    def e_vector_length(self, n: int) -> int:
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]
        e_block = self.reordered_matrix.loc[
            self.non_disaggregated_sector_names, disaggregated_sectors
        ].values

        return e_block.shape[0] * e_block.shape[1]

    def apply_f_vector(self, n: int, f_vector: np.ndarray) -> np.ndarray:
        """Apply the F vector to the appropriate block in the matrix.

        Args:
            n: Index of the sector (1-based as in math notation)
            f_vector: Vector containing technical coefficients from subsectors of sector n
                to all undisaggregated sectors, in column-major (F) order

        Returns:
            The reshaped block that was applied to the matrix
        """
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        block = self.reordered_matrix.loc[
            disaggregated_sectors, self.non_disaggregated_sector_names
        ].values

        # check that product of block dimensions is equal to f_vector length
        assert block.shape[0] * block.shape[1] == f_vector.shape[0]

        # f_vector has order 11, 21, 31..., 12, 22, 32... (column-major)
        # put it in the block by reshaping it into the block's shape using order="F"
        f_block = f_vector.reshape(block.shape, order="F")

        # apply the f_vector to the block
        self.reordered_matrix.loc[disaggregated_sectors, self.non_disaggregated_sector_names] = (
            f_block
        )

        return f_block

    def f_vector_length(self, n: int) -> int:
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]
        f_block = self.reordered_matrix.loc[
            disaggregated_sectors, self.non_disaggregated_sector_names
        ].values

        return f_block.shape[0] * f_block.shape[1]

    def apply_f_block_guess(self, n: int, c_n: np.ndarray):
        aggregated_sector = self.aggregated_sectors_list[n - 1]
        disaggregated_sectors = self.sector_mapping[aggregated_sector]

        k_n = len(disaggregated_sectors)

        dims = k_n, len(self.non_disaggregated_sector_names)

        f_block = np.zeros(dims)
        # f_block[i,j] is c_n[j] / k_n
        for i in range(dims[0]):
            for j in range(dims[1]):
                f_block[i, j] = c_n[j] / k_n

        self.reordered_matrix.loc[disaggregated_sectors, self.non_disaggregated_sector_names] = (
            f_block
        )

    def apply_gnl_vector(self, n: int, l: int, g_vector: np.ndarray) -> np.ndarray:
        """Apply the G vector to the appropriate block in the matrix.

        Args:
            n: Index of the source sector (1-based as in math notation)
            l: Index of the destination sector (1-based as in math notation)
            g_vector: Vector containing technical coefficients from subsectors of sector n
                to subsectors of sector l, in column-major (F) order

        Returns:
            The reshaped block that was applied to the matrix
        """
        # Get source and destination sectors and their subsectors
        source_sector = self.aggregated_sectors_list[n - 1]
        dest_sector = self.aggregated_sectors_list[l - 1]
        source_subsectors = self.sector_mapping[source_sector]
        dest_subsectors = self.sector_mapping[dest_sector]

        # Get the current block
        block = self.reordered_matrix.loc[source_subsectors, dest_subsectors].values

        # check that product of block dimensions is equal to g_vector length
        assert block.shape[0] * block.shape[1] == g_vector.shape[0]

        # g_vector has order 11, 21, 31..., 12, 22, 32... (column-major)
        # put it in the block by reshaping it into the block's shape using order="F"
        g_block = g_vector.reshape(block.shape)

        # apply the g_vector to the block
        self.reordered_matrix.loc[source_subsectors, dest_subsectors] = g_block

        return g_block

    def apply_gnl_block_guess(self, n: int, l: int, d_nl: float, weights: np.ndarray):
        source_sector = self.aggregated_sectors_list[n - 1]
        dest_sector = self.aggregated_sectors_list[l - 1]
        source_subsectors = self.sector_mapping[source_sector]
        dest_subsectors = self.sector_mapping[dest_sector]

        k_n = len(source_subsectors)
        k_l = len(dest_subsectors)

        dims = k_n, k_l

        g_block = np.zeros(dims)
        # g_block[i,j] is d_nl / (weights[j] * k_n * k_l)
        for i in range(dims[0]):
            for j in range(dims[1]):
                g_block[i, j] = d_nl / (weights[j] * k_n * k_l)

        self.reordered_matrix.loc[source_subsectors, dest_subsectors] = g_block

    def g_nl_vector_length(self, n: int, l: int) -> int:
        source_sector = self.aggregated_sectors_list[n - 1]
        dest_sector = self.aggregated_sectors_list[l - 1]
        source_subsectors = self.sector_mapping[source_sector]
        dest_subsectors = self.sector_mapping[dest_sector]
        g_block = self.reordered_matrix.loc[source_subsectors, dest_subsectors].values

        return g_block.shape[0] * g_block.shape[1]

    def apply_gn_vector(self, n: int, g_n: np.ndarray):
        start = 0
        for l in range(1, len(self.aggregated_sectors_list) + 1):
            g_n_l = g_n[start : start + self.g_nl_vector_length(n, l)]
            self.apply_gnl_vector(n, l, g_n_l)
            start += len(g_n_l)

    def g_vector_length(self, n: int) -> int:
        return sum(
            [self.g_nl_vector_length(n, l) for l in range(1, len(self.aggregated_sectors_list) + 1)]
        )

    def apply_xn(self, n: int, x_n: np.ndarray):
        """Apply the x_n vector to the appropriate block in the matrix.

        Args:
            n: Index of the sector (1-based as in math notation)
            x_n: Vector containing concatenated vectors of technical coefficients from subsectors
                of sector n, in order [E, F, G, B] (vector B is not handled here)
        """
        e_length = self.e_vector_length(n)
        f_length = self.f_vector_length(n)
        g_length = self.g_vector_length(n)

        start = 0
        e_vector = x_n[start:e_length]
        start += e_length
        f_vector = x_n[start : start + f_length]
        start += f_length
        g_vector = x_n[start : start + g_length]

        self.apply_e_vector(n, e_vector)
        self.apply_f_vector(n, f_vector)
        self.apply_gn_vector(n, g_vector)

    def get_intermediate_use(self) -> pd.DataFrame:
        """Convert technical coefficients to intermediate use values.

        The technical coefficients matrix A is obtained from the use matrix U by:
            A[i,j] = U[i,j] / output[j]

        Therefore, to get U back:
            U[i,j] = A[i,j] * output[j]

        Returns:
            DataFrame containing the intermediate use values with the same structure
            as the technical coefficients matrix.

        Raises:
            ValueError: If the technical coefficients matrix contains NaN values,
                indicating that solutions have not been fully applied.
        """
        # Check for NaN values in the technical coefficients
        if self.reordered_matrix.isna().any().any():
            raise ValueError(
                "Technical coefficients matrix contains NaN values. "
                "Solutions must be fully applied before getting intermediate use."
            )

        # For each column j, multiply by the output of sector j
        intermediate_use = pd.DataFrame(
            index=self.reordered_matrix.index, columns=self.reordered_matrix.columns, dtype=float
        )

        for col in self.reordered_matrix.columns:
            intermediate_use.loc[:, col] = self.reordered_matrix.loc[:, col] * self.output.loc[col]

        return intermediate_use
