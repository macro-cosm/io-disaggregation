"""Module for assembling solved disaggregation results into an ICIO reader-like object."""

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.problem import DisaggregationProblem
from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)

# Type alias for sector identifiers
SectorId: TypeAlias = str | tuple[str, str]


@dataclass
class AssembledData:
    """
    Class holding the assembled ICIO table after disaggregation.

    This class contains a DataFrame that matches the structure of ICIOReader.data,
    with disaggregated sectors replacing their original aggregated sectors.

    Attributes:
        data: DataFrame containing the complete ICIO table with disaggregated sectors.
            Has the same structure as ICIOReader.data with MultiIndex for both rows
            and columns.
    """

    data: pd.DataFrame

    @classmethod
    def from_solution(cls, problem: DisaggregationProblem, reader: ICIOReader) -> "AssembledData":
        """
        Create an AssembledData instance from a solved disaggregation problem.

        This method concatenates:
        1. Intermediate use table from solution blocks
        2. Final demand table from final demand blocks
        3. Output row from solution blocks
        4. Value Added and TLS rows from both disaggregated and non-disaggregated sectors

        Args:
            problem: Solved disaggregation problem containing the solution
            reader: Original ICIO reader used for the disaggregation

        Returns:
            AssembledData instance containing the complete assembled table
        """
        # Get intermediate use from technical coefficients
        intermediate_use = problem.solution_blocks.get_intermediate_use()

        # Sort intermediate use index by country and sector
        countries = sorted({idx[0] for idx in intermediate_use.index})
        ordered_rows = []
        for country in countries:
            # Get all industry rows for this country and sort them
            country_rows = sorted([idx for idx in intermediate_use.index if idx[0] == country])
            ordered_rows.extend(country_rows)
        intermediate_use = intermediate_use.reindex(index=ordered_rows)

        # Get final demand for all sectors and ensure it has the same index as intermediate use
        final_demand = problem.final_demand_blocks.disaggregated_final_demand.reindex(
            index=intermediate_use.index, fill_value=np.nan
        )

        # Get final demand columns from reader
        fd_cols = [
            col
            for col in reader.data.columns
            if any(col[1].startswith(p) for p in reader.FINAL_DEMAND_PREFIXES)
        ]

        # Reorder intermediate use columns by country and sector
        countries = sorted({col[0] for col in intermediate_use.columns})
        ordered_int_cols = []
        for country in countries:
            # Get all industry columns for this country and sort them
            country_cols = sorted([col for col in intermediate_use.columns if col[0] == country])
            ordered_int_cols.extend(country_cols)

        # Reorder final demand columns by country
        ordered_fd_cols = []
        for country in countries:
            # Get all final demand columns for this country
            country_cols = sorted([col for col in fd_cols if col[0] == country])
            ordered_fd_cols.extend(country_cols)

        # Create MultiIndex for columns with ordered columns
        all_cols = pd.MultiIndex.from_tuples(
            ordered_int_cols + ordered_fd_cols + [("OUT", "OUT")],
            names=["CountryInd", "industryInd"],
        )

        # Create DataFrame with all columns
        data = pd.DataFrame(
            index=intermediate_use.index,
            columns=all_cols,
            dtype=float,
        )

        # Fill intermediate use values
        data.loc[:, ordered_int_cols] = intermediate_use.loc[:, ordered_int_cols]

        # Fill final demand values
        data.loc[:, ordered_fd_cols] = final_demand.loc[:, ordered_fd_cols]

        # Fill output column with output values
        output = problem.solution_blocks.output
        data.loc[intermediate_use.index, ("OUT", "OUT")] = output

        # Create bottom rows DataFrame with VA and TLS for all sectors
        bottom_rows = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [("VA", "VA"), ("TLS", "TLS")], names=["CountryInd", "industryInd"]
            ),
            columns=all_cols,
            dtype=float,
        )

        # Fill VA and TLS for disaggregated sectors from bottom_blocks
        disagg_bottom = problem.bottom_blocks.data.reindex(columns=all_cols, fill_value=np.nan)
        bottom_rows.loc[("VA", "VA"), ordered_int_cols] = disagg_bottom.loc[
            ("VA", "VA"), ordered_int_cols
        ]

        # Fill VA and TLS for non-disaggregated sectors from reader
        non_disagg_sectors = problem.disaggregation_blocks.non_disagg_sector_names
        bottom_rows.loc[("VA", "VA"), non_disagg_sectors] = reader.data.loc[
            ("VA", "VA"), non_disagg_sectors
        ]

        # Compute TLS values for all intermediate use columns
        for col in ordered_int_cols:
            intermediate_demand = intermediate_use[col].sum()  # Sum over all input sectors
            va = bottom_rows.loc[("VA", "VA"), col]
            out = output[col]
            tls = out - intermediate_demand - va
            bottom_rows.loc[("TLS", "TLS"), col] = tls

        # Copy VA and TLS values for final demand and output columns from reader
        bottom_rows.loc[:, ordered_fd_cols + [("OUT", "OUT")]] = reader.data.loc[
            [("VA", "VA"), ("TLS", "TLS")], ordered_fd_cols + [("OUT", "OUT")]
        ]

        # Create output row with values from reader for final demand and output
        output_row = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("OUT", "OUT")], names=["CountryInd", "industryInd"]),
            columns=all_cols,
            dtype=float,
        )
        # Fill intermediate use values
        output_row.loc[("OUT", "OUT"), ordered_int_cols] = output
        # Copy final demand and output values from reader
        output_row.loc[("OUT", "OUT"), ordered_fd_cols + [("OUT", "OUT")]] = reader.data.loc[
            ("OUT", "OUT"), ordered_fd_cols + [("OUT", "OUT")]
        ]

        # Concatenate all rows in the correct order: regular rows, VA, TLS, OUT
        data = pd.concat([data, bottom_rows, output_row])

        return cls(data=data)
