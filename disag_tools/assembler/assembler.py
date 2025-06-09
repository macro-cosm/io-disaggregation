"""Module for assembling solved disaggregation results into an ICIO reader-like object."""

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.disaggregation.problem import DisaggregationProblem
from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.readers.mapping import ICIO_ALL


logger = logging.getLogger(__name__)

# Type alias for sector identifiers
SectorId: TypeAlias = str | tuple[str, str]


# define an error type for the AssembledData class
class AssembledDataError(Exception):
    pass


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
    def from_solution(
        cls, problem: DisaggregationProblem, reader: ICIOReader, check_output: bool = True
    ) -> "AssembledData":
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
            check_output: Whether to check that the output matches the expected output

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
        logger.debug(f"Final demand columns: {final_demand.columns.tolist()}")
        logger.debug(f"Final demand shape: {final_demand.shape}")
        logger.debug(
            f"Final demand column countries: {sorted(set(col[0] for col in final_demand.columns))}"
        )

        # Only distribute ROW final demand to regions if this is a regional disaggregation
        if problem.regionalised:
            sector_mapping = problem.solution_blocks.sector_mapping

            country_to_regions = {}

            for key, value in sector_mapping.items():
                agg_country = key[0]
                if agg_country not in country_to_regions:
                    country_to_regions[agg_country] = []
                for sector in value:
                    region_candidate = sector[0]
                    in_list = region_candidate in country_to_regions[agg_country]
                    not_country = region_candidate != agg_country
                    if not in_list and not_country:
                        country_to_regions[agg_country].append(region_candidate)

            # Calculate output shares for each region
            for country, regions in country_to_regions.items():
                if len(regions) > 1:  # Only need to distribute if there are multiple regions
                    region_outputs = problem.solution_blocks.output.groupby(level=0).sum()
                    region_outputs = region_outputs.loc[regions]
                    region_shares = region_outputs / region_outputs.sum()

                    # Distribute ROW and other countries final demand to each region based on output shares
                    for region in regions:
                        for extra_country in set(countries) - set(regions):
                            final_demand.loc[extra_country, region] = (
                                reader.final_demand_table.loc[extra_country, country]
                                * region_shares.loc[region]
                            ).values

        # Get final demand columns from reader
        fd_cols = [
            col
            for col in reader.data.columns
            if any(col[1].startswith(p) for p in reader.FINAL_DEMAND_PREFIXES)
        ]
        logger.debug(f"Reader final demand columns: {fd_cols}")
        logger.debug(f"Number of reader final demand columns: {len(fd_cols)}")
        logger.debug(
            f"Reader final demand column countries: {sorted(set(col[0] for col in fd_cols))}"
        )

        # Reorder intermediate use columns by country and sector
        countries = sorted({col[0] for col in intermediate_use.columns})
        logger.debug(f"Countries in intermediate use: {countries}")
        ordered_int_cols = []
        for country in countries:
            # Get all industry columns for this country and sort them
            country_cols = sorted([col for col in intermediate_use.columns if col[0] == country])
            ordered_int_cols.extend(country_cols)

        # Reorder final demand columns by country
        ordered_fd_cols = []
        for country in countries:
            # Get all final demand columns for this country from disaggregated final demand
            country_cols = sorted([col for col in final_demand.columns if col[0] == country])
            logger.debug(f"Final demand columns for country {country}: {country_cols}")
            logger.debug(f"Number of final demand columns for {country}: {len(country_cols)}")
            ordered_fd_cols.extend(country_cols)
        logger.debug(f"Ordered final demand columns: {ordered_fd_cols}")
        logger.debug(f"Number of ordered final demand columns: {len(ordered_fd_cols)}")
        logger.debug(
            f"Ordered final demand column countries: {sorted(set(col[0] for col in ordered_fd_cols))}"
        )

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

        # non output columns
        non_out_cols = [col for col in all_cols if col[1] != "OUT"]

        new_output = data.loc[:, non_out_cols].sum(axis=1)

        # Fill output column with output values
        output = problem.solution_blocks.output

        norm_diff = np.linalg.norm(new_output - output) / np.linalg.norm(output)

        if check_output:
            if norm_diff > 1:
                raise AssembledDataError(
                    "Solution's output differs significantly from the expected output"
                )
            if not np.allclose(new_output, output, rtol=1e-5):
                # warn that the output is not exactly the same, and that we are fixing
                logger.warning(
                    "Solution's output differs slightly from the expected output. Fixing."
                )
                output = new_output

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

        # Map disaggregated final demand columns to original reader columns
        fd_mapping = {}
        for col in ordered_fd_cols:
            # For ROW columns, keep as is
            if col[0] == "ROW":
                fd_mapping[col] = col
            # For disaggregated columns, map back to original country
            else:
                # Extract the original country from the disaggregated country code
                orig_country = col[0].split("_")[0]  # e.g., "CAN_AB" -> "CAN"
                fd_mapping[col] = (orig_country, col[1])

        # Copy VA and TLS values for final demand columns using the mapping
        for col in ordered_fd_cols:
            orig_col = fd_mapping[col]
            bottom_rows.loc[:, col] = reader.data.loc[[("VA", "VA"), ("TLS", "TLS")], orig_col]

        # Copy VA and TLS values for output column from reader
        bottom_rows.loc[:, ("OUT", "OUT")] = reader.data.loc[
            [("VA", "VA"), ("TLS", "TLS")], ("OUT", "OUT")
        ]

        # Create output row with values from reader for final demand and output
        output_row = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("OUT", "OUT")], names=["CountryInd", "industryInd"]),
            columns=all_cols,
            dtype=float,
        )
        # Fill intermediate use values
        output_row.loc[("OUT", "OUT"), ordered_int_cols] = output
        # Copy final demand values using the mapping
        for col in ordered_fd_cols:
            orig_col = fd_mapping[col]
            output_row.loc[("OUT", "OUT"), col] = reader.data.loc[("OUT", "OUT"), orig_col]
        # Copy output value from reader
        output_row.loc[("OUT", "OUT"), ("OUT", "OUT")] = reader.data.loc[
            ("OUT", "OUT"), ("OUT", "OUT")
        ]

        # Concatenate all rows in the correct order: regular rows, VA, TLS, OUT
        data = pd.concat([data, bottom_rows, output_row])

        data, countries, industries = ICIOReader._clean_raw_data(data)

        reader = ICIOReader(data=data, countries=countries, industries=industries)

        new_reader = ICIOReader._aggregate_reader(
            reader,
            industry_aggregation=ICIO_ALL,
        )
        data = new_reader.data

        no_va = data.loc[("VA", "Value Added"), new_reader.industry_column_filter] == 0
        no_va_industries = set(no_va[no_va].index)

        output = data.loc[new_reader.industry_index_filter, ("OUT", "OUT")]
        positive_output = set(output[output > 0].index)

        industries_to_readjust = list(no_va_industries.intersection(positive_output))

        data.loc[("VA", "Value Added"), industries_to_readjust] = data.loc[
            ("TLS", "Taxes Less Subsidies"), industries_to_readjust
        ]

        data.loc[("TLS", "Taxes Less Subsidies"), industries_to_readjust] = 0

        data.rename(index={"OUT": "Output"}, level=1, inplace=True)
        data.rename(columns={"OUT": "Output"}, level=1, inplace=True)

        data.rename(index={"OUT": "TOTAL", "VA": "TOTAL", "TLS": "TOTAL"}, level=0, inplace=True)

        return cls(data=data)
