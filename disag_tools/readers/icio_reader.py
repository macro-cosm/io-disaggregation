"""
Module for reading and processing Inter-Country Input-Output (ICIO) tables.

This module provides functionality to read ICIO tables from CSV files and transform
them into a standardized multi-index format suitable for further analysis and
disaggregation. The module handles both single-region and multi-region tables,
with special handling for final demand components and output calculations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ICIOReader:
    """
    A class to read and process Inter-Country Input-Output (ICIO) tables.

    This class handles the reading and processing of ICIO data, providing methods to:
    1. Read data from CSV files with proper handling of multi-region structure
    2. Extract intermediate demand, final demand, and output information
    3. Compute technical coefficients
    4. Handle country and industry aggregation
    5. Validate data consistency

    The class maintains a strict separation between regular industries and special sectors
    (like VA, TLS) to ensure correct calculations. All methods automatically handle this
    separation and work only with regular industries unless explicitly stated otherwise.

    IMPORTANT: The self.industries attribute contains ONLY the regular industry codes,
    with all special sectors (VA, TLS, etc.) already filtered out. When working with
    sectors, ALWAYS use self.industries rather than extracting unique values from
    the data's index or columns, as those may contain special sectors.

    For example:
        # CORRECT way to get sectors:
        sectors = reader.industries

        # INCORRECT way (may include special sectors):
        sectors = list(set(reader.data.index.get_level_values(1)))

    Key Properties:
        - intermediate_demand_table: Get flows between industries
        - final_demand: Get total final demand for each industry
        - technical_coefficients: Get input-output coefficients
        - output_from_out: Get output values from OUT/OUTPUT rows
        - output_from_sums: Calculate output by summing intermediate and final demand

    Attributes:
        data (pd.DataFrame): The processed ICIO table with multi-index structure.
            Both rows and columns are indexed by (country, sector) pairs.
        countries (list[str]): List of unique country codes in the table,
            excluding special elements like VA, TLS.
        industries (list[str]): List of regular industry codes in the table,
            excluding all special sectors. This is the source of truth for
            valid sectors in the data.
        data_path (str | Path | None): Path to the data file, if loaded from file.

    Special Handling:
        - Special sectors (VA, TLS, etc.) are automatically filtered out in calculations
        - Final demand components (HFCE, NPISH, etc.) are handled separately
        - Output can be retrieved from OUT/OUTPUT rows or calculated from sums
        - Country aggregation supports creating ROW (Rest of World) aggregate
        - Industry aggregation preserves special rows/columns
    """

    # Special elements in the ICIO tables
    SPECIAL_ELEMENTS = {
        "VA": "Value added",
        "TLS": "Taxes less subsidies",
        "HFCE": "Household final consumption expenditure",
        "NPISH": "Non-Profit Institutions Serving Households",
        "GGFC": "Government final consumption expenditure",
        "GFCF": "Gross fixed capital formation",
        "INVNT": "Changes in inventories",
        "NONRES": "Direct purchases by non-residents",
        "FD": "Total final demand including discrepancies",
        "DPABR": "Direct purchases abroad by residents",
        "OUT": "Total output",  # Added output as special element
    }

    # Prefixes for special elements in ICIO tables
    SPECIAL_PREFIXES = {
        "VA",
        "VALU",
        "TAXSUB",
        "OUTPUT",
        "TOTAL",
        "TLS",
        "OUT",
        "HFCE",
        "NPISH",
        "GGFC",
        "GFCF",
        "INVNT",
        "NONRES",
        "FD",
        "DPABR",
    }

    # Prefixes for final demand columns
    FINAL_DEMAND_PREFIXES = {
        "HFCE",
        "NPISH",
        "GGFC",
        "GFCF",
        "INVNT",
        "NONRES",
        "FD",
        "DPABR",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        countries: list[str],
        industries: list[str],
    ) -> None:
        """
        Initialize the ICIOReader.

        Args:
            data: The ICIO table as a DataFrame with MultiIndex for both rows and columns
            countries: List of country codes in the data
            industries: List of industry codes in the data

        Raises:
            ValueError: If data format is invalid
        """
        # Validate data structure
        if not isinstance(data.index, pd.MultiIndex) or not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex for both rows and columns")

        if data.index.names != ["CountryInd", "industryInd"] or data.columns.names != [
            "CountryInd",
            "industryInd",
        ]:
            raise ValueError(
                "Index names must be ['CountryInd', 'industryInd'] for both rows and columns"
            )

        # Filter out any columns with ("OUT", x) where x != "OUT" or (x, "OUT") where x != "OUT"
        cols_to_keep = [
            col
            for col in data.columns
            if not (
                (
                    isinstance(col, tuple) and col[0] == "OUT" and col[1] != "OUT"
                )  # Filter ("OUT", x)
                or (
                    isinstance(col, tuple) and col[1] == "OUT" and col[0] != "OUT"
                )  # Filter (x, "OUT")
            )
        ]
        data = data[cols_to_keep]

        self.data = data
        self.countries = countries
        self.industries = industries
        self.data_path: Path | None = None

        initial_shape = self.data.shape

        # Sort the data according to the specified rules
        self.data = self._sort_data(self.data)

        # Validate the data
        self.validate_data()

        final_shape = self.data.shape
        if initial_shape != final_shape:
            raise ValueError(
                f"Data shape changed from {initial_shape} to {final_shape} after sorting"
            )

    def _sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the data according to the specified rules.

        For columns:
        - Within a country, first the industries and at the end the final demand columns
        - Countries are sorted alphabetically
        - Output is at the end

        For indices:
        - Countries first, sorted alphabetically
        - At the end all the non country rows, ordered first as VA, then TLS, and the bottom row is OUT

        Args:
            data: DataFrame to sort

        Returns:
            DataFrame with sorted indices and columns
        """
        # First sort the columns
        # Get all column tuples and separate them into categories
        industry_cols = []
        fd_cols = []
        out_cols = []

        for col in data.columns:
            country, sector = col
            if sector == "OUT":
                out_cols.append(col)
            elif any(sector.startswith(prefix) for prefix in self.FINAL_DEMAND_PREFIXES):
                fd_cols.append(col)
            else:
                industry_cols.append(col)

        # Sort industries within each country
        industry_cols.sort(key=lambda x: (x[0], x[1]))  # Sort by country, then industry
        fd_cols.sort(key=lambda x: (x[0], x[1]))  # Sort by country, then FD type

        # Combine in the correct order: industries, final demand, output
        sorted_cols = industry_cols + fd_cols + out_cols

        # Now sort the indices
        # Get all index tuples and separate them into categories
        country_rows = []
        va_rows = []
        tls_rows = []
        out_rows = []

        for idx in data.index:
            country, sector = idx
            if country == "VA":
                va_rows.append(idx)
            elif country == "TLS":
                tls_rows.append(idx)
            elif country == "OUT":
                out_rows.append(idx)
            else:
                country_rows.append(idx)

        # Sort countries alphabetically and industries within countries
        country_rows.sort(key=lambda x: (x[0], x[1]))

        # Combine in the correct order: countries, VA, TLS, OUT
        sorted_idx = country_rows + va_rows + tls_rows + out_rows

        # Return reindexed DataFrame
        return data.reindex(index=sorted_idx, columns=sorted_cols)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ICIOReader":
        """
        Create an ICIOReader from a CSV file.

        This method reads an ICIO table from a CSV file and processes it into the
        standardized format. The input CSV must have:
        1. A multi-index header with country and industry codes
        2. A multi-index row index with country and industry codes
        3. Both intermediate demand and final demand components

        The method automatically:
        - Identifies and separates regular industries from special sectors
        - Validates the data structure and contents
        - Sets up proper indexing for both rows and columns
        - Handles both single-region and multi-region tables

        Args:
            csv_path: Path to the CSV file containing the ICIO table

        Returns:
            ICIOReader: Initialized reader with processed data

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file format is invalid or data is corrupted
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        # Read the CSV file
        try:
            raw_data = pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1])
        except Exception as e:
            raise ValueError(f"Invalid file format: {e}")

        # Clean up the data
        # First, ensure the index and columns have the correct names
        raw_data.index.names = ["CountryInd", "industryInd"]
        raw_data.columns.names = ["CountryInd", "industryInd"]

        # Extract countries and industries, excluding special elements
        countries = sorted(
            {
                idx[0]
                for idx in raw_data.index
                if not any(idx[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
            }
        )
        # For industries, exclude both special country prefixes and special industry codes
        industries = sorted(
            {
                idx[1]
                for idx in raw_data.index
                if not any(idx[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
                and not any(idx[1].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
                and idx[1] not in cls.SPECIAL_ELEMENTS
            }
        )

        # Create instance with the full data
        reader = cls(raw_data, countries, industries)
        reader.data_path = csv_path
        return reader

    @classmethod
    def from_csv_selection(
        cls,
        csv_path: str | Path,
        selected_countries: list[str] | None = None,
    ) -> "ICIOReader":
        """
        Create an ICIOReader from a CSV file with selected countries.

        This method allows you to work with a subset of countries while aggregating
        all others into a "Rest of World" (ROW) region. This is useful for:
        1. Reducing the size of large ICIO tables
        2. Focusing analysis on specific countries
        3. Creating custom regional aggregations

        The aggregation process:
        1. Reads the full ICIO table
        2. Identifies countries to keep separate
        3. Aggregates all other countries into ROW
        4. Preserves special sectors and final demand components
        5. Maintains consistency in the aggregated data

        Args:
            csv_path: Path to the CSV file containing the ICIO table
            selected_countries: List of country codes to keep separate. All other
                countries will be aggregated into a "ROW" (Rest of World) region.
                If None, returns the full table without aggregation.

        Returns:
            ICIOReader: A new reader with the selected countries separate and
                others aggregated to ROW

        Raises:
            ValueError: If any selected country code is invalid
            FileNotFoundError: If the CSV file doesn't exist
        """
        # First read the full data
        reader = cls.from_csv(csv_path)

        if selected_countries is None:
            return reader

        # Validate country selection
        invalid_countries = set(selected_countries) - set(reader.countries)
        if invalid_countries:
            raise ValueError(f"Invalid country codes in selection: {invalid_countries}")

        # Create mapping for countries not in selected_countries
        country_mapping = {
            country: "ROW" if country not in selected_countries else country
            for country in reader.countries
        }

        # Find all special elements in both index and columns
        special_elements = {
            idx[0]
            for idx in reader.data.index
            if any(idx[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
        } | {
            col[0]
            for col in reader.data.columns
            if any(col[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
        }

        # Add them to the mapping
        for special in special_elements:
            country_mapping[special] = special

        # Create a copy of the data and set index names
        data = reader.data.copy()
        data.index.names = ["CountryInd", "industryInd"]
        data.columns.names = ["CountryCol", "industryCol"]

        # Stack to long format for aggregation, using new implementation
        stacked = (
            data.stack(level=0, future_stack=True)
            .stack(level=0, future_stack=True)
            .to_frame("Value")
        )
        stacked.index.names = ["CountryInd", "industryInd", "CountryCol", "industryCol"]

        # Map countries using the mapping
        stacked.reset_index(inplace=True)
        stacked["CountryInd"] = stacked["CountryInd"].map(country_mapping)
        stacked["CountryCol"] = stacked["CountryCol"].map(country_mapping)

        # Group and aggregate
        grouped = stacked.groupby(["CountryInd", "industryInd", "CountryCol", "industryCol"])[
            "Value"
        ].sum()

        # Unstack back to wide format
        aggregated = grouped.unstack(level=["CountryCol", "industryCol"])

        # Fix index names
        aggregated.columns.names = ["CountryInd", "industryInd"]
        aggregated.index.names = ["CountryInd", "industryInd"]

        # Get new list of countries (selected + ROW if any countries were aggregated)
        new_countries = sorted(set(selected_countries) | {"ROW"})

        # Create new reader with aggregated data
        new_reader = cls(aggregated, new_countries, reader.industries)
        new_reader.data_path = reader.data_path  # Preserve data path
        return new_reader

    @classmethod
    def from_csv_with_aggregation(
        cls,
        csv_path: str | Path,
        selected_countries: list[str] | None = None,
        industry_aggregation: dict[str, list[str]] | None = None,
    ) -> "ICIOReader":
        """
        Create an ICIOReader with both country and industry aggregation.

        This method provides the most flexible way to aggregate an ICIO table,
        allowing both country and industry aggregation in a single step. This
        is particularly useful for:
        1. Creating custom regional and sectoral aggregations
        2. Reducing table dimensions for analysis
        3. Matching different classification schemes

        The aggregation process:
        1. Reads the full ICIO table
        2. Applies country aggregation if specified
           - Keeps selected countries separate
           - Aggregates others into ROW
        3. Applies industry aggregation if specified
           - Combines industries according to the mapping
           - Preserves special sectors
        4. Maintains consistency in the aggregated data

        Args:
            csv_path: Path to the CSV file containing the ICIO table
            selected_countries: List of country codes to keep separate. All other
                countries will be aggregated into a "ROW" (Rest of World) region.
                If None, no country aggregation is performed.
            industry_aggregation: Dictionary mapping new industry codes to lists
                of original industry codes to combine. For example:
                {"AGR": ["A01", "A02", "A03"]} would combine the agricultural
                sectors into a single AGR sector. If None, no industry
                aggregation is performed.

        Returns:
            ICIOReader: A new reader with the specified aggregations applied

        Raises:
            ValueError: If any country or industry code is invalid
            FileNotFoundError: If the CSV file doesn't exist
        """
        # First read the data normally
        reader = cls.from_csv(csv_path)

        # If no aggregation is needed, return as is
        if selected_countries is None and industry_aggregation is None:
            return reader

        # Create country mapping if needed
        country_mapping = None
        if selected_countries is not None:
            # Validate country selection
            invalid_countries = set(selected_countries) - set(reader.countries)
            if invalid_countries:
                raise ValueError(f"Invalid country codes in selection: {invalid_countries}")

            # Create mapping for countries not in selected_countries
            country_mapping = {
                country: "ROW" if country not in selected_countries else country
                for country in reader.countries
            }

            # Find all special elements in both index and columns
            special_elements = {
                idx[0]
                for idx in reader.data.index
                if any(idx[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
            } | {
                col[0]
                for col in reader.data.columns
                if any(col[0].startswith(prefix) for prefix in cls.SPECIAL_PREFIXES)
            }

            # Add them to the mapping
            for special in special_elements:
                country_mapping[special] = special

        # Validate industry aggregation if specified
        if industry_aggregation is not None:
            all_industries = set(reader.industries)
            for new_ind, old_inds in industry_aggregation.items():
                invalid_inds = set(old_inds) - all_industries
                if invalid_inds:
                    raise ValueError(
                        f"Invalid industry codes in aggregation mapping: {invalid_inds}"
                    )

        # Aggregate the data using the mappings
        data = cls._aggregate_data(
            reader.data,
            country_mapping=country_mapping,
            industry_mapping=industry_aggregation,
        )

        # Get new list of countries (selected + ROW if any countries were aggregated)
        if selected_countries is not None:
            countries = sorted(set(selected_countries) | {"ROW"})
        else:
            countries = reader.countries

        # Update industries list based on aggregation
        if industry_aggregation is not None:
            # Start with original industries
            industries = set(reader.industries)
            # Remove only the industries that are being aggregated
            aggregated_industries = set()
            for old_inds in industry_aggregation.values():
                aggregated_industries.update(old_inds)
            industries -= aggregated_industries
            # Add new aggregated industries
            industries |= set(industry_aggregation.keys())
            # Convert to sorted list
            industries = sorted(industries)
        else:
            industries = reader.industries

        # Create new reader with aggregated data
        new_reader = cls(data, countries, industries)
        new_reader.data_path = reader.data_path  # Preserve data path
        return new_reader

    @staticmethod
    def _aggregate_data(
        data: pd.DataFrame,
        country_mapping: dict[str, str] | None = None,
        industry_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate data using provided country and/or industry mappings.

        Args:
            data: DataFrame with MultiIndex for both rows and columns
            country_mapping: Optional mapping of original countries to aggregated countries
            industry_mapping: Optional mapping of original industries to aggregated industries

        Returns:
            DataFrame with aggregated data
        """

        if country_mapping is None and industry_mapping is None:
            return data

        # Start with a copy of the data
        data = data.copy()

        # Handle industry aggregation first if specified
        if industry_mapping is not None:
            logger.debug(f"Industry mapping: {industry_mapping}")

            # Create mapping for all industries
            full_mapping = {}
            for new_ind, old_inds in industry_mapping.items():
                for old_ind in old_inds:
                    full_mapping[old_ind] = new_ind

            # Add special handling for OUT and OUTPUT rows/columns
            full_mapping["OUT"] = "OUT"
            full_mapping["OUTPUT"] = "OUTPUT"
            full_mapping["TOTAL"] = "TOTAL"

            logger.debug(f"Full industry mapping: {full_mapping}")

            # Create new index and columns with aggregated industries
            def map_index(x):
                # Special handling for OUT/OUTPUT/TOTAL rows
                if x[0] in ["OUT", "OUTPUT", "TOTAL"]:
                    return (x[0], x[1])
                return (x[0], full_mapping.get(x[1], x[1]))

            new_index = data.index.map(map_index)
            new_columns = data.columns.map(map_index)

            logger.debug(f"Sample of new index: {list(new_index)[:5]}")
            logger.debug(f"Sample of new columns: {list(new_columns)[:5]}")

            # Create MultiIndex for aggregation
            new_index = pd.MultiIndex.from_tuples(new_index, names=["CountryInd", "industryInd"])
            new_columns = pd.MultiIndex.from_tuples(
                new_columns, names=["CountryInd", "industryInd"]
            )

            # Aggregate the data using transpose for column aggregation
            data = data.groupby(new_index).sum().T.groupby(new_columns).sum().T

            # Ensure MultiIndex is preserved
            data.index = pd.MultiIndex.from_tuples(data.index, names=["CountryInd", "industryInd"])
            data.columns = pd.MultiIndex.from_tuples(
                data.columns, names=["CountryInd", "industryInd"]
            )

            logger.debug(f"After industry aggregation shape: {data.shape}")
            logger.debug(f"After industry aggregation index levels: {data.index.levels}")
            logger.debug(f"After industry aggregation columns levels: {data.columns.levels}")

        # Handle country aggregation if specified
        if country_mapping is not None:
            logger.debug(f"Country mapping: {country_mapping}")

            # Create new index and columns with aggregated countries
            def map_country(x):
                # Don't map special rows like OUT/OUTPUT/TOTAL
                if x[0] in ["OUT", "OUTPUT", "TOTAL"]:
                    return x
                return (country_mapping.get(x[0], x[0]), x[1])

            new_index = data.index.map(map_country)
            new_columns = data.columns.map(map_country)

            logger.debug(f"Sample of new index after country mapping: {list(new_index)[:5]}")
            logger.debug(f"Sample of new columns after country mapping: {list(new_columns)[:5]}")

            # Create MultiIndex for aggregation
            new_index = pd.MultiIndex.from_tuples(new_index, names=["CountryInd", "industryInd"])
            new_columns = pd.MultiIndex.from_tuples(
                new_columns, names=["CountryInd", "industryInd"]
            )

            # Aggregate the data using transpose for column aggregation
            data = data.groupby(new_index).sum().T.groupby(new_columns).sum().T

            # Ensure MultiIndex is preserved
            data.index = pd.MultiIndex.from_tuples(data.index, names=["CountryInd", "industryInd"])
            data.columns = pd.MultiIndex.from_tuples(
                data.columns, names=["CountryInd", "industryInd"]
            )

            logger.debug(f"After country aggregation shape: {data.shape}")
            logger.debug(f"After country aggregation index levels: {data.index.levels}")
            logger.debug(f"After country aggregation columns levels: {data.columns.levels}")

        return data

    def validate_data(self) -> bool:
        """
        Validate the data structure and contents.

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails
        """
        # Check for missing values
        if self.data.isna().any().any():
            raise ValueError("Data contains missing values")

        # Check for infinite values
        if not np.isfinite(self.data.values).all():
            raise ValueError("Data contains infinite values")

        # Check for negative values (warning only)
        if (self.data < 0).any().any():
            print("Warning: Data contains negative values")

        return True

    @property
    def output_from_out(self) -> pd.Series:
        """
        Get output values from the OUT/OUTPUT row of the ICIO table.

        This method attempts to retrieve output values from dedicated output rows
        in the following order:
        1. Tries the "OUT" row first
        2. Falls back to "OUTPUT" row if "OUT" is not found
        3. Raises an error if neither is found

        The output values are essential for:
        - Computing technical coefficients
        - Validating table consistency
        - Analyzing sector sizes and importance

        Returns:
            pd.Series: Output values for each country-industry pair, indexed by
                a MultiIndex of (country, industry). Only regular industries
                are included (no special sectors).

        Raises:
            ValueError: If output values cannot be found in either OUT or
                OUTPUT rows
        """
        # Create valid pairs for regular countries
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )

        try:
            # First try the OUT row
            output_values = pd.Series(
                [
                    self.data.loc[("OUT", "OUT"), (country, industry)]
                    for country, industry in valid_pairs
                ],
                index=valid_pairs,
                name="Output",
            )
            return output_values
        except KeyError:
            # If that fails, try the OUTPUT row
            try:
                output_values = pd.Series(
                    [
                        self.data.loc[("OUTPUT", "OUTPUT"), (country, industry)]
                        for country, industry in valid_pairs
                    ],
                    index=valid_pairs,
                    name="Output",
                )
                return output_values
            except KeyError:
                raise ValueError("Could not find output values in OUT or OUTPUT rows")

    @property
    def output_from_sums(self) -> pd.Series:
        """
        Calculate total output by summing intermediate and final demand.

        This method provides an alternative way to compute output values by
        summing all uses of each sector's production:
        1. Intermediate demand (flows to other industries)
        2. Final demand (all components like HFCE, GGFC, etc.)

        This is useful for:
        - Validating output values from OUT/OUTPUT rows
        - Computing output when OUT/OUTPUT rows are missing
        - Checking table consistency

        The calculation:
        total_output[i] = sum(z[i,j] for all j) + sum(fd[i] for all fd_components)
        where:
        - z[i,j] is the flow from sector i to sector j
        - fd[i] is final demand for sector i's output

        Returns:
            pd.Series: Computed output values for each country-industry pair,
                indexed by a MultiIndex of (country, industry). Only regular
                industries are included (no special sectors).
        """
        logger.debug("Calculating output from sums")

        # Create valid pairs excluding special sectors
        regular_sectors = [s for s in self.industries if s not in ["VA", "TLS"]]
        valid_pairs = [(c, s) for c in self.countries for s in regular_sectors]
        valid_pairs = pd.MultiIndex.from_tuples(valid_pairs, names=["CountryInd", "industryInd"])

        logger.debug(f"Created {len(valid_pairs)} valid pairs")
        logger.debug(f"First few valid pairs: {list(valid_pairs[:5])}")

        # Calculate intermediate demand (sum across columns)
        intermediate = self.data.loc[valid_pairs, valid_pairs].sum(axis=1)
        logger.debug(f"Intermediate demand shape: {intermediate.shape}")

        # Calculate final demand (sum across all final demand columns)
        final = self.final_demand
        logger.debug(f"Final demand shape: {final.shape}")

        # Total output is sum of intermediate and final demand
        total_output = intermediate + final
        logger.debug(f"Total output shape: {total_output.shape}")

        return total_output

    @property
    def intermediate_demand_table(self) -> pd.DataFrame:
        """
        Get the intermediate demand table (flows between industries).

        This method extracts the core inter-industry flows from the ICIO table,
        excluding all special sectors and final demand components. The resulting
        table shows how much of each industry's output is used as input by
        other industries.

        Special handling:
        - Excludes all special sectors (VA, TLS, etc.)
        - Excludes all final demand components
        - Works with both single-region and multi-region tables
        - Validates data for missing or infinite values

        Returns:
            pd.DataFrame: Intermediate demand table where:
                - Rows: Supplying industries (outputs)
                - Columns: Using industries (inputs)
                - Values: Flow values from row industry to column industry
                - Index/Columns: MultiIndex of (country, industry) pairs

        Raises:
            ValueError: If the data contains missing or infinite values
        """
        logger = logging.getLogger(__name__)
        logger.debug("Getting intermediate demand table")

        # Filter out special sectors
        regular_countries = [
            c
            for c in self.countries
            if not any(c.startswith(prefix) for prefix in self.SPECIAL_PREFIXES)
        ]
        regular_industries = [
            i
            for i in self.industries
            if not any(i.startswith(prefix) for prefix in self.SPECIAL_PREFIXES)
            and i not in self.SPECIAL_ELEMENTS
        ]
        logger.debug(f"Regular countries: {regular_countries}")
        logger.debug(f"Regular industries: {regular_industries}")

        # Create valid pairs for regular countries and industries
        valid_pairs = pd.MultiIndex.from_product(
            [regular_countries, regular_industries], names=["CountryInd", "industryInd"]
        )
        logger.debug(f"Created {len(valid_pairs)} valid pairs")

        # Get the intermediate demand table
        table = self.data.loc[valid_pairs, valid_pairs].copy()
        logger.debug(f"Extracted table with shape {table.shape}")

        # Validate the data
        if table.isna().any().any():
            raise ValueError("Intermediate demand table contains missing values")
        if not np.isfinite(table.values).all():
            raise ValueError("Intermediate demand table contains infinite values")

        return table

    @property
    def final_demand_table(self) -> pd.DataFrame:
        """
        Get the complete final demand table with all components.

        This method extracts all final demand components from the ICIO table,
        maintaining the separation between different types of final demand
        (HFCE, NPISH, GGFC, etc.). This is useful for:
        - Analyzing the composition of final demand
        - Studying consumption patterns
        - Computing detailed multipliers

        The table includes all final demand components:
        - HFCE: Household final consumption expenditure
        - NPISH: Non-profit institutions serving households
        - GGFC: Government final consumption expenditure
        - GFCF: Gross fixed capital formation
        - INVNT: Changes in inventories
        - Others as defined in FINAL_DEMAND_PREFIXES

        Returns:
            pd.DataFrame: Final demand table where:
                - Rows: Supplying industries
                - Columns: Final demand components by country
                - Values: Final demand values
                - Row Index: MultiIndex of (country, industry) pairs
                - Column Index: MultiIndex of (country, final_demand_type)
        """
        logger.debug("Calculating total final demand")

        # Create valid pairs from regular sectors only
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )
        logger.debug(f"Created {len(valid_pairs)} valid pairs for final demand")

        # Get final demand columns
        fd_cols = [
            col
            for col in self.data.columns.get_level_values(1).unique()
            if any(col.startswith(p) for p in self.FINAL_DEMAND_PREFIXES)
        ]
        logger.debug(f"Found final demand columns: {fd_cols}")

        # Create column index for final demand
        fd_idx = pd.MultiIndex.from_product(
            [self.countries, fd_cols], names=["CountryInd", "industryInd"]
        )
        logger.debug(f"Created final demand column index with {len(fd_idx)} pairs")

        # Sum across all final demand columns
        final = self.data.loc[valid_pairs, fd_idx]

        return final

    @property
    def final_demand(self) -> pd.Series:
        """
        Get total final demand for each sector.

        This method computes the total final demand by summing across all final
        demand components for each sector. This gives a single value representing
        the total final use of each sector's output.

        The calculation includes:
        1. All final demand components (HFCE, NPISH, GGFC, etc.)
        2. All demanding countries
        3. Only regular sectors (no VA, TLS, etc.)

        This is particularly useful for:
        - Computing output multipliers
        - Analyzing sector dependencies
        - Checking table balances

        Returns:
            pd.Series: Total final demand where:
                - Index: MultiIndex of (country, industry) pairs
                - Values: Sum of all final demand components
                - Only regular industries included (no special sectors)
        """
        logger.debug("Calculating total final demand")

        # Create valid pairs from regular sectors only
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )
        logger.debug(f"Created {len(valid_pairs)} valid pairs for final demand")

        # Get final demand columns
        fd_cols = [
            col
            for col in self.data.columns.get_level_values(1).unique()
            if any(col.startswith(p) for p in self.FINAL_DEMAND_PREFIXES)
        ]
        logger.debug(f"Found final demand columns: {fd_cols}")

        # Create column index for final demand
        fd_idx = pd.MultiIndex.from_product(
            [self.countries, fd_cols], names=["CountryInd", "industryInd"]
        )
        logger.debug(f"Created final demand column index with {len(fd_idx)} pairs")

        # Sum across all final demand columns
        final = self.data.loc[valid_pairs, fd_idx].sum(axis=1)
        logger.debug(f"Final demand shape: {final.shape}")

        return final

    @property
    def intermediate_consumption(self) -> pd.Series:
        """
        Get total intermediate consumption for each sector.

        This method computes the total intermediate use of each sector's output
        by summing across all using industries. This represents how much of
        each sector's output is used as input by other sectors.

        The calculation:
        - Sums across all columns of the intermediate demand table
        - Excludes final demand components
        - Excludes special sectors

        This is useful for:
        - Analyzing supply chain dependencies
        - Computing forward linkages
        - Checking table balances

        Returns:
            pd.Series: Total intermediate consumption where:
                - Index: MultiIndex of (country, industry) pairs
                - Values: Sum of all intermediate uses
                - Only regular industries included (no special sectors)
        """
        return self.intermediate_demand_table.sum(axis=1)

    @property
    def technical_coefficients(self) -> pd.DataFrame:
        """
        Compute the technical coefficients matrix.

        The technical coefficients matrix A shows the input requirements per unit
        of output. Each entry a[i,j] represents the amount of input from sector i
        needed to produce one unit of output in sector j.

        Calculation:
        a[i,j] = z[i,j] / output[j]
        where:
        - z[i,j] is the flow from sector i to sector j
        - output[j] is the total output of sector j

        Special handling:
        - If output[j] = 0, all coefficients in column j are set to 0
        - Special sectors (VA, TLS, etc.) are excluded
        - Only regular industries are included in both rows and columns

        Returns:
            pd.DataFrame: Technical coefficients matrix with:
                - Rows: Supplying sectors (inputs)
                - Columns: Using sectors (outputs)
                - Values: Input requirements per unit of output
                - Index/Columns: MultiIndex of (country, industry) pairs
        """
        # Get intermediate flows and output
        z = self.intermediate_demand_table
        output = self.output_from_out

        # Compute coefficients by dividing each column by its industry's output
        coefficients = z.copy()
        for col in z.columns:
            out = output[col]
            if out > 0:
                coefficients[col] = z[col] / out
            else:
                coefficients[col] = 0.0

        return coefficients

    @staticmethod
    def load_industry_list() -> list[str]:
        """
        Load the list of valid ICIO industries from the yaml file.

        This method reads the standard list of ICIO industry codes from a
        configuration file. This ensures consistency in industry codes across
        different ICIO tables and provides a reference for valid codes.

        The industry list:
        - Contains all standard ICIO industry codes
        - Excludes special sectors (VA, TLS, etc.)
        - Is sorted alphabetically
        - Is used for validation and reference

        Returns:
            list[str]: Sorted list of valid industry codes that can appear
                in ICIO tables

        Raises:
            FileNotFoundError: If the industry list file is not found
            yaml.YAMLError: If the file is not valid YAML
        """
        industry_file = Path(__file__).parent / "data" / "industry_list.yaml"
        try:
            with open(industry_file, "r") as f:
                industry_data = yaml.safe_load(f)
            return sorted(industry_data["industries"])
        except FileNotFoundError:
            logger.warning(f"Industry list file not found at {industry_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading industry list: {str(e)}")
            return []
