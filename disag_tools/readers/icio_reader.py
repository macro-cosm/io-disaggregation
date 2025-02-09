"""
Module for reading and processing Inter-Country Input-Output (ICIO) tables.

This module provides functionality to read ICIO tables from CSV files and transform
them into a standardized multi-index format suitable for further analysis and
disaggregation.
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

    This class handles the reading of ICIO data from CSV files and transforms it
    into a multi-index DataFrame where both rows and columns are indexed by
    (country, sector) pairs.

    Attributes:
        data (pd.DataFrame): The processed ICIO table with multi-index structure
        countries (list[str]): List of unique country codes in the table
        industries (list[str]): List of unique industry codes in the table
        data_path (str | Path | None): Path to the data file, if loaded from file
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

        self.data = data
        self.countries = countries
        self.industries = industries
        self.data_path: Path | None = None

        # Validate the data
        self.validate_data()

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ICIOReader":
        """
        Create an ICIOReader from a CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            ICIOReader: Initialized reader

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
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

        # Get unique countries and industries, excluding special elements
        special_prefixes = {
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

        # Extract countries and industries, excluding special elements
        countries = sorted(
            {
                idx[0]
                for idx in raw_data.index
                if not any(idx[0].startswith(prefix) for prefix in special_prefixes)
            }
        )
        industries = sorted(
            {
                idx[1]
                for idx in raw_data.index
                if not any(idx[0].startswith(prefix) for prefix in special_prefixes)
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

        Args:
            csv_path: Path to the CSV file
            selected_countries: List of countries to keep separate (others will be aggregated to ROW)

        Returns:
            A new ICIOReader with aggregated data

        Raises:
            ValueError: If any selected country is invalid
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

        # Add special elements to the mapping (they map to themselves)
        special_prefixes = {
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

        # Find all special elements in both index and columns
        special_elements = {
            idx[0]
            for idx in reader.data.index
            if any(idx[0].startswith(prefix) for prefix in special_prefixes)
        } | {
            col[0]
            for col in reader.data.columns
            if any(col[0].startswith(prefix) for prefix in special_prefixes)
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
        Get output values from the OUT row.

        Returns:
            pd.Series: Output values indexed by country-industry pairs

        Raises:
            ValueError: If output values cannot be retrieved
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
        Get output values by summing intermediate consumption and final demand.

        Returns:
            pd.Series: Output values indexed by country-industry pairs
        """
        # Create valid pairs for regular countries
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )

        # Get intermediate consumption (sum across regular country-industry columns)
        intermediate = self.data.loc[valid_pairs, valid_pairs].sum(axis=1)

        # Get final demand (sum across final demand columns)
        final_demand_columns = ["GFCF", "GGFC", "DPABR", "HFCE", "INVNT", "NPISH"]
        fd_filter = self.data.columns.get_level_values(1).isin(final_demand_columns)
        final_demand = self.data.loc[valid_pairs, fd_filter].sum(axis=1)

        # Total output is intermediate consumption plus final demand
        return intermediate + final_demand

    @property
    def final_demand_table(self) -> pd.DataFrame:
        """
        Get the final demand table (all columns with final demand prefixes).

        Returns:
            pd.DataFrame: Final demand table with country-industry pairs as index
                and final demand categories as columns
        """
        # Create valid pairs for regular countries
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )

        # Get final demand columns (those with special prefixes)
        final_demand_prefixes = {
            "HFCE",
            "NPISH",
            "GGFC",
            "GFCF",
            "INVNT",
            "NONRES",
            "FD",
            "DPABR",
        }
        final_demand_cols = [
            col
            for col in self.data.columns
            if any(col[0].startswith(prefix) for prefix in final_demand_prefixes)
        ]

        # Return the final demand table
        return self.data.loc[valid_pairs, final_demand_cols]

    @property
    def final_demand_totals(self) -> pd.Series:
        """
        Get total final demand values (sum of all final demand columns).

        Returns:
            pd.Series: Total final demand values indexed by country-industry pairs
        """
        return self.final_demand_table.sum(axis=1)

    @property
    def intermediate_demand_table(self) -> pd.DataFrame:
        """
        Get the intermediate demand table (only country-industry flows).

        Returns:
            pd.DataFrame: Intermediate demand table with country-industry pairs
                as both index and columns

        Raises:
            ValueError: If the data contains invalid values
        """
        # Create valid pairs for regular countries
        valid_pairs = pd.MultiIndex.from_product(
            [self.countries, self.industries], names=["CountryInd", "industryInd"]
        )

        # Get the intermediate demand table
        table = self.data.loc[valid_pairs, valid_pairs].copy()

        # Validate the data
        if table.isna().any().any():
            raise ValueError("Intermediate demand table contains missing values")
        if not np.isfinite(table.values).all():
            raise ValueError("Intermediate demand table contains infinite values")

        return table

    @property
    def final_demand(self) -> pd.Series:
        """
        Get total final demand values.

        This is an alias for final_demand_totals for backward compatibility.

        Returns:
            pd.Series: Final demand values indexed by country-industry pairs
        """
        return self.final_demand_totals

    @property
    def intermediate_consumption(self) -> pd.Series:
        """
        Get intermediate consumption values.

        Returns:
            pd.Series: Intermediate consumption values indexed by country-industry pairs
        """
        return self.intermediate_demand_table.sum(axis=1)

    @property
    def technical_coefficients(self) -> pd.DataFrame:
        """
        Get the technical coefficients matrix.

        The technical coefficients matrix A is computed such that entry a[i,j] is the flow
        in the intermediate use table from industry i to industry j (z[i,j]) divided by
        the output of industry j: a[i,j] = z[i,j]/output[j].

        Any coefficients that would be infinity (due to zero output) are set to 0.

        Returns:
            pd.DataFrame: Technical coefficients matrix with the same index/columns structure
                as the intermediate demand table
        """
        # Get intermediate flows and output
        z = self.intermediate_demand_table
        output = self.output_from_out

        # Compute coefficients by dividing each column by its industry's output
        coefficients = z.copy()
        for col in z.columns:
            output_val = output[col]
            if output_val > 0:
                coefficients[col] = z[col] / output_val
            else:
                coefficients[col] = 0

        return coefficients

    def get_reordered_technical_coefficients(
        self, sectors_to_disaggregate: list[str]
    ) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]]]:
        """
        Get the technical coefficients matrix reordered for disaggregation.

        This method reorders the technical coefficients matrix so that sectors to be
        disaggregated are moved to the bottom-right blocks. The resulting matrix has
        the following block structure:

        [A_0    B^1  ... B^K]
        [C^1    D^11 ... D^1K]
        [...    ...  ... ...]
        [C^K    D^K1 ... D^KK]

        where:
        - A_0 is the block for undisaggregated sectors
        - B^i are the blocks from undisaggregated to disaggregated sectors
        - C^i are the blocks from disaggregated to undisaggregated sectors
        - D^ij are the blocks between disaggregated sectors

        Args:
            sectors_to_disaggregate: List of sector codes to be disaggregated

        Returns:
            tuple containing:
            - Reordered technical coefficients matrix
            - Dictionary mapping each block name to list of (country, sector) pairs it contains
        """
        # Get technical coefficients
        coefficients = self.technical_coefficients

        # Split sectors into disaggregated and undisaggregated
        undisaggregated_sectors = [s for s in self.industries if s not in sectors_to_disaggregate]

        # Create lists of (country, sector) pairs for each group
        undisaggregated_pairs = [
            (country, sector) for country in self.countries for sector in undisaggregated_sectors
        ]
        disaggregated_pairs = [
            (country, sector) for country in self.countries for sector in sectors_to_disaggregate
        ]

        # Create new order for rows and columns
        new_order = undisaggregated_pairs + disaggregated_pairs

        # Reorder the matrix
        reordered = coefficients.reindex(index=new_order, columns=new_order)

        # Create mapping of blocks to their (country, sector) pairs
        n_undis = len(undisaggregated_pairs)
        blocks = {
            "A_0": undisaggregated_pairs,  # Undisaggregated block
        }

        # Add B^i blocks (undisaggregated to disaggregated)
        for i, sector in enumerate(sectors_to_disaggregate, 1):
            sector_pairs = [(c, sector) for c in self.countries]
            blocks[f"B^{i}"] = sector_pairs

        # Add C^i blocks (disaggregated to undisaggregated)
        for i, sector in enumerate(sectors_to_disaggregate, 1):
            sector_pairs = [(c, sector) for c in self.countries]
            blocks[f"C^{i}"] = sector_pairs

        # Add D^ij blocks (between disaggregated sectors)
        for i, sector_i in enumerate(sectors_to_disaggregate, 1):
            for j, sector_j in enumerate(sectors_to_disaggregate, 1):
                sector_pairs = [(c, sector_j) for c in self.countries]
                blocks[f"D^{i}{j}"] = sector_pairs

        return reordered, blocks

    @staticmethod
    def load_industry_list() -> list[str]:
        """
        Load the list of valid ICIO industries from the yaml file.

        Returns:
            list[str]: List of valid industry codes
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
