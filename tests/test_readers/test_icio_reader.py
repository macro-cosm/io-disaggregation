"""Tests for the ICIO reader module."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)


class TestICIOReader:
    """Test suite for the ICIOReader class."""

    def test_sample_data_structure(self, sample_reader: ICIOReader):
        """Test the basic structure of the reader with sample data."""
        # Check that data was loaded
        assert sample_reader.data is not None
        assert isinstance(sample_reader.data, pd.DataFrame)

        # Check multi-index structure
        assert isinstance(sample_reader.data.index, pd.MultiIndex)
        assert isinstance(sample_reader.data.columns, pd.MultiIndex)

        # Check index names
        assert sample_reader.data.index.names == ["CountryInd", "industryInd"]
        assert sample_reader.data.columns.names == ["CountryInd", "industryInd"]

        # Check unique countries and industries
        assert set(sample_reader.countries) == {"USA", "CHN", "ROW"}
        assert set(sample_reader.industries) == {"AGR", "MFG"}

    def test_sample_data_values(self, sample_reader: ICIOReader):
        """Test specific known values in the sample data."""
        # Check specific values from the sample data
        assert sample_reader.data.loc[("USA", "AGR"), ("USA", "AGR")] == 10.0
        assert sample_reader.data.loc[("CHN", "MFG"), ("USA", "MFG")] == 35.0

        # Check row sums (should match the sample data)
        assert sample_reader.data.loc[("USA", "AGR")].sum() == 70.0  # Updated for ROW columns
        assert sample_reader.data.loc[("CHN", "MFG")].sum() == 180.0  # Updated for ROW columns

    def test_real_data_structure(self, icio_reader: ICIOReader):
        """Test the structure of the reader with real ICIO data."""
        # Check basic data loading
        assert icio_reader.data is not None
        assert isinstance(icio_reader.data, pd.DataFrame)

        # Check multi-index structure
        assert isinstance(icio_reader.data.index, pd.MultiIndex)
        assert isinstance(icio_reader.data.columns, pd.MultiIndex)

        # Check index names
        assert icio_reader.data.index.names == ["CountryInd", "industryInd"]
        assert icio_reader.data.columns.names == ["CountryInd", "industryInd"]

        # Check data is not empty
        assert not icio_reader.data.empty
        assert len(icio_reader.countries) > 0
        assert len(icio_reader.industries) > 0

    def test_real_data_consistency(self, icio_reader: ICIOReader):
        """Test consistency of the real ICIO data."""
        # Check no missing values
        assert not icio_reader.data.isna().any().any()

        # Check all values are finite
        assert np.isfinite(icio_reader.data.values).all()

        # Get regular country indices (excluding special elements)
        row_countries = {
            idx[0] for idx in icio_reader.data.index if idx[0] in icio_reader.countries
        }
        col_countries = {
            idx[0] for idx in icio_reader.data.columns if idx[0] in icio_reader.countries
        }

        # Check regular country indices match
        assert row_countries == col_countries, "Regular country indices should match"

        # Get regular industry indices (excluding special elements)
        row_industries = {
            idx[1] for idx in icio_reader.data.index if idx[0] in icio_reader.countries
        }
        col_industries = {
            idx[1]
            for idx in icio_reader.data.columns
            if idx[0] in icio_reader.countries and idx[1] in icio_reader.industries
        }

        # Check regular industry indices match
        assert row_industries == col_industries, "Regular industry indices should match"

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            ICIOReader.from_csv(Path("nonexistent.csv"))

    def test_invalid_data_format(self, tmp_path):
        """Test handling of invalid data format."""
        # Create invalid CSV file
        invalid_csv = tmp_path / "invalid.csv"
        with open(invalid_csv, "w") as f:
            f.write("invalid,data\n1,2,3")

        with pytest.raises(ValueError):
            ICIOReader.from_csv(invalid_csv)

    def test_constructor(self):
        """Test direct constructor with valid data."""
        # Create sample data
        index = pd.MultiIndex.from_tuples(
            [("USA", "AGR"), ("USA", "MFG")], names=["CountryInd", "industryInd"]
        )
        columns = pd.MultiIndex.from_tuples(
            [("USA", "AGR"), ("USA", "MFG")], names=["CountryInd", "industryInd"]
        )
        data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=columns)

        reader = ICIOReader(data=data, countries=["USA"], industries=["AGR", "MFG"])

        assert reader.data is not None
        assert set(reader.countries) == {"USA"}
        assert set(reader.industries) == {"AGR", "MFG"}

    def test_sample_data_selection(self, sample_csv):
        """Test country selection and aggregation with sample data."""
        # Create reader with only USA selected (CHN and ROW should be aggregated)
        reader = ICIOReader.from_csv_selection(sample_csv, selected_countries=["USA"])

        # Check countries
        assert set(reader.countries) == {"USA", "ROW"}
        assert set(reader.industries) == {"AGR", "MFG"}

        # Check that data structure is correct
        assert reader.data.index.names == ["CountryInd", "industryInd"]
        assert reader.data.columns.names == ["CountryInd", "industryInd"]

        # Check specific aggregation results
        # For USA-AGR to USA-AGR, should be unchanged at 10.0
        assert reader.data.loc[("USA", "AGR"), ("USA", "AGR")] == 10.0

        # For ROW-AGR to USA-AGR, should be sum of CHN-AGR and ROW-AGR to USA-AGR
        # Original: CHN-AGR to USA-AGR = 5.0, ROW-AGR to USA-AGR = 8.0
        assert reader.data.loc[("ROW", "AGR"), ("USA", "AGR")] == 13.0

        # Check that row sums are preserved after aggregation
        usa_agr_sum = reader.data.loc[("USA", "AGR")].sum()
        row_agr_sum = reader.data.loc[("ROW", "AGR")].sum()

        # Original sums from sample data for USA-AGR
        assert usa_agr_sum == 70.0  # 10 + 20 + 5 + 15 + 8 + 12

        # Original sums from sample data for ROW-AGR (CHN-AGR + ROW-AGR)
        # CHN-AGR: 5 + 15 + 10 + 20 + 6 + 14 = 70
        # ROW-AGR: 8 + 12 + 6 + 14 + 10 + 20 = 70
        assert row_agr_sum == 140.0  # 70 + 70

    def test_real_data_selection(self, icio_reader: ICIOReader):
        """Test country selection and aggregation with real ICIO data."""
        # Get original countries for testing
        original_countries = icio_reader.countries[:3]  # Take first 3 countries for test

        # Create reader with selected countries
        selected_reader = ICIOReader.from_csv_selection(
            icio_reader.data_path, selected_countries=original_countries
        )

        # Check basic structure
        assert set(selected_reader.countries) == set(original_countries) | {"ROW"}
        assert selected_reader.industries == icio_reader.industries

        # Check that the data is properly structured
        assert isinstance(selected_reader.data, pd.DataFrame)
        assert isinstance(selected_reader.data.index, pd.MultiIndex)
        assert isinstance(selected_reader.data.columns, pd.MultiIndex)

        # Validate data consistency
        assert not selected_reader.data.isna().any().any()
        assert np.isfinite(selected_reader.data.values).all()

        # Check that row and column sums match original totals
        # The total sum should be the same before and after aggregation
        assert np.isclose(
            selected_reader.data.values.sum(), icio_reader.data.values.sum(), rtol=1e-10
        )

    def test_invalid_country_selection(self, sample_csv):
        """Test handling of invalid country selection."""
        with pytest.raises(ValueError, match="Invalid country codes in selection"):
            ICIOReader.from_csv_selection(sample_csv, selected_countries=["INVALID"])

    def test_invalid_data_structure(self):
        """Test handling of invalid data structure in constructor."""
        # Test with regular DataFrame (no MultiIndex)
        data = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Data must have MultiIndex"):
            ICIOReader(data=data, countries=["USA"], industries=["AGR"])

        # Test with wrong index names
        index = pd.MultiIndex.from_tuples(
            [("USA", "AGR"), ("USA", "MFG")],
            names=["Country", "Industry"],  # Wrong names
        )
        columns = pd.MultiIndex.from_tuples(
            [("USA", "AGR"), ("USA", "MFG")],
            names=["Country", "Industry"],  # Wrong names
        )
        data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=columns)
        with pytest.raises(ValueError, match="Index names must be"):
            ICIOReader(data=data, countries=["USA"], industries=["AGR", "MFG"])

    def test_data_validation(self, sample_reader: ICIOReader):
        """Test data validation checks."""
        # Make a copy of the reader to avoid modifying the original
        reader_copy = ICIOReader(
            data=sample_reader.data.copy(),
            countries=sample_reader.countries.copy(),
            industries=sample_reader.industries.copy(),
        )

        # Test validation of clean data
        assert reader_copy.validate_data() is True

        # Test with negative values
        reader_copy.data.iloc[0, 0] = -1.0
        reader_copy.validate_data()  # Should log warning but not raise

        # Test with missing values
        reader_copy.data.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains missing values"):
            reader_copy.validate_data()

        # Test with infinite values
        reader_copy.data.iloc[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains infinite values"):
            reader_copy.validate_data()

    def test_output_computation(self, icio_reader: ICIOReader):
        """Test that both methods of computing output give consistent results."""
        # Get output using both methods
        output_from_out = icio_reader.output_from_out
        output_from_sums = icio_reader.output_from_sums

        # Check that indices match
        assert output_from_out.index.equals(output_from_sums.index)

        # Check that values are close (using relative tolerance for floating point comparison)

        # drop ("MEX", "C28")  entry
        output_from_out = output_from_out.drop(("MEX", "C28"))
        output_from_sums = output_from_sums.drop(("MEX", "C28"))

        pd.testing.assert_series_equal(
            output_from_out,
            output_from_sums,
            rtol=1e-2,  # Relative tolerance for floating point comparison
            check_names=False,  # Names might differ but values should match
        )

    def test_final_demand_computation(self, sample_reader: ICIOReader):
        """Test computation of final demand using sample data."""
        final_demand = sample_reader.final_demand

        # Check that we get the expected structure
        assert isinstance(final_demand, pd.Series)
        assert final_demand.index.nlevels == 2
        assert set(final_demand.index.get_level_values(0)) == set(sample_reader.countries)
        assert set(final_demand.index.get_level_values(1)) == set(sample_reader.industries)

        # Values should be non-negative for final demand
        assert (final_demand >= 0).all()

    def test_intermediate_consumption_computation(self, sample_reader: ICIOReader):
        """Test computation of intermediate consumption using sample data."""
        intermediate = sample_reader.intermediate_consumption

        # Check that we get the expected structure
        assert isinstance(intermediate, pd.Series)
        assert intermediate.index.nlevels == 2
        assert set(intermediate.index.get_level_values(0)) == set(sample_reader.countries)
        assert set(intermediate.index.get_level_values(1)) == set(sample_reader.industries)

        # Sum of intermediate consumption should match known values from sample data
        # (Add specific value checks based on the sample data)

    def test_final_demand_table_structure(self, sample_reader: ICIOReader):
        """Test the structure of the final demand table."""
        fd_table = sample_reader.final_demand_table

        # Check basic structure
        assert isinstance(fd_table, pd.DataFrame)
        assert isinstance(fd_table.index, pd.MultiIndex)
        assert isinstance(fd_table.columns, pd.MultiIndex)

        # Check index structure (should be country-industry pairs)
        assert fd_table.index.names == ["CountryInd", "industryInd"]
        assert set(fd_table.index.get_level_values(0)) == set(sample_reader.countries)
        assert set(fd_table.index.get_level_values(1)) == set(sample_reader.industries)

        # Check column structure (should be final demand categories)
        expected_fd_categories = {
            "HFCE",
            "NPISH",
            "GGFC",
            "GFCF",
            "INVNT",
            "NONRES",
            "FD",
            "DPABR",
        }
        actual_fd_categories = set(fd_table.columns.get_level_values(1).unique())
        assert actual_fd_categories.issubset(expected_fd_categories), (
            f"Found unexpected final demand categories: "
            f"{actual_fd_categories - expected_fd_categories}"
        )

        # Values should be non-negative
        assert (fd_table >= 0).all().all()

    def test_intermediate_demand_table_structure(self, sample_reader: ICIOReader):
        """Test the structure of the intermediate demand table."""
        int_table = sample_reader.intermediate_demand_table

        # Check basic structure
        assert isinstance(int_table, pd.DataFrame)
        assert isinstance(int_table.index, pd.MultiIndex)
        assert isinstance(int_table.columns, pd.MultiIndex)

        # Check index structure (should be country-industry pairs)
        assert int_table.index.names == ["CountryInd", "industryInd"]
        assert set(int_table.index.get_level_values(0)) == set(sample_reader.countries)
        assert set(int_table.index.get_level_values(1)) == set(sample_reader.industries)

        # Check column structure (should also be country-industry pairs)
        assert int_table.columns.names == ["CountryInd", "industryInd"]
        assert set(int_table.columns.get_level_values(0)) == set(sample_reader.countries)
        assert set(int_table.columns.get_level_values(1)) == set(sample_reader.industries)

        # Check specific values from sample data
        assert int_table.loc[("USA", "AGR"), ("USA", "AGR")] == 10.0
        assert int_table.loc[("CHN", "MFG"), ("USA", "MFG")] == 35.0

        # Values can be negative in intermediate demand
        # But should be finite
        assert np.isfinite(int_table.values).all()

    def test_technical_coefficients(self, sample_reader: ICIOReader):
        """Test computation of technical coefficients matrix."""
        # Get technical coefficients
        tech_coef = sample_reader.technical_coefficients

        # Check basic structure
        assert isinstance(tech_coef, pd.DataFrame)
        assert tech_coef.index.equals(sample_reader.intermediate_demand_table.index)
        assert tech_coef.columns.equals(sample_reader.intermediate_demand_table.columns)

        # Check specific values from sample data
        # For USA-AGR to USA-AGR: flow is 10.0, output is 70.0
        # So coefficient should be 10.0/70.0
        assert np.isclose(tech_coef.loc[("USA", "AGR"), ("USA", "AGR")], 10.0 / 70.0)

        # Check that all values are non-negative
        # (coefficients represent share of input in total output)
        assert (tech_coef >= 0).all().all()

    def test_technical_coefficients_zero_output(self, sample_reader: ICIOReader):
        """Test handling of zero output in technical coefficients' computation."""
        # Create a copy of the reader with modified data
        reader_copy = ICIOReader(
            data=sample_reader.data.copy(),
            countries=sample_reader.countries.copy(),
            industries=sample_reader.industries.copy(),
        )

        # Set output of USA-AGR to 0
        reader_copy.data.loc[("OUT", "OUT"), ("USA", "AGR")] = 0.0

        # Compute technical coefficients
        tech_coef = reader_copy.technical_coefficients

        # Check that coefficients for USA-AGR column are all 0
        assert (tech_coef[("USA", "AGR")] == 0).all()

        # Other coefficients should still be valid
        assert np.isclose(
            tech_coef.loc[("USA", "MFG"), ("USA", "MFG")],
            40.0 / 170.0,  # Original flow divided by original output
        )

    def test_reordered_technical_coefficients(self, sample_reader: ICIOReader):
        """Test reordering of technical coefficients matrix for disaggregation."""
        # Create sectors_info list with tuples (sector_id, name, k)
        sectors_info = [("MFG", "Manufacturing", 3)]  # Using USA as representative country

        # Get reordered coefficients for disaggregating MFG sector
        blocks = sample_reader.get_reordered_technical_coefficients(sectors_info)

        # Check that the matrix has the same values, just reordered
        assert blocks.reordered_matrix.shape == sample_reader.technical_coefficients.shape
        assert set(blocks.reordered_matrix.index) == set(sample_reader.technical_coefficients.index)
        assert set(blocks.reordered_matrix.columns) == set(
            sample_reader.technical_coefficients.columns
        )

        # Check that MFG sectors are at the end
        mfg_pairs = [(c, "MFG") for c in sample_reader.countries]
        assert list(blocks.reordered_matrix.index[-len(mfg_pairs) :]) == mfg_pairs
        assert list(blocks.reordered_matrix.columns[-len(mfg_pairs) :]) == mfg_pairs

        # Check sector info
        assert blocks.K == len(sectors_info) * len(sample_reader.countries)  # One sector being disaggregated
        sector = blocks.get_sector_info(1)
        assert sector.sector == "MFG"
        assert sector.k == 3  # Value we set in sectors_info

        # Check that values match original matrix
        orig_coef = sample_reader.technical_coefficients
        for i in blocks.reordered_matrix.index:
            for j in blocks.reordered_matrix.columns:
                assert np.isclose(blocks.reordered_matrix.loc[i, j], orig_coef.loc[i, j])

    def test_reordered_technical_coefficients_multiple_sectors(self, sample_reader: ICIOReader):
        """Test reordering with multiple sectors to disaggregate."""
        # Create sectors_info list with tuples (sector_id, name, k)
        sectors_info = [
            ("AGR", "Agriculture", 3),
            ("MFG", "Manufacturing", 3),
        ]

        # Try to disaggregate both sectors
        blocks = sample_reader.get_reordered_technical_coefficients(sectors_info)

        # Check basic properties
        assert blocks.K == len(sectors_info) * len(sample_reader.countries)  # Two sectors being disaggregated


        # Check sector info
        sector1 = blocks.get_sector_info(1)
        sector2 = blocks.get_sector_info(2)
        assert {sector1.sector, sector2.sector} == {"AGR", "MFG"}
        assert sector1.k == 3 and sector2.k == 3  # Values we set in sectors_info

        # Check that the matrix contains all pairs
        all_pairs = [
            (country, sector)
            for country in sample_reader.countries
            for sector in sample_reader.industries
        ]
        assert set(blocks.reordered_matrix.index) == set(all_pairs)
        assert set(blocks.reordered_matrix.columns) == set(all_pairs)

    def test_reordered_technical_coefficients_real_data(self, usa_reader: ICIOReader):
        sectors_info = [("A01", "A01", 1),
                        ("A03", "A03", 1)]

        blocks = usa_reader.get_reordered_technical_coefficients(sectors_info)

        assert blocks.K == 4

        # Check sector info
        block_lists = [("ROW", "A01"), ("ROW", "A03"), ("USA", "A01"), ("USA", "A03")]
        block_lists = sorted(block_lists)

        assert list(blocks.reordered_matrix.columns[-blocks.K:]) == block_lists



    def test_real_data_a01_disaggregation(self, icio_reader: ICIOReader):
        """Test reordering of technical coefficients for A01 disaggregation using real ICIO data."""
        # First create a reader with only USA data
        usa_reader = ICIOReader.from_csv_selection(
            icio_reader.data_path, selected_countries=["USA"]
        )

        # Create sectors_info list with tuples (sector_id, name, k)
        sectors_info = [(("A01"), "Agriculture", 3)]

        # Get reordered coefficients for disaggregating A01 sector
        blocks = usa_reader.get_reordered_technical_coefficients(sectors_info)

        # Check basic structure
        assert blocks.reordered_matrix.shape == usa_reader.technical_coefficients.shape
        assert set(blocks.reordered_matrix.index) == set(usa_reader.technical_coefficients.index)
        assert set(blocks.reordered_matrix.columns) == set(
            usa_reader.technical_coefficients.columns
        )

        # Check that A01 pairs are at the end
        last_pairs = [(c, "A01") for c in usa_reader.countries]
        assert list(blocks.reordered_matrix.index[-len(last_pairs) :]) == last_pairs
        assert list(blocks.reordered_matrix.columns[-len(last_pairs) :]) == last_pairs

        # Check sector info
        assert blocks.K == 2  # One sector being disaggregated
        sector = blocks.get_sector_info(1)
        assert sector.sector == "A01"
        assert sector.k == 3  # Value we set in sectors_info

        # Verify some actual values from the technical coefficients
        # Get original coefficients for comparison
        orig_coef = usa_reader.technical_coefficients

        # Check a few specific values from different blocks
        # A_0 block (non-A01 to non-A01)
        non_a01_pair = ("USA", "A03")  # Example non-A01 sector
        assert np.isclose(
            blocks.reordered_matrix.loc[non_a01_pair, non_a01_pair],
            orig_coef.loc[non_a01_pair, non_a01_pair],
        )

        # B^1 block (non-A01 to A01)
        a01_pair = ("USA", "A01")
        assert np.isclose(
            blocks.reordered_matrix.loc[non_a01_pair, a01_pair],
            orig_coef.loc[non_a01_pair, a01_pair],
        )

        # C^1 block (A01 to non-A01)
        assert np.isclose(
            blocks.reordered_matrix.loc[a01_pair, non_a01_pair],
            orig_coef.loc[a01_pair, non_a01_pair],
        )

        # D^11 block (A01 to A01)
        assert np.isclose(
            blocks.reordered_matrix.loc[a01_pair, a01_pair],
            orig_coef.loc[a01_pair, a01_pair],
        )

    def test_industry_aggregation(self, sample_csv):
        """Test industry aggregation with sample data."""
        # Create reader with industry aggregation
        industry_mapping = {"AGR_MFG": ["AGR", "MFG"]}  # Combine AGR and MFG into one sector
        reader = ICIOReader.from_csv_with_aggregation(
            sample_csv, industry_aggregation=industry_mapping
        )

        # Check industries
        assert set(reader.industries) == {"AGR_MFG"}

        # Check that data structure is correct
        assert reader.data.index.names == ["CountryInd", "industryInd"]
        assert reader.data.columns.names == ["CountryInd", "industryInd"]

        # Check specific aggregation results
        # For USA-AGR_MFG to USA-AGR_MFG, should be sum of all AGR and MFG flows
        # Original: USA-AGR to USA-AGR = 10.0, USA-AGR to USA-MFG = 20.0
        #          USA-MFG to USA-AGR = 30.0, USA-MFG to USA-MFG = 40.0
        assert reader.data.loc[("USA", "AGR_MFG"), ("USA", "AGR_MFG")] == 100.0

    def test_combined_aggregation(self, sample_csv):
        """Test combined country and industry aggregation."""
        # Create reader with both country and industry aggregation
        selected_countries = ["USA"]  # Aggregate CHN and ROW into new ROW
        industry_mapping = {"AGR_MFG": ["AGR", "MFG"]}  # Combine AGR and MFG

        reader = ICIOReader.from_csv_with_aggregation(
            sample_csv,
            selected_countries=selected_countries,
            industry_aggregation=industry_mapping,
        )

        # Check countries and industries
        assert set(reader.countries) == {"USA", "ROW"}
        assert set(reader.industries) == {"AGR_MFG"}

        # Check data structure
        assert reader.data.index.names == ["CountryInd", "industryInd"]
        assert reader.data.columns.names == ["CountryInd", "industryInd"]

        # Check specific aggregation results
        # USA-AGR_MFG to USA-AGR_MFG should include all flows between USA sectors
        usa_to_usa = reader.data.loc[("USA", "AGR_MFG"), ("USA", "AGR_MFG")]
        assert usa_to_usa == 100.0  # 10 + 20 + 30 + 40

        # ROW-AGR_MFG to USA-AGR_MFG should include all flows from CHN and ROW to USA
        row_to_usa = reader.data.loc[("ROW", "AGR_MFG"), ("USA", "AGR_MFG")]
        # Original: CHN-AGR to USA-AGR = 5.0, CHN-AGR to USA-MFG = 15.0
        #          CHN-MFG to USA-AGR = 25.0, CHN-MFG to USA-MFG = 35.0
        #          ROW-AGR to USA-AGR = 8.0, ROW-AGR to USA-MFG = 12.0
        #          ROW-MFG to USA-AGR = 15.0, ROW-MFG to USA-MFG = 25.0
        assert row_to_usa == 140.0  # 5 + 15 + 25 + 35 + 8 + 12 + 15 + 25

    def test_invalid_industry_aggregation(self, sample_csv):
        """Test handling of invalid industry aggregation."""
        # Try to aggregate non-existent industry
        industry_mapping = {"NEW": ["INVALID"]}
        with pytest.raises(ValueError, match="Invalid industry codes in aggregation mapping"):
            ICIOReader.from_csv_with_aggregation(sample_csv, industry_aggregation=industry_mapping)
