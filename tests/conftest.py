"""Fixtures for testing the multi-sector disaggregation package."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from disag_tools.readers import ICIOReader

# Sample data for testing with a small, known dataset
SAMPLE_DATA = """CountryCol,,USA,USA,CHN,CHN,ROW,ROW
industryCol,,AGR,MFG,AGR,MFG,AGR,MFG
CountryInd,industryInd,,,,,,
USA,AGR,10,20,5,15,8,12
USA,MFG,30,40,25,35,15,25
CHN,AGR,5,15,10,20,6,14
CHN,MFG,25,35,30,40,20,30
ROW,AGR,8,12,6,14,10,20
ROW,MFG,15,25,20,30,30,40
OUT,OUT,70,170,70,180,70,140"""


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def sector_config_path(data_dir: Path) -> Path:
    """Get the path to the sector disaggregation config file."""
    return data_dir / "sector_disagg_example.yaml"


@pytest.fixture(scope="session")
def country_config_path(data_dir: Path) -> Path:
    """Get the path to the country disaggregation config file."""
    return data_dir / "country_disagg_example.yaml"


@pytest.fixture(scope="session")
def sample_reader() -> ICIOReader:
    """
    Get an ICIOReader instance initialized with sample data.

    This fixture provides a reader with a small, known dataset
    that can be used to test basic functionality and edge cases.

    Returns:
        ICIOReader: Reader initialized with sample data
    """
    # Create temporary file with sample data
    tmp_path = Path("tmp_sample.csv")
    try:
        with open(tmp_path, "w") as f:
            f.write(SAMPLE_DATA)
        return ICIOReader.from_csv(tmp_path)
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.fixture(scope="session")
def icio_csv_path() -> Path:
    """
    Get the path to the full ICIO CSV file.

    This fixture provides the path to the actual ICIO table used for
    integration testing with real data.

    Returns:
        Path: Path to the ICIO CSV file
    """
    test_dir = Path(__file__).parent
    csv_path = test_dir.parent / "data" / "2021_SML_P.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ICIO CSV file not found at {csv_path}. "
            "Please ensure the data file is present in the data directory."
        )

    return csv_path


@pytest.fixture(scope="session")
def icio_reader(icio_csv_path: Path) -> ICIOReader:
    """
    Get an ICIOReader instance initialized with the full ICIO data.

    This fixture provides a reader initialized with the actual ICIO table
    for integration testing with real data.

    Args:
        icio_csv_path: Path to the ICIO CSV file (from fixture)

    Returns:
        ICIOReader: Reader initialized with full ICIO data
    """
    return ICIOReader.from_csv(icio_csv_path)


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    """
    Create a sample CSV file for testing.

    This fixture creates a temporary CSV file containing sample ICIO data
    that can be used for testing the reader functionality.

    Returns:
        Path: Path to the temporary CSV file
    """
    tmp_path = tmp_path_factory.mktemp("data")
    csv_path = tmp_path / "test_icio.csv"
    with open(csv_path, "w", newline="") as f:  # Use newline="" to ensure consistent line endings
        f.write(SAMPLE_DATA)
    return csv_path


@pytest.fixture(scope="session")
def usa_reader(icio_reader: ICIOReader) -> ICIOReader:
    """
    Get a USA-only reader with original sectors (A01 and A03 separate).

    This fixture provides a reader with only USA data, used for testing
    the disaggregation machinery with known aggregation cases. All other
    countries are automatically aggregated into ROW.

    Args:
        icio_reader: Full ICIO reader (from fixture)

    Returns:
        ICIOReader: USA-only reader with original sectors and ROW
    """
    reader = ICIOReader.from_csv_selection(icio_reader.data_path, selected_countries=["USA"])
    # Verify ROW is present
    countries_in_index = reader.data.index.get_level_values(0).unique()
    countries_in_cols = reader.data.columns.get_level_values(0).unique()
    logger = logging.getLogger(__name__)
    logger.debug(f"Countries in index: {countries_in_index}")
    logger.debug(f"Countries in columns: {countries_in_cols}")
    assert "ROW" in countries_in_index, "ROW should be present in index"
    assert "ROW" in countries_in_cols, "ROW should be present in columns"
    return reader


@pytest.fixture(scope="session")
def usa_aggregated_reader(icio_reader: ICIOReader) -> ICIOReader:
    """
    Get a USA-only reader with A01 and A03 aggregated into sector "A".

    This fixture provides a reader with aggregated data that matches
    the test case in the disaggregation plan. All non-USA countries
    are automatically aggregated into ROW.

    Args:
        icio_reader: Full ICIO reader (from fixture)

    Returns:
        ICIOReader: USA-only reader with aggregated sectors and ROW
    """
    reader = ICIOReader.from_csv_with_aggregation(
        icio_reader.data_path,
        selected_countries=["USA"],
        industry_aggregation={"A": ["A01", "A03"]},
    )
    # Verify ROW is present
    countries_in_index = reader.data.index.get_level_values(0).unique()
    countries_in_cols = reader.data.columns.get_level_values(0).unique()
    logger = logging.getLogger(__name__)
    logger.debug(f"Countries in index: {countries_in_index}")
    logger.debug(f"Countries in columns: {countries_in_cols}")
    assert "ROW" in countries_in_index, "ROW should be present in index"
    assert "ROW" in countries_in_cols, "ROW should be present in columns"
    return reader
