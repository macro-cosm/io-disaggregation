"""Fixtures for testing the multi-sector disaggregation package."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from disag_tools.configurations import CountryConfig, DisaggregationConfig
from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregatedBlocks,
    DisaggregationBlocks,
    unfold_countries,
)
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
def real_disag_config(data_dir: Path, usa_reader) -> DisaggregationConfig:
    """Create the disaggregation configuration with weights computed from actual data.

    Instead of using hardcoded weights from the YAML file, this computes the weights
    from the actual output values in the disaggregated data.
    """
    with open(data_dir / "test_sector_disagg.yaml") as f:
        config_dict = yaml.safe_load(f)

    # Get the actual output values for A01 and A03
    usa_a01_output = usa_reader.output_from_out.loc[("USA", "A01")]
    usa_a03_output = usa_reader.output_from_out.loc[("USA", "A03")]
    usa_total = usa_a01_output + usa_a03_output

    row_a01_output = usa_reader.output_from_out.loc[("ROW", "A01")]
    row_a03_output = usa_reader.output_from_out.loc[("ROW", "A03")]
    row_total = row_a01_output + row_a03_output

    # Compute relative weights
    config_dict["sectors"]["A"]["subsectors"]["A01"]["relative_output_weights"] = {
        "USA": usa_a01_output / usa_total,
        "ROW": row_a01_output / row_total,
    }
    config_dict["sectors"]["A"]["subsectors"]["A03"]["relative_output_weights"] = {
        "USA": usa_a03_output / usa_total,
        "ROW": row_a03_output / row_total,
    }

    return DisaggregationConfig(**config_dict)


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
def can_reader(data_dir):
    return ICIOReader.from_csv_selection(data_dir / "2021_SML_P.csv", selected_countries=["CAN"])


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


@pytest.fixture(scope="session")
def usa_reader_blocks(usa_reader):
    return DisaggregatedBlocks.from_reader(usa_reader, sector_mapping={"A": ["A01", "A03"]})


@pytest.fixture(scope="session")
def aggregated_blocks(usa_aggregated_reader: ICIOReader):
    """Get the aggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    sectors_info = unfold_countries(usa_aggregated_reader.countries, sectors_mapping)
    # setup blocks
    aggregated_blocks = DisaggregationBlocks.from_technical_coefficients(
        tech_coef=usa_aggregated_reader.technical_coefficients,
        sectors_info=sectors_info,
        output=usa_aggregated_reader.output_from_out,
    )

    return aggregated_blocks


@pytest.fixture(scope="session")
def disaggregated_blocks(usa_reader: ICIOReader):
    """Get the disaggregated blocks for the USA."""
    sectors_mapping = {"A": ["A01", "A03"]}
    # setup blocks
    disaggregated_blocks = DisaggregatedBlocks.from_reader(
        reader=usa_reader, sector_mapping=sectors_mapping
    )

    return disaggregated_blocks


@pytest.fixture(scope="function")
def default_problem(real_disag_config, usa_aggregated_reader):
    """Get a default DisaggregationProblem instance for testing.

    This fixture provides a DisaggregationProblem instance initialized with:
    - The real disaggregation configuration (with weights from actual data)
    - The USA aggregated reader (with A01 and A03 aggregated into sector A)

    Args:
        real_disag_config: Configuration with weights from actual data
        usa_aggregated_reader: Reader with aggregated sectors

    Returns:
        DisaggregationProblem: Problem instance ready for testing
    """
    from disag_tools.disaggregation.problem import DisaggregationProblem

    return DisaggregationProblem.from_configuration(
        config=real_disag_config, reader=usa_aggregated_reader
    )


@pytest.fixture(scope="function")
def canada_country_disagg_config(data_dir: Path) -> CountryConfig:
    """Get the disaggregation configuration for Canada."""
    with open(data_dir / "canada_provinces" / "canada_provinces.yaml") as f:
        config_dict = yaml.safe_load(f)

    # Extract just the CAN configuration
    return CountryConfig(**config_dict["countries"]["CAN"])


@pytest.fixture(scope="function")
def canada_provincial_disagg_config(data_dir: Path) -> DisaggregationConfig:
    """Get the disaggregation configuration for Canada."""
    with open(data_dir / "canada_provinces" / "canada_provinces.yaml") as f:
        config_dict = yaml.safe_load(f)

    return DisaggregationConfig(**config_dict)


@pytest.fixture(scope="function")
def canada_technical_coeffs_prior(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "canada_provinces" / "technical_coeffs.csv"
    return pd.read_csv(path, index_col=0)


@pytest.fixture(scope="function")
def canada_final_demand_prior(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "canada_provinces" / "final_demand_prior.csv"
    return pd.read_csv(path)


@pytest.fixture
def usa_planted_solution(aggregated_blocks, usa_aggregated_reader, usa_reader_blocks):
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    return DisaggregatedBlocks.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
        final_demand=usa_aggregated_reader.final_demand,
    )
