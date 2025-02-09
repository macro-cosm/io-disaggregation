"""Tests for the configuration models."""

import pytest
import yaml
from pydantic import ValidationError

from multi_sector_disagg.configurations.config import (
    CountryConfig,
    DisaggregationConfig,
    RegionConfig,
    SectorConfig,
    SubsectorConfig,
)


def test_subsector_config_validation():
    """Test validation of subsector weights."""
    # Valid weights
    valid_subsectors = {
        "MFG1": SubsectorConfig(name="Primary Manufacturing", relative_output_weight=0.6),
        "MFG2": SubsectorConfig(name="Secondary Manufacturing", relative_output_weight=0.4),
    }
    # This should not raise an error
    SectorConfig(subsectors=valid_subsectors)

    # Invalid weights (don't sum to 1)
    invalid_subsectors = {
        "MFG1": SubsectorConfig(name="Primary Manufacturing", relative_output_weight=0.7),
        "MFG2": SubsectorConfig(name="Secondary Manufacturing", relative_output_weight=0.4),
    }
    with pytest.raises(ValidationError):
        SectorConfig(subsectors=invalid_subsectors)


def test_region_config_validation():
    """Test that region weights sum to 1."""
    valid_regions = {
        "USA1": {"name": "Eastern USA", "sector_weights": {"AGR": 0.45, "MFG": 0.6}},
        "USA2": {"name": "Western USA", "sector_weights": {"AGR": 0.55, "MFG": 0.4}},
    }
    CountryConfig(regions=valid_regions)

    invalid_regions = {
        "USA1": {"name": "Eastern USA", "sector_weights": {"AGR": 0.45, "MFG": 0.45}},
        "USA2": {"name": "Western USA", "sector_weights": {"AGR": 0.55, "MFG": 0.65}},
    }
    with pytest.raises(ValidationError) as exc_info:
        CountryConfig(regions=invalid_regions)
    assert "sum to 1" in str(exc_info.value)


def test_load_sector_config(sector_config_path):
    """Test loading sector disaggregation config from YAML."""
    with open(sector_config_path) as f:
        config_dict = yaml.safe_load(f)

    config = DisaggregationConfig(**config_dict)
    assert "MFG" in config.sectors
    assert len(config.sectors["MFG"].subsectors) == 2
    assert config.sectors["MFG"].subsectors["MFG1"].relative_output_weight == 0.6
    assert config.sectors["MFG"].subsectors["MFG2"].relative_output_weight == 0.4


def test_load_country_config():
    """Test loading country disaggregation config from YAML."""
    config = DisaggregationConfig.model_validate(
        {
            "countries": {
                "USA": {
                    "regions": {
                        "USA1": {
                            "name": "Eastern USA",
                            "sector_weights": {"AGR": 0.45, "MFG": 0.6},
                        },
                        "USA2": {
                            "name": "Western USA",
                            "sector_weights": {"AGR": 0.55, "MFG": 0.4},
                        },
                    }
                }
            }
        }
    )
    assert config.countries is not None
    assert len(config.countries["USA"].regions) == 2
    assert config.countries["USA"].regions["USA1"].sector_weights["AGR"] == 0.45


def test_get_mapping_sector_disagg(sector_config_path):
    """Test getting mapping for sector disaggregation."""
    with open(sector_config_path) as f:
        config_dict = yaml.safe_load(f)

    config = DisaggregationConfig(**config_dict)
    mapping = config.get_disagg_mapping()

    # Check that MFG is mapped to MFG1 and MFG2
    assert ("", "MFG") in mapping
    mapped_pairs = mapping[("", "MFG")]
    assert ("", "MFG1") in mapped_pairs
    assert ("", "MFG2") in mapped_pairs


def test_get_mapping_country_disagg():
    """Test getting mapping for country disaggregation."""
    config = DisaggregationConfig.model_validate(
        {
            "countries": {
                "USA": {
                    "regions": {
                        "USA1": {
                            "name": "Eastern USA",
                            "sector_weights": {"AGR": 0.45, "MFG": 0.6},
                        },
                        "USA2": {
                            "name": "Western USA",
                            "sector_weights": {"AGR": 0.55, "MFG": 0.4},
                        },
                    }
                }
            }
        }
    )
    mapping = config.get_disagg_mapping()
    assert ("USA", "AGR") in mapping
    assert set(mapping[("USA", "AGR")]) == {("USA1", "AGR"), ("USA2", "AGR")}


def test_get_final_size(sector_config_path):
    """Test computation of final matrix size."""
    # Load sector config
    with open(sector_config_path) as f:
        config = DisaggregationConfig(**yaml.safe_load(f))

    # Test with sample data dimensions
    original_countries = ["USA", "CHN", "ROW"]
    original_sectors = ["AGR", "MFG"]
    rows, cols = config.get_final_size(original_countries, original_sectors)

    # Original size is 3 countries × 2 sectors = 6
    # After disaggregation of MFG into MFG1 and MFG2:
    # We have 3 countries × (1 AGR + 2 MFG) = 9
    assert rows == 9
    assert cols == 9


def test_missing_sector_weights():
    """Test handling of missing sector weights in region config."""
    # Create config where one region is missing a sector
    regions = {
        "USA1": RegionConfig(
            name="Eastern United States",
            sector_weights={"AGR": 0.45, "MFG": 0.60},
        ),
        "USA2": RegionConfig(
            name="Western United States",
            sector_weights={"AGR": 0.55},  # Missing MFG
        ),
    }

    # Should raise validation error because MFG weights don't sum to 1
    with pytest.raises(ValidationError):
        CountryConfig(regions=regions)


def test_empty_disaggregation_config():
    """Test that DisaggregationConfig requires at least one type of disaggregation."""
    # Try to create config with no countries and no sectors
    with pytest.raises(
        ValidationError, match="Must specify at least one country or sector to disaggregate"
    ):
        DisaggregationConfig(countries=None, sectors=None)


def test_combined_disaggregation_mapping():
    """Test mapping generation when both country and sector disaggregation are specified."""
    # Create a config with both country and sector disaggregation
    config = DisaggregationConfig(
        countries={
            "USA": CountryConfig(
                regions={
                    "USA1": RegionConfig(
                        name="Eastern US",
                        sector_weights={"MFG": 0.6},
                    ),
                    "USA2": RegionConfig(
                        name="Western US",
                        sector_weights={"MFG": 0.4},
                    ),
                }
            )
        },
        sectors={
            "MFG": SectorConfig(
                subsectors={
                    "MFG1": SubsectorConfig(
                        name="Primary Manufacturing", relative_output_weight=0.7
                    ),
                    "MFG2": SubsectorConfig(
                        name="Secondary Manufacturing", relative_output_weight=0.3
                    ),
                }
            )
        },
    )

    mapping = config.get_disagg_mapping()

    # Check that USA-MFG is mapped to all combinations
    assert ("USA", "MFG") in mapping
    mapped_pairs = mapping[("USA", "MFG")]
    assert len(mapped_pairs) == 4  # 2 regions × 2 subsectors
    assert ("USA1", "MFG1") in mapped_pairs
    assert ("USA1", "MFG2") in mapped_pairs
    assert ("USA2", "MFG1") in mapped_pairs
    assert ("USA2", "MFG2") in mapped_pairs


def test_sector_first_key():
    """Test that sector key is used first in mapping when no country disaggregation."""
    config = DisaggregationConfig(
        sectors={
            "MFG": SectorConfig(
                subsectors={
                    "MFG1": SubsectorConfig(
                        name="Primary Manufacturing", relative_output_weight=0.7
                    ),
                    "MFG2": SubsectorConfig(
                        name="Secondary Manufacturing", relative_output_weight=0.3
                    ),
                }
            )
        }
    )

    mapping = config.get_disagg_mapping()
    assert ("", "MFG") in mapping  # Should use empty string for country
    assert len(mapping) == 1  # Should only have one mapping


def test_invalid_sector_weights():
    """Test that sector weights must sum to 1 for each sector."""
    invalid_regions = {
        "USA1": {
            "name": "Eastern USA",
            "sector_weights": {"AGR": 0.45, "MFG": 0.45},  # MFG weights will sum to more than 1
        },
        "USA2": {
            "name": "Western USA",
            "sector_weights": {"AGR": 0.55, "MFG": 0.65},
        },  # MFG total = 1.1
    }
    with pytest.raises(ValidationError) as exc_info:
        CountryConfig(regions=invalid_regions)
    assert "sum to 1" in str(exc_info.value)
