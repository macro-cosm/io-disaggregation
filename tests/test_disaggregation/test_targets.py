"""Tests for the disaggregation target vectors module."""

import logging

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from disag_tools.configurations.config import DisaggregationConfig, SectorConfig, SubsectorConfig
from disag_tools.disaggregation.targets import DisaggregationTargets
from disag_tools.readers.blocks import DisaggregationBlocks
from disag_tools.readers.icio_reader import ICIOReader

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_single_region_tech_coef():
    """Create a sample technical coefficients matrix for single region."""
    sectors = ["A01", "A02", "A03"]  # A03 will be disaggregated
    data = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    return pd.DataFrame(data, index=sectors, columns=sectors)


@pytest.fixture
def sample_multi_region_tech_coef():
    """Create a sample technical coefficients matrix for multi region."""
    # Create indices with A03 to be disaggregated
    indices = [
        ("USA", "A01"),
        ("USA", "A02"),
        ("USA", "A03"),
        ("CAN", "A01"),
        ("CAN", "A02"),
        ("CAN", "A03"),
    ]
    data = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
            [0.6, 0.7, 0.8, 0.9, 0.1, 0.2],
        ]
    )
    return pd.DataFrame(data, index=indices, columns=indices)


@pytest.fixture
def single_region_blocks(sample_single_region_tech_coef):
    """Create sample DisaggregationBlocks for single region."""
    sectors_to_disaggregate = [
        ("A03", "Manufacturing", 3),  # A03 splits into 3 subsectors
    ]
    return DisaggregationBlocks.from_technical_coefficients(
        sample_single_region_tech_coef, sectors_to_disaggregate
    )


@pytest.fixture
def multi_region_blocks(sample_multi_region_tech_coef):
    """Create sample DisaggregationBlocks for multi region."""
    sectors_to_disaggregate = [
        (("USA", "A03"), "USA Manufacturing", 3),  # USA-A03 splits into 3
        (("CAN", "A03"), "CAN Manufacturing", 3),  # CAN-A03 splits into 3
    ]
    return DisaggregationBlocks.from_technical_coefficients(
        sample_multi_region_tech_coef, sectors_to_disaggregate
    )


@pytest.fixture
def single_region_config():
    """Create sample DisaggregationConfig for single region."""
    return DisaggregationConfig(
        sectors={
            "A03": SectorConfig(
                subsectors={
                    "A03a": SubsectorConfig(
                        name="First subsector",
                        relative_output_weights={"": 0.4},  # Empty string for single region
                    ),
                    "A03b": SubsectorConfig(
                        name="Second subsector",
                        relative_output_weights={"": 0.3},
                    ),
                    "A03c": SubsectorConfig(
                        name="Third subsector",
                        relative_output_weights={"": 0.3},
                    ),
                }
            )
        }
    )


@pytest.fixture
def multi_region_config():
    """Create sample DisaggregationConfig for multi region."""
    return DisaggregationConfig(
        sectors={
            "A03": SectorConfig(
                subsectors={
                    "A03a": SubsectorConfig(
                        name="First subsector",
                        relative_output_weights={"USA": 0.4, "CAN": 0.4},
                    ),
                    "A03b": SubsectorConfig(
                        name="Second subsector",
                        relative_output_weights={"USA": 0.3, "CAN": 0.3},
                    ),
                    "A03c": SubsectorConfig(
                        name="Third subsector",
                        relative_output_weights={"USA": 0.3, "CAN": 0.3},
                    ),
                }
            )
        }
    )


def test_get_weights_single_region(single_region_blocks, single_region_config):
    """Test getting weights for single region case."""
    targets = DisaggregationTargets(single_region_blocks, single_region_config)
    sector_info = single_region_blocks.get_sector_info(1)  # A03
    weights = targets.get_weights(sector_info)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == (3,)  # 3 subsectors
    assert np.allclose(weights, [0.4, 0.3, 0.3])
    assert np.isclose(weights.sum(), 1.0)


def test_get_weights_multi_region(multi_region_blocks, multi_region_config):
    """Test getting weights for multi region case."""
    targets = DisaggregationTargets(multi_region_blocks, multi_region_config)

    # Test USA weights
    usa_sector = multi_region_blocks.get_sector_info(1)  # USA-A03
    usa_weights = targets.get_weights(usa_sector)
    assert usa_weights.shape == (3,)
    assert np.allclose(usa_weights, [0.4, 0.3, 0.3])

    # Test CAN weights
    can_sector = multi_region_blocks.get_sector_info(2)  # CAN-A03
    can_weights = targets.get_weights(can_sector)
    assert can_weights.shape == (3,)
    assert np.allclose(can_weights, [0.4, 0.3, 0.3])


def test_get_target_vector_single_region(single_region_blocks, single_region_config):
    """Test constructing target vector for single region case."""
    targets = DisaggregationTargets(single_region_blocks, single_region_config)
    Y = targets.get_target_vector(1)  # A03

    # Check that Y has the correct structure
    N_K = single_region_blocks.N - single_region_blocks.K  # 2 (A01, A02)
    k = single_region_blocks.get_sector_info(1).k  # 3 subsectors

    # Expected components:
    # B: (N-K) values repeated k times
    # C: (N-K) values repeated k times
    # D: k × k values for self-interaction
    # w: k weights
    expected_length = (N_K * k) + (N_K * k) + (k * k) + k

    assert Y.shape == (expected_length,)
    assert np.all(np.isfinite(Y))


def test_get_target_vector_multi_region(multi_region_blocks, multi_region_config):
    """Test constructing target vector for multi region case."""
    targets = DisaggregationTargets(multi_region_blocks, multi_region_config)

    # Test for USA-A03
    Y_usa = targets.get_target_vector(1)
    N_K = multi_region_blocks.N - multi_region_blocks.K  # 4 (USA/CAN-A01/A02)
    k = multi_region_blocks.get_sector_info(1).k  # 3 subsectors
    K = multi_region_blocks.K  # 2 sectors being disaggregated

    # Expected components for each sector n:
    # B: (N-K) values repeated k times
    # C: (N-K) values repeated k times
    # D: k × sum(k_l) values for interaction with all disaggregated sectors
    # w: k weights
    expected_length = (N_K * k) + (N_K * k) + (k * K * k) + k

    assert Y_usa.shape == (expected_length,)
    assert np.all(np.isfinite(Y_usa))

    # Test for CAN-A03
    Y_can = targets.get_target_vector(2)
    assert Y_can.shape == (expected_length,)
    assert np.all(np.isfinite(Y_can))


def test_invalid_sector_index(single_region_blocks, single_region_config):
    """Test error handling for invalid sector indices."""
    targets = DisaggregationTargets(single_region_blocks, single_region_config)

    with pytest.raises(ValueError, match="out of range"):
        targets.get_target_vector(0)  # Too small
    with pytest.raises(ValueError, match="out of range"):
        targets.get_target_vector(2)  # Too large


def test_missing_sector_config(single_region_blocks):
    """Test error handling for missing sector configuration."""
    # Create config with empty sectors but valid country config to pass validation
    config = DisaggregationConfig(
        sectors={},
        countries={
            "USA": {
                "regions": {
                    "USA1": {
                        "name": "Eastern USA",
                        "sector_weights": {"A03": 0.6},
                    },
                    "USA2": {
                        "name": "Western USA",
                        "sector_weights": {"A03": 0.4},
                    },
                }
            }
        },
    )
    targets = DisaggregationTargets(single_region_blocks, config)

    with pytest.raises(ValueError, match="No sector disaggregation configuration"):
        targets.get_target_vector(1)


def test_mismatched_weights(single_region_blocks, single_region_config):
    """Test error handling for mismatched number of weights."""
    # Modify config to have wrong number of subsectors
    config = DisaggregationConfig(
        sectors={
            "A03": SectorConfig(
                subsectors={
                    "A03a": SubsectorConfig(
                        name="First subsector",
                        relative_output_weights={"": 0.5},
                    ),
                    "A03b": SubsectorConfig(
                        name="Second subsector",
                        relative_output_weights={"": 0.5},
                    ),
                    # Missing third subsector
                }
            )
        }
    )
    targets = DisaggregationTargets(single_region_blocks, config)

    with pytest.raises(ValueError, match="Expected 3 weights"):
        targets.get_target_vector(1)


def test_real_data_target_vector(icio_reader: ICIOReader):
    """Test target vector construction with real ICIO data."""
    # First create a reader with only USA data
    usa_reader = ICIOReader.from_csv_selection(icio_reader.data_path, selected_countries=["USA"])

    # Create blocks for A01 disaggregation
    blocks = usa_reader.get_reordered_technical_coefficients(["A01"])

    # Create config for A01 disaggregation into three subsectors
    config = DisaggregationConfig(
        sectors={
            "A01": SectorConfig(
                subsectors={
                    "A01a": SubsectorConfig(
                        name="Crop Production",
                        relative_output_weights={
                            "USA": 0.4,  # 40% of USA's A01
                            "ROW": 0.4,  # Same weights for ROW
                        },
                    ),
                    "A01b": SubsectorConfig(
                        name="Animal Production",
                        relative_output_weights={
                            "USA": 0.35,  # 35% of USA's A01
                            "ROW": 0.35,  # Same weights for ROW
                        },
                    ),
                    "A01c": SubsectorConfig(
                        name="Support Activities",
                        relative_output_weights={
                            "USA": 0.25,  # 25% of USA's A01
                            "ROW": 0.25,  # Same weights for ROW
                        },
                    ),
                }
            )
        }
    )

    # Create targets instance
    targets = DisaggregationTargets(blocks, config)

    # Get target vector for A01 (first sector)
    Y = targets.get_target_vector(1)

    # Get the original technical coefficients
    tech_coef = usa_reader.technical_coefficients

    # Get sector info
    sector_info = blocks.get_sector_info(1)
    assert sector_info.sector == "A01"
    assert sector_info.k == 3

    # Check components match original data
    N_K = blocks.N - blocks.K  # Number of undisaggregated sectors
    k = sector_info.k  # Number of subsectors (3)

    # 1. Check B^n component (flows from undisaggregated to A01)
    B = Y[: (N_K * k)].reshape(N_K, k)
    B_orig = blocks.get_B(1)
    assert np.allclose(B, B_orig)

    # 2. Check C^n component (flows from A01 to undisaggregated)
    C = Y[(N_K * k) : (2 * N_K * k)].reshape(k, N_K)
    C_orig = blocks.get_C(1)
    assert np.allclose(C, C_orig)

    # 3. Check D^n component (flows between A01 and other disaggregated sectors)
    D = Y[(2 * N_K * k) : -k].reshape(k, k)  # -k to exclude weights
    D_orig = blocks.get_D(1, 1)  # Only one disaggregated sector
    assert np.allclose(D, D_orig)

    # 4. Check weights
    w = Y[-k:]  # Last k elements are weights
    assert np.allclose(w, [0.4, 0.35, 0.25])
    assert np.isclose(w.sum(), 1.0)


def test_get_target_vector_by_sector_id_single_region(single_region_blocks, single_region_config):
    """Test getting target vector by sector ID for single region case."""
    targets = DisaggregationTargets(single_region_blocks, single_region_config)

    # Get vector using sector ID
    Y_by_id = targets.get_target_vector_by_sector_id("A03")
    # Get vector using index for comparison
    Y_by_index = targets.get_target_vector(1)

    # Vectors should be identical
    assert np.allclose(Y_by_id, Y_by_index)


def test_get_target_vector_by_sector_id_multi_region(multi_region_blocks, multi_region_config):
    """Test getting target vector by sector ID for multi region case."""
    targets = DisaggregationTargets(multi_region_blocks, multi_region_config)

    # Test USA-A03
    Y_usa_by_id = targets.get_target_vector_by_sector_id(("USA", "A03"))
    Y_usa_by_index = targets.get_target_vector(1)
    assert np.allclose(Y_usa_by_id, Y_usa_by_index)

    # Test CAN-A03
    Y_can_by_id = targets.get_target_vector_by_sector_id(("CAN", "A03"))
    Y_can_by_index = targets.get_target_vector(2)
    assert np.allclose(Y_can_by_id, Y_can_by_index)


def test_get_target_vector_by_sector_id_not_found(single_region_blocks, single_region_config):
    """Test error handling when sector ID is not found."""
    targets = DisaggregationTargets(single_region_blocks, single_region_config)

    with pytest.raises(ValueError, match="not found in blocks"):
        targets.get_target_vector_by_sector_id("A99")  # Non-existent sector
