"""Tests for the block structure handling module."""

import logging

import numpy as np
import pandas as pd
import pytest

from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregatedBlocks,
    DisaggregationBlocks,
    SectorInfo,
)
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
    # Create indices with sectors that will be disaggregated
    index = pd.MultiIndex.from_tuples(
        [
            ("USA", "AGR"),
            ("USA", "MFG"),
            ("USA", "A03"),
            ("CAN", "AGR"),
            ("CAN", "MFG"),
            ("CAN", "A03"),
            ("OUT", "OUT"),  # Add OUT row
        ],
        names=["CountryInd", "industryInd"],
    )
    # Create sample data with known values for testing G blocks
    # Each sector-to-sector flow will have a unique value for easy identification
    data = np.array(
        [
            [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.0],  # USA-AGR to all
            [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.0],  # USA-MFG to all
            [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.0],  # USA-A03 to all
            [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.0],  # CAN-AGR to all
            [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.0],  # CAN-MFG to all
            [0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.0],  # CAN-A03 to all
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # OUT row (output values)
        ]
    )
    return pd.DataFrame(data, index=index, columns=index)


@pytest.fixture
def single_region_blocks(sample_single_region_tech_coef):
    """Create sample DisaggregationBlocks for single region."""
    sectors_to_disaggregate = [
        ("A03", "Manufacturing", 3),  # A03 splits into 3 subsectors
    ]
    return DisaggregationBlocks.from_technical_coefficients(
        sample_single_region_tech_coef,
        sectors_to_disaggregate,
        output=pd.Series(
            index=sample_single_region_tech_coef.index,
            data=np.full(sample_single_region_tech_coef.shape[0], 100.0),
        ),
    )


@pytest.fixture
def multi_region_blocks(sample_multi_region_tech_coef):
    """Create sample DisaggregationBlocks for multi region."""
    sectors_to_disaggregate = [
        (("USA", "A03"), "USA Manufacturing", 3),  # USA-A03 splits into 3
        (("CAN", "A03"), "CAN Manufacturing", 3),  # CAN-A03 splits into 3
    ]
    return DisaggregationBlocks.from_technical_coefficients(
        sample_multi_region_tech_coef,
        sectors_to_disaggregate,
        output=pd.Series(
            index=sample_multi_region_tech_coef.index,
            data=np.full(sample_multi_region_tech_coef.shape[0], 100.0),
        ),
    )


@pytest.fixture
def sample_reader(sample_multi_region_tech_coef):
    """Create a sample ICIOReader for testing."""
    # Create a minimal reader with the sample data
    reader = ICIOReader(
        data=sample_multi_region_tech_coef,
        countries=["USA", "CAN"],
        industries=["AGR", "MFG", "A03"],
    )
    reader.data_path = None  # No file path for test data
    return reader


def test_sector_info_single_region(single_region_blocks):
    """Test SectorInfo properties for single-region case."""
    sector = single_region_blocks.get_sector_info(1)
    assert not sector.is_multi_region
    assert sector.country is None
    assert sector.sector == "A03"
    assert sector.k == 3


def test_sector_info_multi_region(multi_region_blocks):
    """Test SectorInfo properties for multi-region case."""
    sector = multi_region_blocks.get_sector_info(1)
    assert sector.is_multi_region
    assert sector.country == "CAN"
    assert sector.sector == "A03"
    assert sector.k == 3


def test_blocks_properties_single_region(single_region_blocks):
    """Test basic properties for single-region case."""
    assert not single_region_blocks.is_multi_region
    assert single_region_blocks.N == 3  # Total sectors
    assert single_region_blocks.K == 1  # Number of sectors being disaggregated (A03)
    assert single_region_blocks.M == 3  # Total subsectors (3 for A03)


def test_blocks_properties_multi_region(multi_region_blocks):
    """Test basic properties for multi-region case."""
    assert multi_region_blocks.is_multi_region
    # Total sectors should be 6 (2 countries × 3 sectors), excluding OUT row
    assert multi_region_blocks.N == 6  # Total sectors (2 countries × 3 sectors)
    assert multi_region_blocks.K == 2  # Number of sectors being disaggregated
    assert multi_region_blocks.M == 6  # Total subsectors (2×3 for USA-A03 and 2×3 for CAN-A03)


def test_get_A0_single_region(single_region_blocks):
    """Test getting A₀ block for single-region case."""
    A0 = single_region_blocks.get_A0()
    assert isinstance(A0, np.ndarray)
    # Should be 2×2 for A01 and A02 (undisaggregated)
    assert A0.shape == (2, 2)


def test_get_A0_multi_region(multi_region_blocks):
    """Test getting A₀ block for multi-region case."""
    A0 = multi_region_blocks.get_A0()
    assert isinstance(A0, np.ndarray)
    # Should be 4×4 for USA/CAN-A01/A02 (undisaggregated)
    assert A0.shape == (4, 4)


def test_get_B_single_region(single_region_blocks, caplog):
    """Test getting B block for single region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_B for single region")
    logger.debug(f"Matrix shape: {single_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in single_region_blocks.sectors]}")
    logger.debug(
        f"N={single_region_blocks.N}, K={single_region_blocks.K}, M={single_region_blocks.M}"
    )

    # Get B block for first sector (A03)
    B1 = single_region_blocks.get_B(1)
    logger.debug(f"B1 shape: {B1.shape}")
    logger.debug(
        f"Expected shape: ({single_region_blocks.N - single_region_blocks.K}, {single_region_blocks.sectors[0].k})"
    )

    # Test shape - should be (2 undisaggregated rows × 3 A03 subsectors)
    N_K = single_region_blocks.N - single_region_blocks.K  # 2 (A01, A02)
    k1 = single_region_blocks.sectors[0].k  # 3 subsectors for A03
    assert B1.shape[0] == N_K, f"Expected shape ({N_K}, {k1}), got {B1.shape}"


def test_get_B_multi_region(multi_region_blocks, caplog):
    """Test getting B block for multi region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_B for multi region")
    logger.debug(f"Matrix shape: {multi_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in multi_region_blocks.sectors]}")
    logger.debug(f"N={multi_region_blocks.N}, K={multi_region_blocks.K}, M={multi_region_blocks.M}")

    # Get B block for first sector (USA-A03)
    B1 = multi_region_blocks.get_B(1)
    logger.debug(f"B1 shape: {B1.shape}")
    logger.debug(
        f"Expected shape: ({multi_region_blocks.N - multi_region_blocks.K}, {multi_region_blocks.sectors[0].k})"
    )

    # Test shape - should be (4 undisaggregated rows × 3 USA-A03 subsectors)
    N_K = multi_region_blocks.N - multi_region_blocks.K  # 4 (USA/CAN-A01/A02)
    k1 = multi_region_blocks.sectors[0].k  # 3 subsectors for USA-A03
    assert B1.shape == (N_K,), f"Expected shape ({N_K}), got {B1.shape}"

    # Get B block for second sector (CAN-A03)
    B2 = multi_region_blocks.get_B(2)
    logger.debug(f"B2 shape: {B2.shape}")
    logger.debug(f"Expected shape: ({N_K}, {multi_region_blocks.sectors[1].k})")
    assert B2.shape == (N_K,)  # Same shape as B1 since both split into 3


def test_matrix_reordering_single_region(single_region_blocks, caplog):
    """Test matrix reordering for single region case."""
    caplog.set_level(logging.DEBUG)

    logger.debug("Testing matrix reordering for single region")
    logger.debug(f"Original index: {single_region_blocks.reordered_matrix.index.tolist()}")
    logger.debug(f"Original columns: {single_region_blocks.reordered_matrix.columns.tolist()}")

    # Check that undisaggregated sectors come first
    undisaggregated = ["A01", "A02"]  # These should come first
    disaggregated = ["A03"]  # This is being disaggregated

    actual_order = single_region_blocks.reordered_matrix.index.tolist()
    logger.debug(f"Expected order: {undisaggregated + disaggregated}")
    logger.debug(f"Actual order: {actual_order}")

    # Check index order
    assert actual_order[: len(undisaggregated)] == undisaggregated
    assert all(d in actual_order[len(undisaggregated) :] for d in disaggregated)

    # Check that columns match index exactly
    assert (
        single_region_blocks.reordered_matrix.index.tolist()
        == single_region_blocks.reordered_matrix.columns.tolist()
    )

    # Verify that all expected sectors are present
    all_sectors = set(undisaggregated + disaggregated)
    assert set(single_region_blocks.reordered_matrix.index) == all_sectors
    assert set(single_region_blocks.reordered_matrix.columns) == all_sectors

    # Verify that the matrix shape matches the number of sectors
    assert single_region_blocks.reordered_matrix.shape == (len(all_sectors), len(all_sectors))

    # Verify that the order is consistent with sector indices
    for n, sector_info in enumerate(single_region_blocks.sectors, start=1):
        sector_position = actual_order.index(sector_info.sector_id)
        # Position should be after all undisaggregated sectors
        assert sector_position >= len(undisaggregated)
        # And in the same order as the sectors list
        if n > 1:
            prev_sector = single_region_blocks.sectors[n - 2]
            prev_position = actual_order.index(prev_sector.sector_id)
            assert sector_position > prev_position


def test_matrix_reordering_multi_region(multi_region_blocks, caplog):
    """Test matrix reordering for multi region case."""
    caplog.set_level(logging.DEBUG)

    logger.debug("Testing matrix reordering for multi region")
    logger.debug(f"Original index: {multi_region_blocks.reordered_matrix.index.tolist()}")
    logger.debug(f"Original columns: {multi_region_blocks.reordered_matrix.columns.tolist()}")

    # Check that undisaggregated sectors come first, in alphabetical order by country
    undisaggregated = [
        ("CAN", "AGR"),
        ("CAN", "MFG"),
        ("USA", "AGR"),
        ("USA", "MFG"),
    ]  # Alphabetical order
    disaggregated = [("USA", "A03"), ("CAN", "A03")]  # These are being disaggregated

    actual_order = multi_region_blocks.reordered_matrix.index.tolist()
    logger.debug(f"Expected order: {undisaggregated + disaggregated}")
    logger.debug(f"Actual order: {actual_order}")

    # Check index order
    assert actual_order[: len(undisaggregated)] == undisaggregated
    assert set(actual_order[len(undisaggregated) :]) == set(
        disaggregated
    )  # Order doesn't matter for disaggregated sectors

    # Check that columns match index exactly
    assert (
        multi_region_blocks.reordered_matrix.index.tolist()
        == multi_region_blocks.reordered_matrix.columns.tolist()
    )

    # Verify that all expected sectors are present
    all_sectors = set(undisaggregated + disaggregated)
    assert set(multi_region_blocks.reordered_matrix.index) == all_sectors
    assert set(multi_region_blocks.reordered_matrix.columns) == all_sectors

    # Verify that the matrix shape matches the number of sectors
    assert multi_region_blocks.reordered_matrix.shape == (len(all_sectors), len(all_sectors))

    # Verify that the order is consistent with sector indices
    for n, sector_info in enumerate(multi_region_blocks.sectors, start=1):
        sector_position = actual_order.index(sector_info.sector_id)
        # Position should be after all undisaggregated sectors
        assert sector_position >= len(undisaggregated)
        # And in the same order as the sectors list
        if n > 1:
            prev_sector = multi_region_blocks.sectors[n - 2]
            prev_position = actual_order.index(prev_sector.sector_id)
            assert sector_position > prev_position


def test_get_C_single_region(single_region_blocks, caplog):
    """Test getting C block for single region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_C for single region")
    logger.debug(f"Matrix shape: {single_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in single_region_blocks.sectors]}")
    logger.debug(
        f"N={single_region_blocks.N}, K={single_region_blocks.K}, M={single_region_blocks.M}"
    )

    # Get C block for first sector (A03)
    C1 = single_region_blocks.get_C(1)
    logger.debug(f"C1 shape: {C1.shape}")

    # Test shape - should be (3 A03 subsectors × 2 undisaggregated columns)
    N_K = single_region_blocks.N - single_region_blocks.K  # 2 (A01, A02)
    k1 = single_region_blocks.sectors[0].k  # 3 subsectors for A03
    assert C1.shape == (N_K,), f"Expected shape ({k1}, {N_K}), got {C1.shape}"


def test_get_C_multi_region(multi_region_blocks, caplog):
    """Test getting C block for multi region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_C for multi region")
    logger.debug(f"Matrix shape: {multi_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in multi_region_blocks.sectors]}")
    logger.debug(f"N={multi_region_blocks.N}, K={multi_region_blocks.K}, M={multi_region_blocks.M}")

    # Get C block for first sector (USA-A03)
    C1 = multi_region_blocks.get_C(1)
    logger.debug(f"C1 shape: {C1.shape}")

    # Test shape - should be (3 USA-A03 subsectors × 4 undisaggregated columns)
    N_K = multi_region_blocks.N - multi_region_blocks.K  # 4 (USA/CAN-A01/A02)
    k1 = multi_region_blocks.sectors[0].k  # 3 subsectors for USA-A03
    assert C1.shape == (N_K,), f"Expected shape ({N_K}), got {C1.shape}"

    # Get C block for second sector (CAN-A03)
    C2 = multi_region_blocks.get_C(2)
    logger.debug(f"C2 shape: {C2.shape}")
    assert C2.shape == (N_K,)  # Same shape as C1 since both split into 3


def test_get_D_single_region(single_region_blocks, caplog):
    """Test getting D block for single region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_D for single region")
    logger.debug(f"Matrix shape: {single_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in single_region_blocks.sectors]}")

    # Get D block for sector 1 to itself (D^{11})
    D1 = single_region_blocks.get_D(1)
    logger.debug(f"D11 shape: {D1.shape}")

    # Test shape - should be (3 A03 subsectors × 3 A03 subsectors)
    k1 = single_region_blocks.K
    # D1 should be of size k1
    assert D1.shape == (k1,), f"Expected shape ({k1},), got {D1.shape}"


def test_get_D_multi_region(multi_region_blocks, caplog):
    """Test getting D block for multi region case."""
    caplog.set_level(logging.DEBUG)

    # Log initial state
    logger.debug("Testing get_D for multi region")
    logger.debug(f"Matrix shape: {multi_region_blocks.reordered_matrix.shape}")
    logger.debug(f"Sectors: {[s.sector_id for s in multi_region_blocks.sectors]}")

    # Get D block for USA-A03 to itself (D^{11})
    D11 = multi_region_blocks.get_D(1)
    logger.debug(f"D11 shape: {D11.shape}")

    # Test shape - should be (3 USA-A03 subsectors × 3 USA-A03 subsectors)
    k1 = multi_region_blocks.K  # 3 subsectors for USA-A03
    assert D11.shape == (k1,), f"Expected shape ({k1},), got {D11.shape}"


def test_invalid_sector_indices():
    """Test error handling for invalid sector indices."""
    # Create a minimal blocks object
    matrix = pd.DataFrame([[1]], index=["A01"], columns=["A01"])
    blocks = DisaggregationBlocks(
        sectors=[SectorInfo(1, "A01", "Test", 2)],
        reordered_matrix=matrix,
        to_disagg_sector_names=["A01"],
        non_disagg_sector_names=[],
        output=pd.Series(index=matrix.index, data=[100]),  # No non-disaggregated sectors
    )

    # Test invalid indices for get_B
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_B(0)  # Too small
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_B(2)  # Too large

    # Test invalid indices for get_C
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_C(0)  # Too small
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_C(2)  # Too large

    # Test invalid indices for get_D
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_D(0)  # First index too small
    with pytest.raises(ValueError, match="out of range"):
        blocks.get_D(2)  # First index too large


def test_fixture(usa_reader_blocks):
    assert usa_reader_blocks.to_disagg_sector_names == [
        ("ROW", "A01"),
        ("ROW", "A03"),
        ("USA", "A01"),
        ("USA", "A03"),
    ]


def test_E_block(usa_reader_blocks):
    """Test getting E block for USA reader."""
    E = usa_reader_blocks.get_e_vector(1)
    assert isinstance(E, np.ndarray)

    N = usa_reader_blocks.N
    K = usa_reader_blocks.K

    kl = 2

    # theoretical length is (N-K) kl
    assert E.shape[0] == (N - K) * kl, f"Expected shape ({(N-K) * kl}), got {E.shape}"


def test_F_block(usa_reader_blocks):
    """Test getting E block for USA reader."""
    F = usa_reader_blocks.get_f_vector(1)
    assert isinstance(F, np.ndarray)

    N = usa_reader_blocks.N
    K = usa_reader_blocks.K

    kl = 2

    # theoretical length is (N-K) kl
    assert F.shape[0] == (N - K) * kl, f"Expected shape ({(N-K) * kl}), got {F.shape}"


def test_gnl_vector(usa_reader_blocks):
    """Test getting gnl vector for USA reader."""
    gnl = usa_reader_blocks.get_gnl_vector(1, 1)
    assert isinstance(gnl, np.ndarray)

    assert gnl.shape[0] == 4, f"Expected shape (4), got {gnl.shape}"


def test_gn_vector(usa_reader_blocks):
    """Test getting gn vector for USA reader."""
    gn = usa_reader_blocks.get_gn_vector(1)
    assert isinstance(gn, np.ndarray)

    assert (
        gn.shape[0] == 4 * usa_reader_blocks.m
    ), f"Expected shape ({4 * usa_reader_blocks.m}), got {gn.shape}"


def test_bn_vector(usa_reader_blocks):
    bn = usa_reader_blocks.get_bn_vector(1)
    # should be size 2
    assert bn.shape[0] == 2, f"Expected shape (2), got {bn.shape}"


def test_xn_vector(usa_reader_blocks):
    # theoretical length is 2( 2(N-K)+1+4)
    xn = usa_reader_blocks.get_xn_vector(1)
    th_size = 2 * (2 * (usa_reader_blocks.N - usa_reader_blocks.K) + 1 + 4)
    assert xn.shape[0] == th_size, f"Expected shape ({th_size}), got {xn.shape}"
