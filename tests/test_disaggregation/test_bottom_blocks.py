"""Tests for the bottom blocks (VA and TLS) functionality."""

import numpy as np
import pandas as pd
import pytest

from disag_tools.disaggregation.bottom_blocks import BottomBlocks


def test_bottom_blocks_structure(default_problem, usa_reader):
    """Test that BottomBlocks creates correct structure with VA and TLS rows."""
    # Get the blocks from the problem
    blocks = default_problem.bottom_blocks

    # Check that we have a single DataFrame with correct structure
    assert isinstance(blocks.data, pd.DataFrame)
    assert blocks.data.index.names == ["CountryInd", "industryInd"]
    assert blocks.data.columns.names == ["CountryInd", "industryInd"]

    # Check that we have VA and TLS rows
    assert ("VA", "VA") in blocks.data.index
    assert ("TLS", "TLS") in blocks.data.index
    assert len(blocks.data.index) == 2


def test_va_allocation(default_problem, usa_reader):
    """Test that VA is allocated correctly based on output weights."""
    blocks = default_problem.bottom_blocks

    # For each aggregated sector being disaggregated
    for agg_sector, subsectors in default_problem.solution_blocks.sector_mapping.items():
        if agg_sector in usa_reader.data.columns:
            # Get original VA value
            original_va = usa_reader.data.loc[("VA", "VA"), agg_sector]

            # Sum VA values of subsectors
            subsector_va_sum = sum(
                blocks.data.loc[("VA", "VA"), subsector] for subsector in subsectors
            )

            # Check that total VA is preserved
            assert pytest.approx(original_va, rel=1e-10) == subsector_va_sum

            # Check that VA ratios match output weights
            weights = [
                default_problem.solution_blocks.output[s]
                / default_problem.solution_blocks.output[subsectors[0]]
                for s in subsectors
            ]
            va_ratios = [
                blocks.data.loc[("VA", "VA"), s] / blocks.data.loc[("VA", "VA"), subsectors[0]]
                for s in subsectors
            ]
            assert np.allclose(weights, va_ratios, rtol=1e-10)


def test_tls_initialization(default_problem):
    """Test that TLS values are initialized as NaN."""
    blocks = default_problem.bottom_blocks

    # Check that all TLS values are NaN
    assert blocks.data.loc[("TLS", "TLS")].isna().all()


def test_bottom_blocks_columns(default_problem, usa_reader):
    """Test that bottom blocks only contain columns for disaggregated sectors."""
    blocks = default_problem.bottom_blocks

    # Get all subsectors from the mapping
    expected_sectors = []
    for subsectors in default_problem.solution_blocks.sector_mapping.values():
        expected_sectors.extend(subsectors)
    expected_sectors = sorted(expected_sectors)

    # Check that bottom blocks columns match exactly the disaggregated sectors
    assert list(blocks.data.columns) == expected_sectors
