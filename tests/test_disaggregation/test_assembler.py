"""Tests for the assembler module."""

import numpy as np
import pandas as pd

from disag_tools.assembler.assembler import AssembledData


def test_assembled_data_structure(default_problem, usa_reader):
    """Test that the assembled data has the correct structure and values."""
    # First solve the problem
    default_problem.solve()

    # Create assembled data
    assembled = AssembledData.from_solution(default_problem, usa_reader, check_output=False)

    # Check that the data has the correct structure
    assert isinstance(assembled.data, pd.DataFrame)
    assert isinstance(assembled.data.index, pd.MultiIndex)
    assert isinstance(assembled.data.columns, pd.MultiIndex)
    assert assembled.data.index.names == ["CountryInd", "industryInd"]
    assert assembled.data.columns.names == ["CountryInd", "industryInd"]

    # Check that shapes match
    assert assembled.data.shape == usa_reader.data.shape, (
        f"Shape mismatch: assembled {assembled.data.shape} vs " f"reader {usa_reader.data.shape}"
    )

    assert np.all(assembled.data.index == usa_reader.data.index), "Indices do not match"
    assert np.all(assembled.data.columns == usa_reader.data.columns), "Columns do not match"

    # Get all indices and columns from both DataFrames
    assembled_indices = set(assembled.data.index)
    reader_indices = set(usa_reader.data.index)
    assembled_columns = set(assembled.data.columns)
    reader_columns = set(usa_reader.data.columns)

    # Check for missing or extra indices
    missing_indices = reader_indices - assembled_indices
    extra_indices = assembled_indices - reader_indices
    assert not missing_indices, f"Missing indices in assembled data: {missing_indices}"
    assert not extra_indices, f"Extra indices in assembled data: {extra_indices}"

    # Check for missing or extra columns
    missing_columns = reader_columns - assembled_columns
    extra_columns = assembled_columns - reader_columns
    assert not missing_columns, f"Missing columns in assembled data: {missing_columns}"
    assert not extra_columns, f"Extra columns in assembled data: {extra_columns}"

    # Now that we know they have the same indices and columns, reindex both to ensure same order
    # Sort both DataFrames by index and columns to ensure consistent order
    sorted_indices = sorted(reader_indices)
    sorted_columns = sorted(reader_columns)

    assembled_data = assembled.data.reindex(index=sorted_indices, columns=sorted_columns)
    reader_data = usa_reader.data.reindex(index=sorted_indices, columns=sorted_columns)

    # Check that NaN positions match exactly
    assembled_nans = assembled_data.isna()
    reader_nans = reader_data.isna()
    mismatches = assembled_nans != reader_nans

    # If there are mismatches, show the positions
    if mismatches.any().any():  # type: ignore
        mismatch_positions = []
        for idx in sorted_indices:
            for col in sorted_columns:
                if mismatches.loc[idx, col]:  # type: ignore
                    mismatch_positions.append(
                        f"Position ({idx}, {col}): "
                        f"assembled={assembled_data.loc[idx, col]}, "
                        f"reader={reader_data.loc[idx, col]}"
                    )
        assert not mismatch_positions, "NaN positions do not match:\n" + "\n".join(
            mismatch_positions
        )

    # Check that all disaggregated sectors are present
    for agg_sector in default_problem.disaggregation_blocks.to_disagg_sector_names:
        subsectors = default_problem.solution_blocks.sector_mapping[agg_sector]
        for subsector in subsectors:
            assert subsector in assembled_data.index
            assert subsector in assembled_data.columns

    # Check that non-disaggregated sectors are present
    for sector in default_problem.disaggregation_blocks.non_disagg_sector_names:
        assert sector in assembled_data.index
        assert sector in assembled_data.columns

    # Check that final demand columns are present
    fd_cols = [
        col
        for col in usa_reader.data.columns
        if any(col[1].startswith(p) for p in usa_reader.FINAL_DEMAND_PREFIXES)
    ]
    for col in fd_cols:
        assert col in assembled_data.columns

    # Verify intermediate use values match what we get from solution blocks
    intermediate_use = default_problem.solution_blocks.get_intermediate_use()
    for row in intermediate_use.index:
        for col in intermediate_use.columns:
            assert np.allclose(
                assembled_data.loc[row, col], intermediate_use.loc[row, col], rtol=1e-2
            )

    # Verify final demand values match
    final_demand = default_problem.final_demand_blocks.disaggregated_final_demand
    # Only check rows that are in both DataFrames
    common_rows = assembled_data.index.intersection(final_demand.index)
    for row in common_rows:
        for col in fd_cols:
            if not pd.isna(final_demand.loc[row, col]):
                assert np.allclose(
                    assembled_data.loc[row, col], final_demand.loc[row, col], rtol=1e-2
                )

    # Verify output values match
    output = default_problem.solution_blocks.output
    for col in intermediate_use.columns:
        assert np.allclose(assembled_data.loc[("OUT", "OUT"), col], output[col], rtol=1e-2)

    # Check that we have all indices from the original data
    reader_indices = set(usa_reader.data.index)
    assembled_indices = set(assembled_data.index)
    missing_indices = reader_indices - assembled_indices
    extra_indices = assembled_indices - reader_indices
    assert not missing_indices, f"Missing indices: {missing_indices}"
    assert not extra_indices, f"Extra indices: {extra_indices}"

    # Check that we have all columns from the original data
    reader_cols = set(usa_reader.data.columns)
    assembled_cols = set(assembled_data.columns)
    missing_cols = reader_cols - assembled_cols
    extra_cols = assembled_cols - reader_cols
    assert not missing_cols, f"Missing columns: {missing_cols}"
    assert not extra_cols, f"Extra columns: {extra_cols}"


def test_column_ordering(default_problem, usa_reader):
    """Test that columns are ordered correctly within each country."""
    # Solve the problem first
    default_problem.solve()

    # Create assembled data
    assembled = AssembledData.from_solution(default_problem, usa_reader, check_output=False)

    # Get unique countries from assembled data
    countries = sorted({col[0] for col in assembled.data.columns})

    # For each country, verify column ordering
    for country in countries:
        # Get all columns for this country
        country_cols = [col for col in assembled.data.columns if col[0] == country]

        # Split into industry and final demand columns
        industry_cols = [
            col
            for col in country_cols
            if not any(col[1].startswith(p) for p in usa_reader.FINAL_DEMAND_PREFIXES)
        ]
        fd_cols = [
            col
            for col in country_cols
            if any(col[1].startswith(p) for p in usa_reader.FINAL_DEMAND_PREFIXES)
        ]

        # Check that industry columns are alphabetically ordered
        assert industry_cols == sorted(
            industry_cols
        ), f"Industry columns for {country} are not alphabetically ordered"

        # Check that all industry columns come before final demand columns
        if industry_cols and fd_cols:
            last_industry_idx = max(
                i for i, col in enumerate(assembled.data.columns) if col in industry_cols
            )
            first_fd_idx = min(i for i, col in enumerate(assembled.data.columns) if col in fd_cols)
            assert (
                last_industry_idx < first_fd_idx
            ), f"Final demand columns appear before industry columns for {country}"

    # Check that OUT column is at the end
    assert assembled.data.columns[-1] == ("OUT", "OUT"), "OUT column is not at the end"
