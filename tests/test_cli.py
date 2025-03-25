"""Tests for the command line interface."""

import logging

import pytest
import yaml
from click.testing import CliRunner

from disag_tools.cli import disaggregate
from disag_tools.readers import ICIOReader


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def real_config_file(tmp_path):
    """Create a config file with real sector disaggregation."""
    config = {
        "sectors": {
            "A": {
                "subsectors": {
                    "A01": {
                        "name": "Agriculture",
                        "relative_output_weights": {
                            "USA": 0.990,
                            "ROW": 0.915,
                        },
                    },
                    "A03": {
                        "name": "Fishing",
                        "relative_output_weights": {
                            "USA": 0.010,
                            "ROW": 0.085,
                        },
                    },
                }
            }
        }
    }
    config_path = tmp_path / "test_sector_disagg.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def industry_mapping_file(tmp_path):
    """Create an industry mapping file for sector aggregation."""
    mapping = {"A": ["A01", "A03"]}
    mapping_path = tmp_path / "industry_mapping.yaml"
    with open(mapping_path, "w") as f:
        yaml.dump(mapping, f)
    return mapping_path


@pytest.fixture
def aggregated_data_file(tmp_path, usa_aggregated_reader):
    """Create a CSV file with aggregated data."""
    data_path = tmp_path / "aggregated_data.csv"
    usa_aggregated_reader.data.to_csv(data_path)
    return data_path


def test_cli_help(cli_runner):
    """Test that the CLI help works."""
    result = cli_runner.invoke(disaggregate, ["--help"])
    assert result.exit_code == 0
    assert "Disaggregate an IO table using the specified configuration" in result.output


def test_cli_missing_args(cli_runner):
    """Test that the CLI fails gracefully with missing arguments."""
    result = cli_runner.invoke(disaggregate)
    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_cli_invalid_config(cli_runner, tmp_path, temp_output_dir):
    """Test that the CLI fails gracefully with invalid configuration."""
    # Create an invalid config file
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("invalid: yaml: content")

    # Create a dummy input file
    input_path = tmp_path / "input.csv"
    input_path.write_text("dummy,data\n1,2")

    result = cli_runner.invoke(
        disaggregate, [str(config_path), str(input_path), str(temp_output_dir)]
    )
    assert result.exit_code != 0


def test_cli_invalid_input(cli_runner, tmp_path, temp_output_dir):
    """Test that the CLI fails gracefully with invalid input data."""
    # Create a valid config file
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        sectors:
            MFG:
                subsectors:
                    MFG1:
                        name: Primary Manufacturing
                        relative_output_weights:
                            USA: 0.6
                            CHN: 0.7
                            ROW: 0.5
                    MFG2:
                        name: Secondary Manufacturing
                        relative_output_weights:
                            USA: 0.4
                            CHN: 0.3
                            ROW: 0.5
        """
    )

    # Create an invalid input file
    input_path = tmp_path / "invalid_input.csv"
    input_path.write_text("invalid,csv,format\n1,2,3")

    result = cli_runner.invoke(
        disaggregate, [str(config_path), str(input_path), str(temp_output_dir)]
    )
    assert result.exit_code != 0


def test_cli_invalid_prior(cli_runner, tmp_path, temp_output_dir):
    """Test that the CLI fails gracefully with invalid prior information."""
    # Create a valid config file
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        sectors:
            MFG:
                subsectors:
                    MFG1:
                        name: Primary Manufacturing
                        relative_output_weights:
                            USA: 0.6
                            CHN: 0.7
                            ROW: 0.5
                    MFG2:
                        name: Secondary Manufacturing
                        relative_output_weights:
                            USA: 0.4
                            CHN: 0.3
                            ROW: 0.5
        """
    )

    # Create a dummy input file
    input_path = tmp_path / "input.csv"
    input_path.write_text("dummy,data\n1,2")

    # Create an invalid prior file
    prior_path = tmp_path / "invalid_prior.csv"
    prior_path.write_text("invalid,prior,data\n1,2,3")

    result = cli_runner.invoke(
        disaggregate,
        [
            str(config_path),
            str(input_path),
            str(temp_output_dir),
            "--prior-info",
            str(prior_path),
        ],
    )
    assert result.exit_code != 0


def test_cli_log_level(cli_runner, tmp_path, temp_output_dir, caplog):
    """Test that the log level option works."""
    # Create a valid config file
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        sectors:
            MFG:
                subsectors:
                    MFG1:
                        name: Primary Manufacturing
                        relative_output_weights:
                            USA: 0.6
                            CHN: 0.7
                            ROW: 0.5
                    MFG2:
                        name: Secondary Manufacturing
                        relative_output_weights:
                            USA: 0.4
                            CHN: 0.3
                            ROW: 0.5
        """
    )

    # Create a dummy ICIO table
    input_path = tmp_path / "input.csv"
    input_data = (
        "CountryInd,industryInd,USA,USA,CHN,CHN,ROW,ROW\n"
        ",,MFG,SRV,MFG,SRV,MFG,SRV\n"
        "USA,MFG,0.1,0.2,0.3,0.4,0.5,0.6\n"
        "USA,SRV,0.2,0.3,0.4,0.5,0.6,0.7\n"
        "CHN,MFG,0.3,0.4,0.5,0.6,0.7,0.8\n"
        "CHN,SRV,0.4,0.5,0.6,0.7,0.8,0.9\n"
        "ROW,MFG,0.5,0.6,0.7,0.8,0.9,1.0\n"
        "ROW,SRV,0.6,0.7,0.8,0.9,1.0,1.1\n"
        "VA,VA,0.1,0.2,0.3,0.4,0.5,0.6\n"
        "TLS,TLS,0.2,0.3,0.4,0.5,0.6,0.7\n"
        "OUT,OUT,1.0,2.0,3.0,4.0,5.0,6.0\n"
    )
    input_path.write_text(input_data)

    # Run command with DEBUG log level
    with caplog.at_level(logging.DEBUG):
        try:
            result = cli_runner.invoke(
                disaggregate,
                [str(config_path), str(input_path), str(temp_output_dir), "--log-level", "DEBUG"],
                catch_exceptions=False,
            )
        except Exception:
            # We expect an error because our test data is incomplete,
            # but we should still see DEBUG level logs
            pass

    # We should see both INFO and DEBUG level logs
    assert any(record.levelno == logging.INFO for record in caplog.records)
    assert any(record.levelno == logging.DEBUG for record in caplog.records)


def test_cli_real_data(
    cli_runner,
    tmp_path,
    temp_output_dir,
    real_config_file,
    aggregated_data_file,
):
    """Test that the CLI works with real data."""
    # Run the disaggregation
    try:
        cli_runner.invoke(
            disaggregate,
            [
                str(real_config_file),
                str(aggregated_data_file),
                str(temp_output_dir),
                "--log-level",
                "DEBUG",
            ],
            catch_exceptions=True,
        )
    except ValueError:
        # Ignore the Click output stream error
        pass

    # Check that the output file exists
    output_file = temp_output_dir / "disaggregated_table.csv"
    assert output_file.exists()

    # Load the output data and check that it has the expected structure
    # output_data = pd.read_csv(output_file, index_col=[0, 1], header=[0, 1])

    output_icio = ICIOReader.from_csv(output_file)

    output_data = output_icio.data

    assert ("USA", "A01") in output_data.index
    assert ("USA", "A03") in output_data.index
    assert ("ROW", "A01") in output_data.index
    assert ("ROW", "A03") in output_data.index

    assert ("USA", "A01") in output_data.columns
    assert ("USA", "A03") in output_data.columns
    assert ("ROW", "A01") in output_data.columns
    assert ("ROW", "A03") in output_data.columns
