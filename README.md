# IO Disaggregation Tools

[![Tests](https://github.com/macro-cosm/io-disaggregation/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/macro-cosm/io-disaggregation/actions/workflows/test.yml)

A Python package to disagregate input-output tables. Developed by [José Moran](https://github.com/jose-moran).

## Features

- Read and process ICIO tables from CSV files
- Transform data into standardized multi-index format
- Support country selection and aggregation
- Validate data consistency and structure
- Multiple methods for output computation and validation
- Flexible sector disaggregation with configurable weights
- Support for both single-region and multi-region tables
- Comprehensive block structure for disaggregation problems
- Command-line interface for easy disaggregation tasks

## Installation

```bash
# Clone the repository
git clone git@github.com:macro-cosm/io-disaggregation.git
cd io-disaggregation

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

The package provides a command-line interface for disaggregating IO tables:

```bash
# Basic usage
disaggregate CONFIG_PATH INPUT_PATH OUTPUT_DIR

# Example with all options
disaggregate config.yaml input.csv output/ \
    --prior-info prior_info.csv \
    --final-demand-prior fd_prior.csv \
    --lambda-sparse 1.0 \
    --mu-prior 10.0 \
    --log-level DEBUG
```

Arguments:

- `CONFIG_PATH`: Path to YAML configuration file specifying disaggregation structure
- `INPUT_PATH`: Path to input CSV file containing the IO table
- `OUTPUT_DIR`: Directory to write output files to

Options:

- `--prior-info`: CSV file with prior information for technical coefficients
- `--final-demand-prior`: CSV file with prior information for final demand
- `--lambda-sparse`: Weight for L1 penalty on sparse terms (default: 1.0)
- `--mu-prior`: Weight for L2 penalty on deviation from known terms (default: 10.0)
- `--log-level`: Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

Configuration File Format:

```yaml
sectors:
  A:  # Sector to disaggregate
    subsectors:
      A01:  # First subsector
        name: Agriculture
        relative_output_weights:
          USA: 0.990
          ROW: 0.915
      A03:  # Second subsector
        name: Fishing
        relative_output_weights:
          USA: 0.010
          ROW: 0.085
```

### Python API Usage

#### Basic IO Table Operations

```python
from disag_tools.readers import ICIOReader

# Read full ICIO table
reader = ICIOReader.from_csv("path/to/icio_table.csv")

# Read with country selection (others aggregated to ROW)
selected_reader = ICIOReader.from_csv_selection(
    "path/to/icio_table.csv",
    selected_countries=["USA", "CHN"]
)

# Access different aspects of the data
output = reader.output_from_out
final_demand = reader.final_demand
intermediate = reader.intermediate_consumption
```

#### Sector Disaggregation

```python
from disag_tools.configurations import DisaggregationConfig
from disag_tools.disaggregation.problem import DisaggregationProblem

# Load disaggregation configuration
config = DisaggregationConfig(
    sectors={
        "A": {
            "subsectors": {
                "A01": {
                    "name": "Agriculture",
                    "relative_output_weights": {"USA": 0.990, "ROW": 0.915}
                },
                "A03": {
                    "name": "Fishing",
                    "relative_output_weights": {"USA": 0.010, "ROW": 0.085}
                }
            }
        }
    }
)

# Create and solve disaggregation problem
problem = DisaggregationProblem.from_configuration(
    config=config,
    reader=reader,
    prior_df=None,  # Optional prior information
    final_demand_prior_df=None  # Optional final demand prior
)

# Solve the problem
problem.solve(lambda_sparse=1.0, mu_prior=10.0)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=disag_tools

# Run specific test
pytest tests/test_cli.py::test_cli_real_data -v --log-cli-level=DEBUG
```

### Code Style

The project uses:

- Black for code formatting
- isort for import sorting
- mypy for type checking

Run style checks:

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy disag_tools/
```

## Documentation

For detailed documentation, see the `documentation.md` file in the repository. The documentation includes:

- Comprehensive overview of ICIO tables
- Detailed API reference
- Configuration system guide
- Usage examples and best practices
- Development setup and guidelines
- CLI tool reference and examples

## Author

José Moran ([@jose-moran](https://github.com/jose-moran)) 