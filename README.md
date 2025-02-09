# IO Disaggregation Tools

[![Tests](https://github.com/macro-cosm/io-disaggregation/actions/workflows/test.yml/badge.svg)](https://github.com/macro-cosm/io-disaggregation/actions/workflows/test.yml)

A Python package for reading and processing Inter-Country Input-Output (ICIO) tables, with a focus on sector disaggregation.

## Features

- Read and process ICIO tables from CSV files
- Transform data into standardized multi-index format
- Support country selection and aggregation
- Validate data consistency and structure
- Multiple methods for output computation and validation

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

```python
from multi_sector_disagg.readers import ICIOReader

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

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=multi_sector_disagg
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking

## License

MIT License. See LICENSE file for details.

## Documentation

For detailed documentation, see the `documentation.md` file in the repository. 