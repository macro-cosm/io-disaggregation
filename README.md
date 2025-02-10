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

### Basic IO Table Operations
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

### Sector Disaggregation
```python
from disag_tools.configurations import DisaggregationConfig
from disag_tools.disaggregation.targets import DisaggregationTargets

# Load disaggregation configuration
config = DisaggregationConfig(
    sectors={
        "A01": {
            "subsectors": {
                "A01a": {
                    "name": "Crop Production",
                    "relative_output_weights": {"USA": 0.4, "ROW": 0.4}
                },
                "A01b": {
                    "name": "Animal Production",
                    "relative_output_weights": {"USA": 0.6, "ROW": 0.6}
                }
            }
        }
    }
)

# Get blocks for disaggregation
blocks = reader.get_reordered_technical_coefficients(["A01"])

# Create targets instance
targets = DisaggregationTargets(blocks, config)

# Get target vector for sector A01
Y = targets.get_target_vector_by_sector_id(("USA", "A01"))
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=disag_tools
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


## Author

José Moran ([@jose-moran](https://github.com/jose-moran)) 