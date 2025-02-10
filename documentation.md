# Multi-Sector Input-Output Disaggregation Documentation

This documentation describes the implementation and usage of tools for reading and processing Inter-Country Input-Output (ICIO) tables, with a focus on sector disaggregation.

## Table of Contents

1. [ICIO Tables Overview](#icio-tables-overview)
2. [Data Structure](#data-structure)
3. [Configuration System](#configuration-system)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Development Environment](#development-environment)

## ICIO Tables Overview

Inter-Country Input-Output (ICIO) tables represent economic flows between industries and countries in a standardized format. These tables capture:

- **Intermediate consumption**: Industry-to-industry flows within and between countries
- **Final demand**: Consumption by households, government, and other end users
- **Value added**: Additional economic value created by each industry
- **Output**: Total output of each industry, available both as a row and as the sum of intermediate consumption and final demand

All flows are measured in current million USD.

### Special Elements

The tables include several special elements denoted by specific prefixes:

- `VA`: Value added
- `TLS`: Taxes less subsidies
- `HFCE`: Household final consumption expenditure
- `NPISH`: Non-Profit Institutions Serving Households
- `GGFC`: Government final consumption expenditure
- `GFCF`: Gross fixed capital formation
- `INVNT`: Changes in inventories
- `NONRES`: Direct purchases by non-residents
- `FD`: Total final demand including discrepancies
- `DPABR`: Direct purchases abroad by residents
- `OUT`: Total output (available both as row and column)

## Data Structure

### CSV Format

The ICIO tables are provided in CSV format with a specific structure:

```txt
CountryCol,,USA,USA,CHN,CHN,...
industryCol,,AGR,MFG,AGR,MFG,...
CountryInd,industryInd,,,,,
USA,AGR,value,value,value,value,...
USA,MFG,value,value,value,value,...
...
```

Where:

- First row: Country codes for columns
- Second row: Industry codes for columns
- Third row: Headers for index (CountryInd, industryInd)
- Remaining rows: Data with country-industry pairs and their corresponding values

### Internal Representation

The data is transformed into a pandas DataFrame with MultiIndex structure:

- Index: (CountryInd, industryInd) pairs
- Columns: (CountryInd, industryInd) pairs
- Values: Flow values in current million USD

## Configuration System

The package uses a robust configuration system based on Pydantic models to specify disaggregation rules.

### Configuration Models

1. **SubsectorConfig**
   ```python
   class SubsectorConfig(BaseModel):
       name: str  # Human-readable name
       relative_output_weight: float  # Must be between 0 and 1
   ```

2. **RegionConfig**
   ```python
   class RegionConfig(BaseModel):
       name: str  # Human-readable name
       sector_weights: dict[str, float]  # Sector code to weight mapping
   ```

3. **SectorConfig**
   ```python
   class SectorConfig(BaseModel):
       subsectors: dict[str, SubsectorConfig]  # Subsector code to config mapping
   ```

4. **CountryConfig**
   ```python
   class CountryConfig(BaseModel):
       regions: dict[str, RegionConfig]  # Region code to config mapping
   ```

5. **DisaggregationConfig**
   ```python
   class DisaggregationConfig(BaseModel):
       countries: dict[str, CountryConfig] | None  # Country disaggregation
       sectors: dict[str, SectorConfig] | None  # Sector disaggregation
   ```

### Configuration Files

The system uses YAML files for configuration:

1. **Sector Disaggregation** (sector_disagg_example.yaml):
   ```yaml
   sectors:
     MFG:  # Sector to disaggregate
       subsectors:
         MFG1:
           name: "Primary Manufacturing"
           relative_output_weight: 0.6
         MFG2:
           name: "Secondary Manufacturing"
           relative_output_weight: 0.4
   ```

2. **Country Disaggregation** (country_disagg_example.yaml):
   ```yaml
   countries:
     USA:  # Country to disaggregate
       regions:
         USA1:
           name: "Eastern United States"
           sector_weights:
             AGR: 0.45  # 45% of USA's agriculture is in the East
             MFG: 0.60  # 60% of USA's manufacturing is in the East
         USA2:
           name: "Western United States"
           sector_weights:
             AGR: 0.55  # 55% of USA's agriculture is in the West
             MFG: 0.40  # 40% of USA's manufacturing is in the West
   ```

### Validation Rules

The configuration system enforces several validation rules:

1. **Subsector Weights**
   - Must sum to 1 for each sector being disaggregated
   - Must be between 0 and 1

2. **Region Weights**
   - Must sum to 1 for each sector across all regions of a country
   - Must be between 0 and 1
   - All sectors must have weights defined in all regions

3. **General Rules**
   - At least one type of disaggregation (country or sector) must be specified
   - Names must be provided for all regions and subsectors
   - No duplicate codes allowed

## Implementation Details

### ICIOReader Class

The `ICIOReader` class provides functionality to:

1. Read and parse ICIO tables from CSV files
2. Transform data into a standardized multi-index format
3. Validate data consistency and structure
4. Support country selection and aggregation
5. Compute and validate output values through multiple methods

Key features:

- Proper handling of special elements (VA, TLS, etc.)
- Support for country aggregation into Rest of World (ROW)
- Validation of data structure and consistency
- Efficient handling of large datasets
- Multiple methods for output computation and validation

### Properties

The class provides several properties for accessing different aspects of the data:

1. `output_from_out`: Output values read directly from the OUT row
2. `output_from_sums`: Output computed as the sum of intermediate consumption and final demand
3. `final_demand_table`: Complete final demand table with all categories
4. `final_demand_totals`: Total final demand values (sum across all categories)
5. `intermediate_demand_table`: Table of industry-to-industry flows
6. `intermediate_consumption`: Total intermediate consumption values
7. `final_demand`: Alias for final_demand_totals (for backward compatibility)
8. `technical_coefficients`: Matrix of input coefficients

### Data Validation

The implementation includes several validation checks:

- Verification of MultiIndex structure
- Checking for missing or infinite values
- Validation of row and column sums
- Monitoring of negative values (which may be valid in some cases)
- Consistency checks between different output computation methods

### Country Aggregation

When performing country selection and aggregation:

1. Countries not in the selection are aggregated into a "ROW" (Rest of World) category
2. Special elements (OUT, VA, etc.) are preserved and mapped to themselves
3. All values are properly aggregated while maintaining the table's structure
4. Both row and column indices are consistently updated

### DisaggregationBlocks Class

The `DisaggregationBlocks` class handles the block structure of disaggregation problems, supporting both single-region and multi-region Input-Output tables. It maintains the mapping between sector indices and their codes/names, and provides methods to access various blocks in the mathematical formulation.

#### Key Features

- Support for both single-region and multi-region IO tables
- Proper handling of sector ordering and uniqueness
- Efficient block extraction for disaggregation calculations
- Comprehensive sector information tracking

#### Data Structures

1. **SectorInfo**
   ```python
   class SectorInfo(NamedTuple):
       index: int          # The index n in the mathematical formulation
       sector_id: SectorId # Sector identifier (code or country-code pair)
       name: str          # Human readable name
       k: int            # Number of subsectors this splits into
   ```

2. **SectorId Type**
   - Single-region tables: `str` (sector code)
   - Multi-region tables: `tuple[str, str]` (country, sector)

#### Properties

- `N`: Total number of sectors
- `K`: Number of sectors being disaggregated
- `M`: Total number of subsectors after disaggregation
- `is_multi_region`: Whether this is a multi-region disaggregation

#### Block Access Methods

1. **A₀ Block**
   ```python
   def get_A0(self) -> np.ndarray:
       """Get the A₀ block (undisaggregated sectors)."""
   ```

2. **B^n Block**
   ```python
   def get_B(self, n: int) -> np.ndarray:
       """Get the B^n block (undisaggregated to sector n)."""
   ```

3. **C^n Block**
   ```python
   def get_C(self, n: int) -> np.ndarray:
       """Get the C^n block (sector n to undisaggregated)."""
   ```

4. **D^{nℓ} Block**
   ```python
   def get_D(self, n: int, l: int) -> np.ndarray:
       """Get the D^{nℓ} block (sector n to sector ℓ)."""
   ```

#### Matrix Reordering

The class implements a sophisticated reordering strategy for technical coefficient matrices:

1. **Single-region Tables**
   - Undisaggregated sectors are sorted alphabetically
   - Disaggregated sectors maintain the order specified in the configuration

2. **Multi-region Tables**
   - Undisaggregated sectors are sorted by country, then sector
   - Disaggregated sectors maintain their specified order while ensuring:
     - Each sector's own country-sector pair appears first
     - Additional country-sector pairs for the same sector follow
     - No duplicate pairs are included
     - Original sector ordering is preserved

#### Usage Example

```python
# Create blocks from technical coefficients
blocks = DisaggregationBlocks.from_technical_coefficients(
    tech_coef=technical_coefficients,
    sectors_to_disaggregate=[
        ((country, sector), name, k),  # Multi-region
        # OR
        (sector_code, name, k),        # Single-region
    ]
)

# Access blocks
A0 = blocks.get_A0()
B1 = blocks.get_B(1)  # B block for first sector
C1 = blocks.get_C(1)  # C block for first sector
D12 = blocks.get_D(1, 2)  # D block between sectors 1 and 2

# Get sector information
sector_info = blocks.get_sector_info(1)
print(f"Sector {sector_info.name} splits into {sector_info.k} subsectors")
```

### DisaggregationTargets Class

The `DisaggregationTargets` class is responsible for constructing target vectors for disaggregation problems. For each sector `n` being disaggregated, it constructs a target vector Y^n that contains:

1. B^n: Original flows from undisaggregated sectors to sector n
2. C^n: Original flows from sector n to undisaggregated sectors
3. D^n: Original flows between sector n and other disaggregated sectors
4. w^n: Relative output weights for sector n's subsectors

#### Key Features

- Support for both single-region and multi-region IO tables
- Flexible sector identification using indices or sector IDs
- Comprehensive weight validation and error handling
- Efficient block extraction and vector construction

#### Methods

1. **get_weights**
   ```python
   def get_weights(self, sector_info: SectorInfo) -> np.ndarray:
       """Get the relative output weights for a sector's subsectors."""
   ```
   - Returns array of weights, one per subsector
   - Handles both single-region and multi-region cases
   - Validates weight consistency and completeness

2. **get_target_vector**
   ```python
   def get_target_vector(self, n: int) -> np.ndarray:
       """Construct the target vector Y^n for sector n."""
   ```
   - Constructs target vector using sector index (1-based)
   - Returns concatenated array [B^n, C^n, D^n, w^n]
   - Validates sector index and configuration

3. **get_target_vector_by_sector_id**
   ```python
   def get_target_vector_by_sector_id(self, sector_id: SectorId) -> np.ndarray:
       """Construct the target vector Y^n for a sector identified by its ID."""
   ```
   - Accepts sector ID (str for single-region, tuple[str, str] for multi-region)
   - Returns same structure as get_target_vector
   - Raises ValueError if sector ID not found

#### Usage Examples

```python
from disag_tools.disaggregation.targets import DisaggregationTargets
from disag_tools.configurations.config import DisaggregationConfig

# Create targets instance
targets = DisaggregationTargets(blocks, config)

# Get target vector using index
Y1 = targets.get_target_vector(1)  # First sector

# Get target vector using sector ID
# Single region case
Y_agr = targets.get_target_vector_by_sector_id("A01")

# Multi-region case
Y_usa_agr = targets.get_target_vector_by_sector_id(("USA", "A01"))
```

## Usage Examples

### Basic Usage

```python
from disag_tools.readers import ICIOReader

# Read full table
reader = ICIOReader.from_csv("path/to/icio_table.csv")

# Read with country selection (others aggregated to ROW)
selected_reader = ICIOReader.from_csv_selection(
    "path/to/icio_table.csv",
    selected_countries=["USA", "CHN"]
)
```

### Loading Configuration

```python
from disag_tools.configurations import DisaggregationConfig
import yaml

# Load sector disaggregation config
with open("sector_disagg_example.yaml") as f:
    config = DisaggregationConfig(**yaml.safe_load(f))

# Get disaggregation mapping
mapping = config.get_disagg_mapping()
```

### Accessing Data

```python
# Get technical coefficients
tech_coef = reader.technical_coefficients

# Get final demand
final_demand = reader.final_demand

# Get intermediate consumption
intermediate = reader.intermediate_consumption
```

### Data Validation

```python
# Validate data structure and contents
reader.validate_data()  # Returns True if valid, raises ValueError if not

# Access validated subsets of data
int_table = reader.intermediate_demand_table  # Validates before returning
fd_table = reader.final_demand_table  # Ensures non-negative values
```

## Development Environment

### Package Configuration

The project uses modern Python packaging tools and configurations:

1. **pyproject.toml**
   ```toml
   [build-system]
   requires = ["setuptools>=45", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "disag_tools"
   version = "0.1.0"
   description = "Tools for disaggregating IO tables"
   authors = [{name = "Author Name", email = "author@example.com"}]
   readme = "README.md"
   requires-python = ">=3.11"
   license = {text = "MIT"}
   dependencies = [
       "numpy>=1.21.0",
       "pandas>=1.3.0",
       "cvxpy>=1.2.0",
       "pyyaml>=6.0.0",
       "pydantic>=2.5.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7.0.0",
       "pytest-cov>=4.1.0",
       "black>=24.1.0",
       "isort>=5.12.0",
       "mypy>=1.8.0",
   ]
   ```

2. **Development Tools Configuration**
   ```toml
   [tool.black]
   line-length = 100
   target-version = ['py311']
   include = '\.pyi?$'

   [tool.isort]
   profile = "black"
   line_length = 100
   multi_line_output = 3
   include_trailing_comma = true
   force_grid_wrap = 0
   use_parentheses = true
   ensure_newline_before_comments = true

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   addopts = "-v --cov=disag_tools --cov-report=term-missing"
   filterwarnings = [
       "ignore::DeprecationWarning",
       "ignore::UserWarning",
   ]
   ```

### Development Setup

1. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Installation**
   ```bash
   # Install package with development dependencies
   pip install -e ".[dev]"
   ```

3. **Running Tests**
   ```bash
   # Run all tests with coverage
   pytest tests/ --cov=disag_tools
   ```

4. **Code Formatting**
   ```bash
   # Format code
   black .
   
   # Sort imports
   isort .
   
   # Type checking
   mypy disag_tools/
   ```

### Best Practices

1. **Package Structure**
   - Use `pyproject.toml` for all package configuration
   - Avoid `setup.py` unless required for legacy reasons
   - Keep dependencies up to date and version-pinned
   - Use optional dependencies for development tools

2. **Development Workflow**
   - Always work in a virtual environment
   - Run tests before committing changes
   - Format code using black and isort
   - Check types with mypy
   - Update documentation when adding features

3. **Testing**
   - Write tests for all new functionality
   - Maintain high test coverage
   - Use pytest fixtures for test data
   - Include both unit and integration tests

4. **Documentation**
   - Keep docstrings up to date
   - Follow Google docstring format
   - Include type hints for all functions
   - Document complex algorithms and design decisions

---
**Note**: This documentation will be updated as new features and improvements are added to the codebase.
