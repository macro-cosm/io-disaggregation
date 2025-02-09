# Multi-Sector Input-Output Disaggregation Documentation

This documentation describes the implementation and usage of tools for reading and processing Inter-Country Input-Output (ICIO) tables, with a focus on sector disaggregation.

## Table of Contents

1. [ICIO Tables Overview](#icio-tables-overview)
2. [Data Structure](#data-structure)
3. [Configuration System](#configuration-system)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)

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

## Development

For development guidelines and best practices, please refer to the `.cursorrules` file in the repository root.

---
**Note**: This documentation will be updated as new features and improvements are added to the codebase.
