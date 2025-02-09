# Multi-Sector Input-Output Disaggregation Documentation

This documentation describes the implementation and usage of tools for reading and processing Inter-Country Input-Output (ICIO) tables, with a focus on sector disaggregation.

## Table of Contents

1. [ICIO Tables Overview](#icio-tables-overview)
2. [Data Structure](#data-structure)
3. [Implementation Details](#implementation-details)
4. [Usage Examples](#usage-examples)

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
from multi_sector_disagg.readers import ICIOReader

# Read full table
reader = ICIOReader.from_csv("path/to/icio_table.csv")

# Read with country selection (others aggregated to ROW)
selected_reader = ICIOReader.from_csv_selection(
    "path/to/icio_table.csv",
    selected_countries=["USA", "CHN"]
)
```

### Data Access

```python
# Access the full table
data = reader.data

# Get specific flows
usa_to_chn = data.loc[("USA", "AGR"), ("CHN", "MFG")]

# Get output values using different methods
output_from_out = reader.output_from_out
output_from_sums = reader.output_from_sums

# Get final demand and intermediate consumption
final_demand = reader.final_demand
intermediate = reader.intermediate_consumption

# Get complete tables
final_demand_table = reader.final_demand_table
intermediate_demand_table = reader.intermediate_demand_table
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
