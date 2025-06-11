# Nim GroupedSeries Implementation

This directory contains a native Nim implementation of GroupedSeries that can be exported to Python and integrates with DataFusion.

## Overview

GroupedSeries is a data structure for handling time-series data organized by groups (tags). This Nim implementation provides:

- High-performance native Nim data structures using NumpyArray for storage
- Python interoperability via nimpy
- Integration with DataFusion for SQL-based operations
- Memory-efficient operations with automatic memory management

## Files

- `groupedseries.nim` - Main Nim implementation
- `example.nim` - Nim usage examples
- `python_interface.py` - Python wrapper and examples
- `gsr.py` - Original Python reference implementation

## Key Features

### Core Data Structure

```nim
type
  GroupedSeries* = ref object
    epochs*: NumpyArray[int32]           # timestamps as int32 epochs
    dense_epochs*: Option[NumpyArray[int32]]  # optional dense epochs
    groups*: seq[seq[Tag]]               # list of group tag sequences
    values*: NumpyArray[float64]         # 2D array: (groups, epochs)
    is_interpolated*: NumpyArray[bool]   # boolean flags for interpolation
    # ... additional optional fields for forecasting/anomalies
```

### Key Operations

- **Construction**: `newGroupedSeries()`, `emptyGroupedSeries()`
- **Data Access**: `view()`, `viewGroups()`, `numGroups()`, `numEpochs()`
- **Manipulation**: `addGroup()`, `removeInterpolated()`, `copy()`
- **Export**: Python interop via nimpy

## Usage Examples

### Nim Usage

```nim
import groupedseries
import scinim/scinim/numpyarrays

# Create sample data
let epochs_data = @[1640995200'i32, 1640995260'i32, 1640995320'i32]
let epochs_arr = initNumpyArray[int32](@[epochs_data.len])
for i, epoch in epochs_data:
  epochs_arr{i} = epoch

# Define groups (tags) - list of tuples equivalent
let groups = @[
  @["host:server1", "metric:cpu_usage", "datacenter:us-east"],
  @["host:server2", "metric:cpu_usage", "datacenter:us-east"]
]

# Create values array
let values_arr = initNumpyArray[float64](@[groups.len, epochs_data.len])
values_arr[0, 0] = 50.0; values_arr[0, 1] = 55.0; values_arr[0, 2] = 60.0
values_arr[1, 0] = 45.0; values_arr[1, 1] = 50.0; values_arr[1, 2] = 48.0

# Create is_interpolated flags
let is_interp_arr = initNumpyArray[bool](@[epochs_data.len])
for i in 0..<epochs_data.len:
  is_interp_arr{i} = false

# Create GroupedSeries
let gs = newGroupedSeries(epochs_arr, groups, values_arr, is_interp_arr)

echo gs  # Display info
echo fmt"Groups: {gs.numGroups()}, Epochs: {gs.numEpochs()}"
```

### Python Usage

```python
import numpy as np
from python_interface import NimGroupedSeries

# Create sample data
epochs = np.array([1640995200, 1640995260, 1640995320], dtype=np.int32)
groups = [
    ["host:server1", "metric:cpu_usage", "datacenter:us-east"],
    ["host:server2", "metric:cpu_usage", "datacenter:us-east"]
]
values = np.array([
    [50.0, 55.0, 60.0],  # Server1 CPU
    [45.0, 50.0, 48.0]   # Server2 CPU
], dtype=np.float64)
is_interpolated = np.array([False, False, False], dtype=bool)

# Create GroupedSeries
gs = NimGroupedSeries(epochs, groups, values, is_interpolated)

print(f"Groups: {gs.num_groups()}, Epochs: {gs.num_epochs()}")
print("First group values:", gs.view(0))

# Convert to pandas DataFrame
df = gs.to_dataframe()
print(df)
```

## Building and Compilation

### For Nim Usage Only

```bash
# Compile the example
nim c -r datafusion/src/groupedseries/example.nim
```

### For Python Export

```bash
# Compile as shared library for Python import
nim c --app:lib --out:groupedseries_nim.so \
  -d:nimpy.export \
  datafusion/src/groupedseries/groupedseries.nim

# Then use from Python
python3 datafusion/src/groupedseries/python_interface.py
```

## Dependencies

### Nim Dependencies
- `nimpy` - Python interoperability
- `scinim/scinim/numpyarrays` - NumPy array bindings
- `std/[sequtils, strformat, tables, sugar, algorithm]` - Standard library

### Python Dependencies (for Python interface)
- `numpy` - Array operations
- `pandas` - DataFrame conversion (optional)

## Integration with DataFusion

The GroupedSeries integrates with DataFusion through conversion functions:

```nim
# Convert GroupedSeries to DataFusion DataFrame
let df = gs.toDataFrame(context)

# This enables SQL operations on the time-series data
let result = context.sql("SELECT * FROM df WHERE value > 50")
```

## Performance Benefits

1. **Native Nim Performance**: Compiled code with zero-cost abstractions
2. **Memory Efficiency**: Direct memory access via NumpyArray
3. **Minimal Python Overhead**: Only at API boundaries
4. **SIMD Optimizations**: Potential for vectorized operations

## Design Notes

### Tag System
Groups are represented as sequences of tags, where each tag is a "key:value" string. This mirrors the original Python implementation while providing type safety.

### Memory Management
- Uses Nim's automatic memory management
- NumpyArray handles numpy memory via SharedPtr
- Reference counting for GroupedSeries objects

### Python Interop Strategy
- Nim code exports core functionality
- Python wrapper provides familiar interface
- Falls back to pure Python when Nim module unavailable

## Future Enhancements

1. **Arrow Integration**: Direct Arrow array support for DataFusion
2. **More Operations**: Concatenation, slicing, aggregations
3. **Serialization**: Protobuf/JSON serialization support
4. **Performance**: SIMD optimizations for bulk operations
5. **Type Safety**: More compile-time validation

## Testing

Run the examples to test functionality:

```bash
# Test Nim implementation
nim c -r example.nim

# Test Python interface
python3 python_interface.py
```

## Contributing

When contributing:

1. Maintain compatibility with original Python GroupedSeries API
2. Add appropriate validation and error handling
3. Update both Nim and Python examples
4. Consider memory safety and performance implications