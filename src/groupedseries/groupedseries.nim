## Nim native implementation of GroupedSeries
##
## This module provides a native Nim implementation of GroupedSeries
## that can be exported to Python and integrates with DataFusion

import std/[sequtils, strformat, tables, sugar, algorithm, options, strutils, math]
import nimpy
import ../datafusion/[context, dataframe]

# Import numpy array support
import ../../../scinim/scinim/numpyarrays

type
  Tag* = string ## A tag key

  GroupedSeries* = ref object of PyNimObjectExperimental
    ## Nim native GroupedSeries implementation
    epochs*: NumpyArray[int32]           # timestamps as int32 epochs
    dense_epochs*: Option[NumpyArray[int32]]  # optional dense epochs
    groups*: Table[Tag, seq[string]]     # table of tag -> values mapping
    values*: NumpyArray[float64]         # 2D array: (groups, epochs)
    is_interpolated*: NumpyArray[bool]   # boolean flags for interpolation

    # Optional forecast values (from forecast operations)
    forecast_lower_values*: Option[NumpyArray[float64]]
    forecast_upper_values*: Option[NumpyArray[float64]]
    forecast_values*: Option[NumpyArray[float64]]

    # Optional anomaly detection values
    anomalies_lower_values*: Option[NumpyArray[float64]]
    anomalies_upper_values*: Option[NumpyArray[float64]]
    anomalies_scores_values*: Option[NumpyArray[float64]]
    anomalies_ratings_values*: Option[NumpyArray[float64]]

    # Metadata
    keep_interpolated*: bool
    interval*: Option[int]
    # source_info and extra_info omitted for now - can be added later

# Helper proc to parse tag strings into key-value pairs
proc parseTag(tag: string): tuple[key: string, value: string] =
  ## Parse a "key:value" tag string into a tuple
  let parts = tag.split(":", 1)  # Split on first colon only
  if parts.len == 2:
    result = (key: parts[0], value: parts[1])
  else:
    raise newException(ValueError, fmt"Invalid tag string: {tag}")

# Groups setter that accepts Python list of tuples
proc setGroups*(gs: GroupedSeries, py_groups: seq[tuple[tag: string]]) {.exportpy.} =
  ## Set groups from Python list of tuples like [("tag1":"value1",), ("tag1":"value2",), ("tag2":"value3")]
  gs.groups = initTable[Tag, seq[string]]()

  for group in py_groups:
    # Each tuple should be a dict-like object with key-value pairs
    # We'll iterate through the items of each tuple/dict
    let (key, value) = parseTag(group.tag)
    discard gs.groups.hasKeyOrPut(key, @[])
    gs.groups[key].add(value)

# Constructor functions
proc newGroupedSeries(
  epochs: NumpyArray[int32],
  values: NumpyArray[float64],
  is_interpolated: NumpyArray[bool],
  dense_epochs: Option[NumpyArray[int32]] = none(NumpyArray[int32]),
  keep_interpolated: bool = false,
  interval: Option[int] = none(int)
): GroupedSeries {.exportpy.} =
  ## Create a new GroupedSeries with validation
  result = GroupedSeries(
    epochs: epochs,
    dense_epochs: dense_epochs,
    groups: initTable[Tag, seq[string]](),
    values: values,
    is_interpolated: is_interpolated,
    keep_interpolated: keep_interpolated,
    interval: interval
  )

  # Validate dimensions
  if values.shape.len > 0:
    assert values.shape[0] == groups.len,
      fmt"Values shape[0] ({values.shape[0]}) must match groups length ({groups.len})"
    if values.shape.len > 1:
      assert values.shape[1] == epochs.len,
        fmt"Values shape[1] ({values.shape[1]}) must match epochs length ({epochs.len})"

  assert is_interpolated.len == epochs.len,
    fmt"is_interpolated length ({is_interpolated.len}) must match epochs length ({epochs.len})"

proc newGroupedSeries*(
  epochs: NumpyArray[int32],
  groups: Table[Tag, seq[string]],
  values: NumpyArray[float64],
  is_interpolated: NumpyArray[bool],
  dense_epochs: Option[NumpyArray[int32]] = none(NumpyArray[int32]),
  keep_interpolated: bool = false,
  interval: Option[int] = none(int)
): GroupedSeries =

  ## Create a new GroupedSeries with validation
  result = newGroupedSeries(epochs, groups, values, is_interpolated, dense_epochs, keep_interpolated, interval)
  result.groups = groups

# Overloaded constructor that accepts Python-style list of tuples
proc newGroupedSeries*(
  epochs: NumpyArray[int32],
  groups_tuples:  seq[tuple[tag: string]],  # Python list of tuples
  values: NumpyArray[float64],
  is_interpolated: NumpyArray[bool],
  dense_epochs: Option[NumpyArray[int32]] = none(NumpyArray[int32]),
  keep_interpolated: bool = false,
  interval: Option[int] = none(int)
): GroupedSeries {.exportpy.} =

  result = newGroupedSeries(
    epochs, values, is_interpolated,
    dense_epochs, keep_interpolated, interval
  )
  result.setGroups(groups_tuples)

proc emptyGroupedSeries*(
  epochs: Option[NumpyArray[int32]] = none(NumpyArray[int32]),
  groups_tuples:  Option[seq[tuple[tag: string]]] = none(seq[tuple[tag: string]]),  # Python list of tuples
  dense_epochs: Option[NumpyArray[int32]] = none(NumpyArray[int32]),
  keep_interpolated: bool = false,
  interval: Option[int] = none(int)
): GroupedSeries =
  ## Create an empty GroupedSeries
  let epochs_arr = if epochs.isSome: epochs.get else: initNumpyArray[int32](@[])
  let groups_seq = if groups_tuples.isSome: groups_tuples.get else: @[]
  let epochs_len = epochs_arr.len
  let groups_len = groups_seq.len

  # Create empty 2D values array
  let values_arr = initNumpyArray[float64](@[groups_len, epochs_len])
  let is_interp_arr = initNumpyArray[bool](@[epochs_len])

  # Fill values with NaN
  for i in 0..<values_arr.len:
    values_arr{i} = NaN

  # Fill is_interpolated with false
  for i in 0..<is_interp_arr.len:
    is_interp_arr{i} = false

  result = newGroupedSeries(
    epochs_arr, groups_seq, values_arr, is_interp_arr,
    dense_epochs, keep_interpolated, interval
  )

# Basic accessors
proc numGroups*(gs: GroupedSeries): int =
  ## Return number of groups
  gs.groups.len

proc numEpochs*(gs: GroupedSeries): int =
  ## Return number of epochs
  gs.epochs.len

proc numDenseEpochs*(gs: GroupedSeries): int =
  ## Return number of dense epochs
  if gs.dense_epochs.isSome: gs.dense_epochs.get.len else: 0

proc isEmpty*(gs: GroupedSeries): bool =
  ## Check if GroupedSeries is empty
  gs.values.len == 0

proc lastEpoch*(gs: GroupedSeries): int32 =
  ## Get the last epoch timestamp
  assert gs.epochs.len > 0, "Cannot get last epoch of empty GroupedSeries"
  gs.epochs{gs.epochs.len - 1}

# Data access methods
proc view*(gs: GroupedSeries, group_idx: int): NumpyArray[float64] =
  ## Get a view of values for a single group by index
  assert group_idx >= 0 and group_idx < gs.numGroups(),
    fmt"Group index {group_idx} out of range [0, {gs.numGroups()})"

  # Create a view into the values array for this group
  # This is a simplified version - in a full implementation we'd need proper slicing
  let group_values = initNumpyArray[float64](@[gs.numEpochs()])
  for i in 0..<gs.numEpochs():
    group_values{i} = gs.values[group_idx, i]
  result = group_values

proc viewGroups*(gs: GroupedSeries, target_groups: seq[seq[Tag]]): NumpyArray[float64] =
  ## Get values for specific groups in the order requested
  var group_indices: seq[int] = @[]

  for target_group in target_groups:
    var found = false
    for i, existing_group in gs.groups:
      if existing_group == target_group:
        group_indices.add(i)
        found = true
        break
    if not found:
      raise newException(ValueError, fmt"Group {target_group} not found")

  # Create result array
  let result_arr = initNumpyArray[float64](@[group_indices.len, gs.numEpochs()])
  for i, group_idx in group_indices:
    for j in 0..<gs.numEpochs():
      result_arr[i, j] = gs.values[group_idx, j]

  result = result_arr

# Data manipulation methods
proc removeInterpolated*(gs: var GroupedSeries) =
  ## Remove interpolated data points in-place
  if gs.keep_interpolated:
    return

  # Find non-interpolated indices
  var kept_indices: seq[int] = @[]
  for i in 0..<gs.is_interpolated.len:
    if not gs.is_interpolated{i}:
      kept_indices.add(i)

  if kept_indices.len == gs.numEpochs():
    return  # Nothing to remove

  if kept_indices.len == 0:
    # Remove all data - inline implementation
    gs.values = initNumpyArray[float64](@[])
    gs.epochs = initNumpyArray[int32](@[])
    gs.is_interpolated = initNumpyArray[bool](@[])
    gs.groups = @[]
    return

  # Create new arrays with only non-interpolated data
  let new_epochs = initNumpyArray[int32](@[kept_indices.len])
  let new_is_interpolated = initNumpyArray[bool](@[kept_indices.len])
  let new_values = initNumpyArray[float64](@[gs.numGroups(), kept_indices.len])

  for i, kept_idx in kept_indices:
    new_epochs{i} = gs.epochs{kept_idx}
    new_is_interpolated{i} = gs.is_interpolated{kept_idx}
    for g in 0..<gs.numGroups():
      new_values[g, i] = gs.values[g, kept_idx]

  gs.epochs = new_epochs
  gs.is_interpolated = new_is_interpolated
  gs.values = new_values

proc removeAllGroupData*(gs: var GroupedSeries) =
  ## Remove all group data, leaving empty arrays
  gs.values = initNumpyArray[float64](@[])
  gs.epochs = initNumpyArray[int32](@[])
  gs.is_interpolated = initNumpyArray[bool](@[])
  gs.groups = @[]

proc addGroup*(gs: var GroupedSeries, group_name: seq[Tag], data: NumpyArray[float64]) =
  ## Add a new group with data
  assert data.len == gs.numEpochs(),
    fmt"Data length ({data.len}) must match epochs length ({gs.numEpochs()})"

  gs.groups.add(group_name)

  # Create new values array with additional group
  let new_values = initNumpyArray[float64](@[gs.numGroups(), gs.numEpochs()])

  # Copy existing data
  for g in 0..<(gs.numGroups() - 1):
    for e in 0..<gs.numEpochs():
      new_values[g, e] = gs.values[g, e]

  # Add new group data
  let new_group_idx = gs.numGroups() - 1
  for e in 0..<gs.numEpochs():
    new_values[new_group_idx, e] = data{e}

  gs.values = new_values

proc copy*(gs: GroupedSeries): GroupedSeries =
  ## Create a deep copy of the GroupedSeries
  result = GroupedSeries(
    epochs: gs.epochs,  # NumpyArray should handle copying
    dense_epochs: gs.dense_epochs,
    groups: gs.groups,  # seq is copied by value
    values: gs.values,
    is_interpolated: gs.is_interpolated,
    forecast_lower_values: gs.forecast_lower_values,
    forecast_upper_values: gs.forecast_upper_values,
    forecast_values: gs.forecast_values,
    anomalies_lower_values: gs.anomalies_lower_values,
    anomalies_upper_values: gs.anomalies_upper_values,
    anomalies_scores_values: gs.anomalies_scores_values,
    anomalies_ratings_values: gs.anomalies_ratings_values,
    keep_interpolated: gs.keep_interpolated,
    interval: gs.interval
  )

# Utility methods for debugging/display
proc `$`*(gs: GroupedSeries): string =
  ## String representation for debugging
  fmt"<GroupedSeries(n_epochs={gs.numEpochs()}, n_groups={gs.numGroups()}, " &
     fmt"values_shape=({gs.values.shape}), interval={gs.interval})>"

# Python export functionality
proc exportToPython*() {.exportpy.} =
  ## Export GroupedSeries type and functions to Python

  # Export the type itself
  proc createGroupedSeries(
    epochs: PyObject,
    groups: PyObject,
    values: PyObject,
    is_interpolated: PyObject
  ): GroupedSeries {.exportpy: "GroupedSeries".} =
    let epochs_arr = asNumpyArray[int32](epochs)
    let values_arr = asNumpyArray[float64](values)
    let is_interp_arr = asNumpyArray[bool](is_interpolated)

    result = newGroupedSeries(epochs_arr, groups, values_arr, is_interp_arr)

  proc createEmptyGroupedSeries(): GroupedSeries {.exportpy: "empty_grouped_series".} =
    result = emptyGroupedSeries()

  # Export key methods
  proc getNumGroups(gs: GroupedSeries): int {.exportpy: "num_groups".} =
    gs.numGroups()

  proc getNumEpochs(gs: GroupedSeries): int {.exportpy: "num_epochs".} =
    gs.numEpochs()

  proc getIsEmpty(gs: GroupedSeries): bool {.exportpy: "is_empty".} =
    gs.isEmpty()

  proc getValues(gs: GroupedSeries): PyObject {.exportpy: "get_values".} =
    result = gs.values.obj()

  proc getEpochs(gs: GroupedSeries): PyObject {.exportpy: "get_epochs".} =
    result = gs.epochs.obj()

  proc getGroups(gs: GroupedSeries): seq[seq[string]] {.exportpy: "get_groups".} =
    gs.groups

# Integration with DataFusion (conversion functions)
when declared(DataFrame) and declared(SessionContext):
  import ../datafusion/arrowarray
  import ../datafusion/arrowschema

  proc toDataFrame*(gs: GroupedSeries, context: SessionContext): DataFrame =
    ## Convert GroupedSeries to DataFusion DataFrame
    ## This is a simplified version - full implementation would need proper Arrow integration

    if gs.numGroups() == 0:
      if gs.values.len > 0:
        raise newException(ValueError,
          fmt"Unexpected - GroupedSeries with zero groups but values shape {gs.values.shape}")

      # Create empty table
      let empty_sql = "SELECT CAST(NULL AS BIGINT) as epoch, CAST(NULL AS DOUBLE) as value, CAST(NULL AS BOOLEAN) as is_interpolated WHERE FALSE"
      return context.sql(empty_sql)

    # For now, convert to a simple SQL query
    # In a full implementation, this would create proper Arrow tables
    var sql_parts: seq[string] = @[]

    for group_idx in 0..<gs.numGroups():
      let group = gs.groups[group_idx]
      for epoch_idx in 0..<gs.numEpochs():
        let epoch = gs.epochs{epoch_idx}
        let value = gs.values[group_idx, epoch_idx]
        let is_interp = gs.is_interpolated{epoch_idx}

        var group_cols: seq[string] = @[]
        for tag in group:
          let parsed = parseTag(tag)
          if parsed.key.len > 0:
            group_cols.add(fmt"'{parsed.value}' as key_{parsed.key}")

        let group_sql = group_cols.join(", ")
        let row_sql = fmt"SELECT {epoch} as epoch, {value} as value, {is_interp} as is_interpolated" &
                     (if group_sql.len > 0: ", " & group_sql else: "")
        sql_parts.add(row_sql)

    let full_sql = sql_parts.join(" UNION ALL ")
    result = context.sql(full_sql)

# Export the module
when isMainModule:
  exportToPython()
