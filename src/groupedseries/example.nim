## Example usage of Nim GroupedSeries implementation

import std/[sequtils, times]
import nimpy
import groupedseries
import ../../../scinim/scinim/numpyarrays

proc main() =
  echo "=== Nim GroupedSeries Example ==="

  # Create sample data
  let epochs_data = @[1640995200'i32, 1640995260'i32, 1640995320'i32, 1640995380'i32]  # 1-minute intervals
  let epochs_arr = initNumpyArray[int32](@[epochs_data.len])
  for i, epoch in epochs_data:
    epochs_arr{i} = epoch

  # Create sample groups (tag sequences)
  let groups = @[
    @["host:server1", "metric:cpu_usage"],
    @["host:server2", "metric:cpu_usage"],
    @["host:server1", "metric:memory_usage"]
  ]

  # Create sample values (groups x epochs)
  let values_arr = initNumpyArray[float64](@[groups.len, epochs_data.len])

  # Fill with sample data
  # Server1 CPU: 50, 55, 60, 58
  values_arr[0, 0] = 50.0; values_arr[0, 1] = 55.0; values_arr[0, 2] = 60.0; values_arr[0, 3] = 58.0
  # Server2 CPU: 45, 50, 48, 52
  values_arr[1, 0] = 45.0; values_arr[1, 1] = 50.0; values_arr[1, 2] = 48.0; values_arr[1, 3] = 52.0
  # Server1 Memory: 75, 78, 80, 77
  values_arr[2, 0] = 75.0; values_arr[2, 1] = 78.0; values_arr[2, 2] = 80.0; values_arr[2, 3] = 77.0

  # Create is_interpolated array (all false for this example)
  let is_interp_arr = initNumpyArray[bool](@[epochs_data.len])
  for i in 0..<epochs_data.len:
    is_interp_arr{i} = false

  # Create GroupedSeries
  let gs = newGroupedSeries(epochs_arr, groups, values_arr, is_interp_arr)

  echo "Created GroupedSeries:"
  echo gs
  echo fmt"Number of groups: {gs.numGroups()}"
  echo fmt"Number of epochs: {gs.numEpochs()}"
  echo fmt"Is empty: {gs.isEmpty()}"

  # Test data access
  echo "\n=== Data Access ==="
  echo "Groups:"
  for i, group in gs.groups:
    echo fmt"  Group {i}: {group}"

  echo "Epochs:"
  for i in 0..<gs.numEpochs():
    echo fmt"  Epoch {i}: {gs.epochs{i}}"

  echo "Values for each group:"
  for group_idx in 0..<gs.numGroups():
    let group_view = gs.view(group_idx)
    echo fmt"  Group {group_idx}: ",
    var values_str: seq[string] = @[]
    for i in 0..<group_view.len:
      values_str.add(fmt"{group_view{i}:.1f}")
    echo values_str.join(", ")

  # Test copy
  echo "\n=== Copy Test ==="
  let gs_copy = gs.copy()
  echo fmt"Original: {gs}"
  echo fmt"Copy: {gs_copy}"

  # Test adding a group
  echo "\n=== Add Group Test ==="
  var gs_mutable = gs.copy()
  let new_group_data = initNumpyArray[float64](@[epochs_data.len])
  new_group_data{0} = 30.0; new_group_data{1} = 32.0; new_group_data{2} = 35.0; new_group_data{3} = 33.0

  gs_mutable.addGroup(@["host:server2", "metric:memory_usage"], new_group_data)
  echo fmt"After adding group: {gs_mutable}"
  echo fmt"New number of groups: {gs_mutable.numGroups()}"

  # Test empty GroupedSeries
  echo "\n=== Empty GroupedSeries Test ==="
  let empty_gs = emptyGroupedSeries()
  echo fmt"Empty GroupedSeries: {empty_gs}"
  echo fmt"Is empty: {empty_gs.isEmpty()}"

when isMainModule:
  main()