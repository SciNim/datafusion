## Test GroupedSeries with Python-style list of tuples

import std/[sequtils, strformat]
import nimpy
import groupedseries
import ../../../scinim/scinim/numpyarrays

proc testPythonStyleGroups() =
  echo "=== Testing Python-style Groups ==="

  # Create sample data
  let epochs_data = @[1640995200'i32, 1640995260'i32, 1640995320'i32, 1640995380'i32]
  let epochs_arr = initNumpyArray[int32](@[epochs_data.len])
  for i, epoch in epochs_data:
    epochs_arr{i} = epoch

  # Define groups as seq[seq[string]] (equivalent to Python list of tuples)
  let groups = @[
    @["host:server1", "metric:cpu_usage", "datacenter:us-east"],
    @["host:server2", "metric:cpu_usage", "datacenter:us-east"],
    @["host:server1", "metric:memory_usage", "datacenter:us-east"],
    @["host:server3", "metric:cpu_usage", "datacenter:us-west"]
  ]

  # Create sample values (4 groups x 4 epochs)
  let values_arr = initNumpyArray[float64](@[groups.len, epochs_data.len])

  # Fill with sample data
  # Server1 CPU
  values_arr[0, 0] = 50.0; values_arr[0, 1] = 55.0; values_arr[0, 2] = 60.0; values_arr[0, 3] = 58.0
  # Server2 CPU
  values_arr[1, 0] = 45.0; values_arr[1, 1] = 50.0; values_arr[1, 2] = 48.0; values_arr[1, 3] = 52.0
  # Server1 Memory
  values_arr[2, 0] = 75.0; values_arr[2, 1] = 78.0; values_arr[2, 2] = 80.0; values_arr[2, 3] = 77.0
  # Server3 CPU
  values_arr[3, 0] = 42.0; values_arr[3, 1] = 47.0; values_arr[3, 2] = 44.0; values_arr[3, 3] = 49.0

  # Create is_interpolated array (all false for this example)
  let is_interp_arr = initNumpyArray[bool](@[epochs_data.len])
  for i in 0..<epochs_data.len:
    is_interp_arr{i} = false

  # Create GroupedSeries with Python-style groups
  let gs = newGroupedSeries(epochs_arr, groups, values_arr, is_interp_arr)

  echo "Created GroupedSeries with Python-style groups:"
  echo gs
  echo fmt"Number of groups: {gs.numGroups()}"
  echo fmt"Number of epochs: {gs.numEpochs()}"

  # Display groups in detail
  echo "\n=== Groups Structure ==="
  for i, group in gs.groups:
    echo fmt"Group {i}: {group}"
    # Parse each tag in the group
    for j, tag in group:
      let parsed = parseTag(tag)
      echo fmt"  Tag {j}: '{tag}' -> key='{parsed.key}', value='{parsed.value}'"

  # Test data access
  echo "\n=== Data Access ==="
  for group_idx in 0..<gs.numGroups():
    let group_view = gs.view(group_idx)
    echo fmt"Group {group_idx} values: ",
    var values_str: seq[string] = @[]
    for i in 0..<group_view.len:
      values_str.add(fmt"{group_view{i}:.1f}")
    echo values_str.join(", ")

  # Test viewGroups with specific groups
  echo "\n=== View Specific Groups ==="
  let target_groups = @[
    @["host:server1", "metric:cpu_usage", "datacenter:us-east"],
    @["host:server2", "metric:cpu_usage", "datacenter:us-east"]
  ]

  let selected_values = gs.viewGroups(target_groups)
  echo fmt"Selected {target_groups.len} groups with shape: {selected_values.shape}"

  # Test adding a new group with Python-style tags
  echo "\n=== Add New Group ==="
  var gs_mutable = gs.copy()
  let new_group_data = initNumpyArray[float64](@[epochs_data.len])
  new_group_data{0} = 35.0; new_group_data{1} = 38.0; new_group_data{2} = 40.0; new_group_data{3} = 37.0

  let new_group_tags = @["host:server4", "metric:memory_usage", "datacenter:us-west"]
  gs_mutable.addGroup(new_group_tags, new_group_data)
  echo fmt"After adding group: {gs_mutable}"
  echo fmt"New number of groups: {gs_mutable.numGroups()}"

  # Display the new group
  echo fmt"New group tags: {gs_mutable.groups[^1]}"

proc testTagParsing() =
  echo "\n=== Testing Tag Parsing ==="

  let test_tags = @[
    "host:server1",
    "metric:cpu_usage",
    "datacenter:us-east",
    "environment:production",
    "simple_tag",  # No colon
    "key:value:with:colons"  # Multiple colons
  ]

  for tag in test_tags:
    let parsed = parseTag(tag)
    let formatted = formatTag(parsed.key, parsed.value)
    echo fmt"'{tag}' -> key='{parsed.key}', value='{parsed.value}' -> '{formatted}'"

when isMainModule:
  testPythonStyleGroups()
  testTagParsing()