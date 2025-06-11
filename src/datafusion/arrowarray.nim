## Arrow Array wrapper for DataFusion

import bindings
import common

export DataFusionError

# High-level ArrowArray wrapper
type
  ArrowArray* = object
    handle*: ptr DFArrowArray

# ArrowArray implementation
proc `=destroy`*(array: var ArrowArray) =
  if array.handle != nil and array.handle.release != nil:
    array.handle.release(array.handle)
    array.handle = nil

proc length*(array: ArrowArray): int64 =
  ## Get the logical length of the array (number of values)
  if array.handle != nil:
    array.handle.length
  else:
    0

proc nullCount*(array: ArrowArray): int64 =
  ## Get the number of null values in the array
  ## Returns -1 if not computed (allows lazy computation)
  if array.handle != nil:
    array.handle.null_count
  else:
    0

proc offset*(array: ArrowArray): int64 =
  ## Get the logical offset into the array (for slicing)
  if array.handle != nil:
    array.handle.offset
  else:
    0

proc bufferCount*(array: ArrowArray): int =
  ## Get the number of buffers in the buffers array
  if array.handle != nil:
    array.handle.n_buffers.int
  else:
    0

proc childCount*(array: ArrowArray): int =
  ## Get the number of child arrays for nested types
  if array.handle != nil:
    array.handle.n_children.int
  else:
    0

proc isEmpty*(array: ArrowArray): bool =
  ## Check if the array is empty
  array.length == 0

proc hasNulls*(array: ArrowArray): bool =
  ## Check if the array contains any null values
  let nc = array.nullCount
  nc > 0 or nc == -1  # -1 means unknown, so we assume it might have nulls