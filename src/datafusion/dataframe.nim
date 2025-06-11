## DataFrame wrapper for DataFusion

import std/sequtils
import bindings
import common
import arrowschema
import arrowarray

# Forward declaration for ParquetWriterProperties
type ParquetWriterProperties* = object
  handle*: ptr DFParquetWriterProperties

# High-level DataFrame wrapper
type
  DataFrame* = object
    handle*: ptr DFDataFrame

# DataFrame implementation
proc `=destroy`*(df: var DataFrame) =
  if df.handle != nil:
    df_data_frame_free(df.handle)
    df.handle = nil

proc show*(df: DataFrame) =
  ## Show the DataFrame contents to standard output
  var error: ptr DFError = nil
  df_data_frame_show(df.handle, error.addr)
  checkError(error)

proc writeParquet*(df: DataFrame, path: string,
                   properties: ParquetWriterProperties = ParquetWriterProperties()) =
  ## Write the DataFrame contents as Apache Parquet format
  var error: ptr DFError = nil
  let success = df_data_frame_write_parquet(
    df.handle, path.cstring,
    if properties.handle.isNil: nil else: properties.handle,
    error.addr
  )
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to write Parquet file")

proc exportData*(df: DataFrame): tuple[schema: ArrowSchema, batches: seq[ArrowArray]] =
  ## Export the DataFrame to Arrow format
  var error: ptr DFError = nil
  var schema_out: ptr DFArrowSchema = nil
  var batches_out: ptr UncheckedArray[ptr DFArrowArray] = nil

  let count = df_data_frame_export(df.handle, schema_out.addr, batches_out.addr, error.addr)
  checkError(error)

  result.schema.handle = schema_out
  result.batches = newSeq[ArrowArray](count)

  for i in 0..<count:
    # batches_out is now ptr UncheckedArray[ptr DFArrowArray]
    # We can directly index into it like an array
    result.batches[i].handle = batches_out[i]

proc recordCount*(df: DataFrame): int64 =
  ## Get the total number of records in the DataFrame
  let exported = df.exportData()
  var total: int64 = 0
  for batch in exported.batches:
    total += batch.length
  total

proc isEmpty*(df: DataFrame): bool =
  ## Check if the DataFrame is empty
  df.recordCount() == 0