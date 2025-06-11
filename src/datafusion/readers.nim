## File format readers and writer options for DataFusion

import bindings
import common
import arrowschema

# High-level wrappers for read options and writer properties
type
  CSVReadOptions* = object
    handle*: ptr DFCSVReadOptions

  ParquetReadOptions* = object
    handle*: ptr DFParquetReadOptions

  ParquetWriterProperties* = object
    handle*: ptr DFParquetWriterProperties

# CSVReadOptions implementation
proc newCSVReadOptions*(): CSVReadOptions =
  result.handle = df_csv_read_options_new()

proc `=destroy`*(opts: var CSVReadOptions) =
  if opts.handle != nil:
    df_csv_read_options_free(opts.handle)
    opts.handle = nil

proc `hasHeader=`*(opts: CSVReadOptions, value: bool) =
  df_csv_read_options_set_has_header(opts.handle, value)

proc hasHeader*(opts: CSVReadOptions): bool =
  df_csv_read_options_get_has_header(opts.handle)

proc `delimiter=`*(opts: CSVReadOptions, value: char) =
  df_csv_read_options_set_delimiter(opts.handle, value.uint8)

proc delimiter*(opts: CSVReadOptions): char =
  df_csv_read_options_get_delimiter(opts.handle).char

proc `schemaInferMaxRecords=`*(opts: CSVReadOptions, value: uint) =
  df_csv_read_options_set_schema_infer_max_records(opts.handle, value)

proc schemaInferMaxRecords*(opts: CSVReadOptions): uint =
  df_csv_read_options_get_schema_infer_max_records(opts.handle)

proc `fileExtension=`*(opts: CSVReadOptions, value: string) =
  var error: ptr DFError = nil
  let success = df_csv_read_options_set_file_extension(opts.handle, value.cstring, error.addr)
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to set file extension")

proc fileExtension*(opts: CSVReadOptions): string =
  let cstr = df_csv_read_options_get_file_extension(opts.handle)
  if cstr != nil: $cstr else: ""

proc `schema=`*(opts: CSVReadOptions, schema: ArrowSchema) =
  var error: ptr DFError = nil
  let success = df_csv_read_options_set_schema(opts.handle, schema.handle, error.addr)
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to set schema")

proc schema*(opts: CSVReadOptions): ArrowSchema =
  var error: ptr DFError = nil
  result.handle = df_csv_read_options_get_schema(opts.handle, error.addr)
  checkError(error)

# ParquetReadOptions implementation
proc newParquetReadOptions*(): ParquetReadOptions =
  result.handle = df_parquet_read_options_new()

proc `=destroy`*(opts: var ParquetReadOptions) =
  if opts.handle != nil:
    df_parquet_read_options_free(opts.handle)
    opts.handle = nil

proc `fileExtension=`*(opts: ParquetReadOptions, value: string) =
  var error: ptr DFError = nil
  let success = df_parquet_read_options_set_file_extension(opts.handle, value.cstring, error.addr)
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to set file extension")

proc fileExtension*(opts: ParquetReadOptions): string =
  let cstr = df_parquet_read_options_get_file_extension(opts.handle)
  if cstr != nil: $cstr else: ""

proc `pruning=`*(opts: ParquetReadOptions, value: bool) =
  df_parquet_read_options_set_pruning(opts.handle, value)

proc unsetPruning*(opts: ParquetReadOptions) =
  df_parquet_read_options_unset_pruning(opts.handle)

proc isPruningSet*(opts: ParquetReadOptions): bool =
  df_parquet_read_options_is_set_pruning(opts.handle)

proc pruning*(opts: ParquetReadOptions): bool =
  df_parquet_read_options_get_pruning(opts.handle)

# ParquetWriterProperties implementation
proc newParquetWriterProperties*(): ParquetWriterProperties =
  result.handle = df_parquet_writer_properties_new()

proc `=destroy`*(props: var ParquetWriterProperties) =
  if props.handle != nil:
    df_parquet_writer_properties_free(props.handle)
    props.handle = nil

proc `maxRowGroupSize=`*(props: ParquetWriterProperties, size: uint) =
  df_parquet_writer_properties_set_max_row_group_size(props.handle, size)