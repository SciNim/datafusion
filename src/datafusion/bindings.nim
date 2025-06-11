## DataFusion Nim bindings - Native equivalent of datafusion.h
##
## This module provides direct Nim equivalents of the C API defined in datafusion.h

import std/[os]
# Linking with the DataFusion library
{.passL: "/usr/lib/x86_64-linux-gnu/libdatafusion.so".}

# Version constants
const
  DATAFUSION_MAJOR* = 21
  DATAFUSION_MINOR* = 0
  DATAFUSION_PATCH* = 0

# Error category enum
type
  DFErrorCode* {.pure.} = enum
    Arrow = "DF_ERROR_CODE_ARROW"
    Parquet = "DF_ERROR_CODE_PARQUET"
    Avro = "DF_ERROR_CODE_AVRO"
    ObjectStore = "DF_ERROR_CODE_OBJECT_STORE"
    IO = "DF_ERROR_CODE_IO"
    SQL = "DF_ERROR_CODE_SQL"
    NotImplemented = "DF_ERROR_CODE_NOT_IMPLEMENTED"
    Internal = "DF_ERROR_CODE_INTERNAL"
    Plan = "DF_ERROR_CODE_PLAN"
    Schema = "DF_ERROR_CODE_SCHEMA"
    Execution = "DF_ERROR_CODE_EXECUTION"
    ResourcesExhausted = "DF_ERROR_CODE_RESOURCES_EXHAUSTED"
    External = "DF_ERROR_CODE_EXTERNAL"
    JIT = "DF_ERROR_CODE_JIT"
    Context = "DF_ERROR_CODE_CONTEXT"
    Substrait = "DF_ERROR_CODE_SUBSTRAIT"

# Arrow C data interface structs
type
  DFArrowSchema* = object
    format*: cstring
    name*: cstring
    metadata*: cstring
    flags*: int64
    n_children*: int64
    children*: ptr UncheckedArray[ptr DFArrowSchema]
    dictionary*: ptr DFArrowSchema
    release*: proc(schema: ptr DFArrowSchema) {.cdecl.}
    private_data*: pointer

  DFArrowArray* = object
    length*: int64
    null_count*: int64
    offset*: int64
    n_buffers*: int64
    n_children*: int64
    buffers*: ptr pointer
    children*: ptr UncheckedArray[ptr DFArrowArray]
    dictionary*: ptr DFArrowArray
    release*: proc(array: ptr DFArrowArray) {.cdecl.}
    private_data*: pointer

{.push header: "datafusion.h".}

# Opaque struct types - these are incomplete types in C
type
  DFCSVReadOptions* = object
  DFDataFrame* = object
  DFError* = object
  DFParquetReadOptions* = object
  DFParquetWriterProperties* = object
  DFSessionContext* = object

# Error handling functions
proc df_error_new*(code: DFErrorCode, message: cstring): ptr DFError {.importc, cdecl.}

proc df_error_free*(error: ptr DFError) {.importc, cdecl.}
  ## Free the given DFError.
  ##
  ## Safety:
  ## - This function should not be called with error that is not created by df_error_new().
  ## - This function should not be called for the same error multiple times.

proc df_error_get_message*(error: ptr DFError): cstring {.importc, cdecl.}
  ## Get a message of this error.
  ##
  ## Safety:
  ## - This function should not be called with error that is not created by df_error_new().
  ## - This function should not be called with error that is freed by df_error_free().

proc df_error_get_code*(error: ptr DFError): DFErrorCode {.importc, cdecl.}
  ## Get a code of this error.
  ##
  ## Safety:
  ## - This function should not be called with error that is not created by df_error_new().
  ## - This function should not be called with error that is freed by df_error_free().

# Parquet writer properties functions
proc df_parquet_writer_properties_new*(): ptr DFParquetWriterProperties {.importc, cdecl.}

proc df_parquet_writer_properties_free*(properties: ptr DFParquetWriterProperties) {.importc, cdecl.}

proc df_parquet_writer_properties_set_max_row_group_size*(
  properties: ptr DFParquetWriterProperties,
  size: uint
) {.importc, cdecl.}

# DataFrame functions
proc df_data_frame_free*(data_frame: ptr DFDataFrame) {.importc, cdecl.}
  ## Free the given DFDataFrame.
  ##
  ## Safety:
  ## - This function should not be called for the same data_frame multiple times.

proc df_data_frame_show*(data_frame: ptr DFDataFrame, error: ptr ptr DFError) {.importc, cdecl.}
  ## Show the given data frame contents to the standard output.

proc df_data_frame_write_parquet*(
  data_frame: ptr DFDataFrame,
  path: cstring,
  writer_properties: ptr DFParquetWriterProperties,
  error: ptr ptr DFError
): bool {.importc, cdecl.}
  ## Write the given data frame contents as Apache Parquet format.

proc df_data_frame_export*(
  data_frame: ptr DFDataFrame,
  c_abi_schema_out: ptr ptr DFArrowSchema,
  c_abi_record_batches_out: ptr ptr UncheckedArray[ptr DFArrowArray],
  error: ptr ptr DFError
): int64 {.importc, cdecl.}

# Session context functions
proc df_session_context_new*(): ptr DFSessionContext {.importc, cdecl.}
  ## Create a new DFSessionContext.
  ##
  ## Returns: A newly created DFSessionContext.
  ## It should be freed by df_session_context_free() when no longer needed.

proc df_session_context_free*(context: ptr DFSessionContext) {.importc, cdecl.}
  ## Free the given DFSessionContext.
  ##
  ## Safety:
  ## - This function should not be called with context that is not created by df_session_context_new().
  ## - This function should not be called for the same context multiple times.

proc df_session_context_sql*(
  context: ptr DFSessionContext,
  sql: cstring,
  error: ptr ptr DFError
): ptr DFDataFrame {.importc, cdecl.}

proc df_session_context_deregister*(
  context: ptr DFSessionContext,
  name: cstring,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_session_context_register_record_batches*(
  context: ptr DFSessionContext,
  name: cstring,
  c_abi_schema: ptr DFArrowSchema,
  c_abi_record_batches: ptr UncheckedArray[ptr DFArrowArray],
  n_record_batches: csize_t,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

# CSV read options functions
proc df_csv_read_options_new*(): ptr DFCSVReadOptions {.importc, cdecl.}

proc df_csv_read_options_free*(options: ptr DFCSVReadOptions) {.importc, cdecl.}

proc df_csv_read_options_set_has_header*(options: ptr DFCSVReadOptions, has_header: bool) {.importc, cdecl.}

proc df_csv_read_options_get_has_header*(options: ptr DFCSVReadOptions): bool {.importc, cdecl.}

proc df_csv_read_options_set_delimiter*(options: ptr DFCSVReadOptions, delimiter: uint8) {.importc, cdecl.}

proc df_csv_read_options_get_delimiter*(options: ptr DFCSVReadOptions): uint8 {.importc, cdecl.}

proc df_csv_read_options_set_schema*(
  options: ptr DFCSVReadOptions,
  schema: ptr DFArrowSchema,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_csv_read_options_get_schema*(
  options: ptr DFCSVReadOptions,
  error: ptr ptr DFError
): ptr DFArrowSchema {.importc, cdecl.}

proc df_csv_read_options_set_schema_infer_max_records*(
  options: ptr DFCSVReadOptions,
  n: uint
) {.importc, cdecl.}

proc df_csv_read_options_get_schema_infer_max_records*(options: ptr DFCSVReadOptions): uint {.importc, cdecl.}

proc df_csv_read_options_set_file_extension*(
  options: ptr DFCSVReadOptions,
  file_extension: cstring,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_csv_read_options_get_file_extension*(options: ptr DFCSVReadOptions): cstring {.importc, cdecl.}

proc df_csv_read_options_set_table_partition_columns*(
  options: ptr DFCSVReadOptions,
  schema: ptr DFArrowSchema,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_csv_read_options_get_table_partition_columns*(
  options: ptr DFCSVReadOptions,
  error: ptr ptr DFError
): ptr DFArrowSchema {.importc, cdecl.}

proc df_session_context_register_csv*(
  context: ptr DFSessionContext,
  name: cstring,
  url: cstring,
  options: ptr DFCSVReadOptions,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

# Parquet read options functions
proc df_parquet_read_options_new*(): ptr DFParquetReadOptions {.importc, cdecl.}

proc df_parquet_read_options_free*(options: ptr DFParquetReadOptions) {.importc, cdecl.}

proc df_parquet_read_options_set_file_extension*(
  options: ptr DFParquetReadOptions,
  file_extension: cstring,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_parquet_read_options_get_file_extension*(options: ptr DFParquetReadOptions): cstring {.importc, cdecl.}

proc df_parquet_read_options_set_table_partition_columns*(
  options: ptr DFParquetReadOptions,
  schema: ptr DFArrowSchema,
  error: ptr ptr DFError
): bool {.importc, cdecl.}

proc df_parquet_read_options_get_table_partition_columns*(
  options: ptr DFParquetReadOptions,
  error: ptr ptr DFError
): ptr DFArrowSchema {.importc, cdecl.}

proc df_parquet_read_options_set_pruning*(options: ptr DFParquetReadOptions, pruning: bool) {.importc, cdecl.}

proc df_parquet_read_options_unset_pruning*(options: ptr DFParquetReadOptions) {.importc, cdecl.}

proc df_parquet_read_options_is_set_pruning*(options: ptr DFParquetReadOptions): bool {.importc, cdecl.}

proc df_parquet_read_options_get_pruning*(options: ptr DFParquetReadOptions): bool {.importc, cdecl.}

proc df_session_context_register_parquet*(
  context: ptr DFSessionContext,
  name: cstring,
  url: cstring,
  options: ptr DFParquetReadOptions,
  error: ptr ptr DFError
): bool {.importc, cdecl.}