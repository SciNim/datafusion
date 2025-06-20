## Session Context wrapper for DataFusion

import bindings
import common
import dataframe
import readers

# High-level SessionContext wrapper
type
  SessionContext* = object
    handle*: ptr DFSessionContext

proc `=wasMoved`*(ctx: var SessionContext) =
  ctx.handle = nil

proc `=destroy`*(ctx: var SessionContext) =
  if ctx.handle != nil:
    df_session_context_free(ctx.handle)


# SessionContext implementation
proc newSessionContext*(): ref SessionContext =
  result = new(SessionContext)
  result.handle = df_session_context_new()

proc sql*(ctx: ref SessionContext, query: string): DataFrame =
  ## Execute a SQL query and return the result as a DataFrame
  var error: ptr DFError = nil
  result.handle = df_session_context_sql(ctx.handle, query.cstring, error.addr)
  checkError(error)

proc registerCSV*(ctx: ref SessionContext, name: string, url: string,
                  options: CSVReadOptions = CSVReadOptions()) =
  ## Register a CSV file/URL as a table in the session context
  var error: ptr DFError = nil
  let success = df_session_context_register_csv(
    ctx.handle, name.cstring, url.cstring,
    if options.handle.isNil: nil else: options.handle,
    error.addr
  )
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to register CSV")

proc registerParquet*(ctx: ref SessionContext, name: string, url: string,
                      options: ParquetReadOptions = ParquetReadOptions()) =
  ## Register a Parquet file/URL as a table in the session context
  var error: ptr DFError = nil
  let success = df_session_context_register_parquet(
    ctx.handle, name.cstring, url.cstring,
    if options.handle.isNil: nil else: options.handle,
    error.addr
  )
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to register Parquet")

proc deregister*(ctx: ref SessionContext, name: string) =
  ## Remove a table from the session context
  var error: ptr DFError = nil
  let success = df_session_context_deregister(ctx.handle, name.cstring, error.addr)
  checkError(error)
  if not success:
    raise newException(DataFusionError, "Failed to deregister table")

# High-level convenience functions
proc createSessionContext*(): ref SessionContext =
  ## Create a new DataFusion session context with automatic cleanup
  newSessionContext()

proc executeSQL*(ctx: ref SessionContext, query: string): DataFrame =
  ## Execute a SQL query and return the result as a DataFrame
  ctx.sql(query)

proc loadCSV*(ctx: ref SessionContext, tableName: string, filePath: string,
              hasHeader: bool = true, delimiter: char = ','): void =
  ## Load a CSV file into the session context as a table
  var options = newCSVReadOptions()
  options.hasHeader = hasHeader
  options.delimiter = delimiter
  ctx.registerCSV(tableName, filePath, options)

proc loadParquet*(ctx: ref SessionContext, tableName: string, filePath: string): void =
  ## Load a Parquet file into the session context as a table
  var options = newParquetReadOptions()
  ctx.registerParquet(tableName, filePath, options)