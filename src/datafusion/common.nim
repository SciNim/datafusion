## Common types and utilities for DataFusion Nim API

import bindings
# Re-export constants for convenience
export DATAFUSION_MAJOR, DATAFUSION_MINOR, DATAFUSION_PATCH

const ARROW_FLAG_DICTIONARY_ORDERED* = 1
const ARROW_FLAG_NULLABLE* = 2
const ARROW_FLAG_MAP_KEYS_SORTED* = 4

# Exception types for better error handling
type
  DataFusionError* = object of CatchableError
    code*: DFErrorCode

# Error handling helper
proc checkError*(error: ptr DFError) =
  if error != nil:
    let message = $df_error_get_message(error)
    let code = df_error_get_code(error)
    df_error_free(error)
    var exc = newException(DataFusionError, message)
    exc.code = code
    raise exc