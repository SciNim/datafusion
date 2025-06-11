## DataFusion High-Level Nim API
##
## This module provides idiomatic Nim wrappers around the DataFusion C API
## with automatic memory management and compile-time type encoding.
##
## This is the main module that re-exports everything from the modular structure.

# Import and re-export all modules
import datafusion/common
import datafusion/arrowschema
import datafusion/arrowarray
import datafusion/dataframe
import datafusion/readers
import datafusion/context
import datafusion/logical_plan

# Re-export all public types and functions
export common
export arrowschema
export arrowarray
export dataframe
export readers
export context
export logical_plan