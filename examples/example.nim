## Example usage of the DataFusion high-level Nim API
##
## This demonstrates the idiomatic Nim interface with automatic memory management
## and compile-time type encoding.

import datafusion

proc main() =
  try:
    # Create a session context - automatically cleaned up when it goes out of scope
    var ctx = createSessionContext()

    echo "DataFusion version: ", DATAFUSION_MAJOR, ".", DATAFUSION_MINOR, ".", DATAFUSION_PATCH

    # Demonstrate compile-time type encoding
    echo "Arrow format for int32: ", arrowFormat(int32)
    echo "Arrow format for string: ", arrowFormat(string)
    echo "Arrow format for float64: ", arrowFormat(float64)
    echo "Arrow format for seq[int32]: ", arrowFormat(seq[int32])

    # Load a CSV file (this would fail if the file doesn't exist)
    # ctx.loadCSV("sales", "sales.csv", hasHeader = true, delimiter = ',')

    # Execute SQL query (this would work with actual data)
    # let df = ctx.executeSQL("SELECT * FROM sales WHERE amount > 100")

    # Show results
    # df.show()

    # Write to Parquet
    # df.writeParquet("output.parquet")

    echo "Example completed successfully!"

  except DataFusionError as e:
    echo "DataFusion error [", e.code, "]: ", e.msg
  except Exception as e:
    echo "Unexpected error: ", e.msg

when isMainModule:
  main()