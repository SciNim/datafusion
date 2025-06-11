## Example: DataFusion Logical Plan API in Nim
##
## This example demonstrates how to build logical plans programmatically
## using a fluent API similar to datafusion-python
import std/strutils
import datafusion

proc main() =
  echo "DataFusion Logical Plan API Example"
  echo "==================================="

  # Create a session context
  let ctx = createSessionContext()

  # Example 1: Simple filtering and projection
  echo "\nExample 1: Filter and Select"

  # This builds a plan equivalent to:
  # SELECT name, salary FROM employees WHERE salary > 50000 LIMIT 10
  let plan1 = ctx.scan("employees")
    .select(col("name"), col("salary"))
    .filter(col("salary") > lit(50000))
    .limit(10)
    .build()

  echo "Generated SQL: ", plan1.query

  # Example 2: Aggregation with grouping
  echo "\nExample 2: Group By and Aggregation"

  # This builds a plan equivalent to:
  # SELECT department, AVG(salary), COUNT(*)
  # FROM employees
  # WHERE active = true
  # GROUP BY department
  # ORDER BY AVG(salary) DESC
  let plan2 = ctx.scan("employees")
    .select(col("department"), avg(col("salary")), count(col("*")))
    .filter(col("active") == lit(true))
    .groupBy(col("department"))
    .orderBy(avg(col("salary")))
    .build()

  echo "Generated SQL: ", plan2.query

  # Example 3: Complex expressions
  echo "\nExample 3: Complex Filtering"

  # This builds a plan equivalent to:
  # SELECT * FROM sales
  # WHERE (amount >= 1000 AND region = 'US') OR priority = 'HIGH'
  let complex_filter = (col("amount") >= lit(1000)) and (col("region") == lit("US"))
  let plan3 = ctx.scan("sales")
    .filter(complex_filter)
    .build()

  echo "Generated SQL: ", plan3.query

  # Example 4: Multi-step plan building
  echo "\nExample 4: Step-by-step Building"

  # Build a plan step by step
  var builder = ctx.scan("orders")
  builder = builder.filter(col("status") == lit("completed"))
  builder = builder.select(col("customer_id"), col("total_amount"), col("order_date"))
  builder = builder.groupBy(col("customer_id"))

  let plan4 = builder.build()
  echo "Generated SQL: ", plan4.query

  # Calculate total order value with tax
  let order_total = (col("price") * col("quantity")) * (lit(1) + col("tax_rate"))

  # Performance metric calculation
  let efficiency = col("output") / (col("hours_worked") + col("overtime_hours"))

  # Complex financial calculation
  let roi = ((col("final_value") - col("initial_investment")) / col("initial_investment")) * lit(100)

  # Use in aggregations
  let plan = ctx.scan("orders")
    .select(
      col("customer_id"),
      sum(col("price") * col("quantity")), # Total order value per customer
      avg(col("discount_percent") * col("order_value")) # Average discount amount
    )
    .groupBy(col("customer_id"))
    .build()

  echo "\nLogical plans built successfully!"
  echo "Note: To execute these plans, you would call plan.execute() after registering tables."

when isMainModule:
  main()