## Logical Plan API for DataFusion
##
## This module provides a logical plan builder API similar to datafusion-python
## Built on top of the DataFusion C API via SessionContext and DataFrame operations

import bindings
import context
import dataframe
import std/[strutils, sequtils]
from arrowschema import ArrowSchema
from arrowarray import ArrowArray

# Expression types for building logical plans
type
  LogicalExpr* = object
    ## Represents a logical expression in a DataFusion plan
    expr_type*: ExprType
    column_name*: string
    literal_value*: string
    children*: seq[LogicalExpr]
    operator*: BinaryOperator  # Track the actual operator used

  ExprType* = enum
    ## Types of logical expressions
    Column
    Literal
    BinaryOp
    Function
    Aggregate

  BinaryOperator* = enum
    ## Binary operators for expressions
    Equal, NotEqual, LessThan, LessThanOrEqual
    GreaterThan, GreaterThanOrEqual, And, Or
    Plus, Minus, Multiply, Divide

  LogicalPlan* = object
    ## Represents a logical plan that can be executed
    context*: SessionContext
    query*: string  # SQL representation of the plan

# Expression builders (similar to datafusion-python's col(), lit(), etc.)
proc col*(name: string): LogicalExpr =
  ## Create a column reference expression
  result = LogicalExpr(
    expr_type: Column,
    column_name: name
  )

proc lit*(value: SomeNumber | string | bool): LogicalExpr =
  ## Create a literal value expression
  result = LogicalExpr(
    expr_type: Literal,
    literal_value: $value
  )

proc `==`*(left, right: LogicalExpr): LogicalExpr =
  ## Equal comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Equal
  )

proc `!=`*(left, right: LogicalExpr): LogicalExpr =
  ## Not equal comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: NotEqual
  )

proc `<`*(left, right: LogicalExpr): LogicalExpr =
  ## Less than comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: LessThan
  )

proc `<=`*(left, right: LogicalExpr): LogicalExpr =
  ## Less than or equal comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: LessThanOrEqual
  )

proc `>`*(left, right: LogicalExpr): LogicalExpr =
  ## Greater than comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: GreaterThan
  )

proc `>=`*(left, right: LogicalExpr): LogicalExpr =
  ## Greater than or equal comparison
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: GreaterThanOrEqual
  )

proc `and`*(left, right: LogicalExpr): LogicalExpr =
  ## Logical AND operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: And
  )

proc `or`*(left, right: LogicalExpr): LogicalExpr =
  ## Logical OR operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Or
  )

# Arithmetic operators
proc `+`*(left, right: LogicalExpr): LogicalExpr =
  ## Addition operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Plus
  )

proc `-`*(left, right: LogicalExpr): LogicalExpr =
  ## Subtraction operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Minus
  )

proc `*`*(left, right: LogicalExpr): LogicalExpr =
  ## Multiplication operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Multiply
  )

proc `/`*(left, right: LogicalExpr): LogicalExpr =
  ## Division operation
  result = LogicalExpr(
    expr_type: BinaryOp,
    children: @[left, right],
    operator: Divide
  )

proc getOperatorSymbol(expr: LogicalExpr): string =
  ## Get SQL operator symbol for binary operations
  case expr.operator:
  of Equal: "="
  of NotEqual: "!="
  of LessThan: "<"
  of LessThanOrEqual: "<="
  of GreaterThan: ">"
  of GreaterThanOrEqual: ">="
  of And: "AND"
  of Or: "OR"
  of Plus: "+"
  of Minus: "-"
  of Multiply: "*"
  of Divide: "/"

# Convert expressions to SQL strings
proc toSql(expr: LogicalExpr): string =
  ## Convert a logical expression to SQL string
  case expr.expr_type:
  of Column:
    return expr.column_name
  of Literal:
    # Handle string literals with quotes
    if expr.literal_value.startsWith("'") or expr.literal_value.all(proc(c: char): bool = c.isDigit or c == '.'):
      return expr.literal_value
    else:
      return "'" & expr.literal_value & "'"
  of BinaryOp:
    if expr.children.len == 2:
      return "(" & expr.children[0].toSql & " " & getOperatorSymbol(expr) & " " & expr.children[1].toSql & ")"
  else:
    return ""

# LogicalPlan operations (similar to DataFrame API)
type
  LogicalPlanBuilder* = object
    ## Builder for constructing logical plans
    context*: SessionContext
    current_plan*: string
    table_name*: string

proc scan*(context: SessionContext, table_name: string): LogicalPlanBuilder =
  ## Create a table scan logical plan
  result = LogicalPlanBuilder(
    context: context,
    table_name: table_name,
    current_plan: "SELECT * FROM " & table_name
  )

proc select*(builder: LogicalPlanBuilder, columns: varargs[LogicalExpr]): LogicalPlanBuilder =
  ## Add projection (SELECT) to the logical plan
  result = builder
  var col_list: seq[string] = @[]
  for col_expr in columns:
    col_list.add(col_expr.toSql)

  # Replace the SELECT clause
  let select_part = "SELECT " & col_list.join(", ")
  let from_idx = builder.current_plan.find(" FROM ")
  if from_idx != -1:
    result.current_plan = select_part & builder.current_plan[from_idx..^1]

proc filter*(builder: LogicalPlanBuilder, condition: LogicalExpr): LogicalPlanBuilder =
  ## Add WHERE clause to the logical plan
  result = builder
  result.current_plan = builder.current_plan & " WHERE " & condition.toSql

proc groupBy*(builder: LogicalPlanBuilder, columns: varargs[LogicalExpr]): LogicalPlanBuilder =
  ## Add GROUP BY clause to the logical plan
  result = builder
  var col_list: seq[string] = @[]
  for col_expr in columns:
    col_list.add(col_expr.toSql)
  result.current_plan = builder.current_plan & " GROUP BY " & col_list.join(", ")

proc orderBy*(builder: LogicalPlanBuilder, columns: varargs[LogicalExpr]): LogicalPlanBuilder =
  ## Add ORDER BY clause to the logical plan
  result = builder
  var col_list: seq[string] = @[]
  for col_expr in columns:
    col_list.add(col_expr.toSql)
  result.current_plan = builder.current_plan & " ORDER BY " & col_list.join(", ")

proc limit*(builder: LogicalPlanBuilder, count: int): LogicalPlanBuilder =
  ## Add LIMIT clause to the logical plan
  result = builder
  result.current_plan = builder.current_plan & " LIMIT " & $count

proc build*(builder: LogicalPlanBuilder): LogicalPlan =
  ## Build the final logical plan
  result = LogicalPlan(
    context: builder.context,
    query: builder.current_plan
  )

proc execute*(plan: LogicalPlan): DataFrame =
  ## Execute the logical plan and return a DataFrame
  result = plan.context.sql(plan.query)

# Aggregate functions
proc count*(expr: LogicalExpr): LogicalExpr =
  ## COUNT aggregate function
  result = LogicalExpr(
    expr_type: Function,
    column_name: "COUNT(" & expr.toSql & ")"
  )

proc sum*(expr: LogicalExpr): LogicalExpr =
  ## SUM aggregate function
  result = LogicalExpr(
    expr_type: Function,
    column_name: "SUM(" & expr.toSql & ")"
  )

proc avg*(expr: LogicalExpr): LogicalExpr =
  ## AVG aggregate function
  result = LogicalExpr(
    expr_type: Function,
    column_name: "AVG(" & expr.toSql & ")"
  )

proc min*(expr: LogicalExpr): LogicalExpr =
  ## MIN aggregate function
  result = LogicalExpr(
    expr_type: Function,
    column_name: "MIN(" & expr.toSql & ")"
  )

proc max*(expr: LogicalExpr): LogicalExpr =
  ## MAX aggregate function
  result = LogicalExpr(
    expr_type: Function,
    column_name: "MAX(" & expr.toSql & ")"
  )

# Example usage:
when isMainModule:
  let ctx = createSessionContext()

  # Register a table
  ctx.registerCSV("employees", "employees.csv")

  # Build a logical plan similar to datafusion-python:
  # SELECT name, department, AVG(salary)
  # FROM employees
  # WHERE salary > 50000
  # GROUP BY department
  # ORDER BY AVG(salary) DESC
  # LIMIT 10

  let plan = ctx.scan("employees")
    .select(col("name"), col("department"), avg(col("salary")))
    .filter(col("salary") > lit(50000))
    .groupBy(col("department"))
    .orderBy(avg(col("salary")))
    .limit(10)
    .build()

  # Execute the plan
  let result = plan.execute()
  result.show()