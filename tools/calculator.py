import ast
import operator

from langchain.tools import tool

from tools.results import tool_result


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> int | float:
    """Evaluate basic arithmetic without eval, names, attributes, or calls."""
    if len(expression) > 120:
        raise ValueError("Expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Enter a basic arithmetic expression.") from error

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if abs(result) > 1e100:
                raise ValueError("Result is too large.")
            return result
        raise ValueError("Only numbers and basic arithmetic operators are allowed.")

    try:
        return evaluate(tree)
    except (ArithmeticError, OverflowError) as error:
        raise ValueError(f"Could not calculate expression: {error}") from error


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        result = calculate(expression)
    except ValueError as error:
        return tool_result("calculator", expression, f"Error: {error}")
    return tool_result("calculator", expression, f"Result: {result}")