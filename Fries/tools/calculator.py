
import ast
import operator
import math
import cmath
import re
import sympy as sp

'''
For now there are some basic set of operations which can be performed by the Model this
calculator inclued all the operations given below
The calculator has now been expanded to support:

- Basic arithmetic
- Powers and roots
- Fractions
- Algebra
- Linear equations
- Quadratic equations
- Polynomial equations
- Systems of equations
- Factorization
- Expansion
- Simplification
- Trigonometry
- Inverse trigonometry
- Logarithms
- Exponentials
- Limits
- Differentiation
- Integration
- Summation
- Products
- Complex numbers
- Complex number operations
- Matrices
- Determinants
- Matrix inverse
- Matrix multiplication
- Vectors
- Geometry
- Coordinate geometry
- Distance
- Midpoint
- Slope
- Circle calculations
- Triangle calculations
- Rectangle calculations
- Square calculations
- Statistics
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Combinations
- Permutations
- Factorials
- GCD / LCM
- Prime checking
- Numerical evaluation
'''
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
x, y, z, t, a, b, c, r = sp.symbols(
    "x y z t a b c r"
)

i = sp.I
MATH_NAMESPACE = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "sqrt": sp.sqrt,
    "cbrt": sp.real_root,
    "abs": sp.Abs,
    "pow": sp.Pow,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda value: sp.log(value, 10),
    "log2": lambda value: sp.log(value, 2),
    "exp": sp.exp,
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    "cancel": sp.cancel,
    "apart": sp.apart,
    "diff": sp.diff,
    "derivative": sp.diff,
    "integrate": sp.integrate,
    "integral": sp.integrate,
    "gamma": sp.gamma,
    "factorial": sp.factorial,
    "comb": sp.binomial,
    "gcd": sp.gcd,
    "lcm": sp.ilcm,
    "isprime": sp.isprime,
    "x": x,
    "y": y,
    "z": z,
    "t": t,
    "a": a,
    "b": b,
    "c": c,
    "r": r,
}

def basic_calculate(expression):

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(
                "FlexAI: Only numbers are allowed."
            )
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            operation = OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError(
                    "FlexAI: Operation not supported"
                )
            return operation(left, right)
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError(
                "FlexAI: Unary operator not supported."
            )
        raise ValueError(
            "FlexAI: Invalid expression"
        )
    return evaluate(tree)
def parse_math(expression):
    expression = expression.strip()
    # Convert common mathematical notation
    expression = expression.replace("^", "**")
    # Convert × and ÷
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    # Convert common square-root notation
    expression = expression.replace("√", "sqrt")
    # Replace i with I for complex numbers
    expression = re.sub(r'(?<![A-Za-z])i(?![A-Za-z])', 'I', expression)
    namespace = dict(MATH_NAMESPACE)
    namespace["I"] = sp.I
    return sp.sympify(
        expression,
        locals=namespace
    )
def solve_equation(expression):
    expression = expression.replace("^", "**")
    if "=" in expression:
        left, right = expression.split("=", 1)
        equation = sp.Eq(
            parse_math(left),
            parse_math(right)
        )
    else:
        equation = parse_math(expression)
    variables = equation.free_symbols
    if not variables:
        return sp.simplify(equation)
    return sp.solve(
        equation,
        list(variables)
    )
def solve_system(equations):
    parsed = []
    for equation in equations:
        left, right = equation.split("=")
        parsed.append(
            sp.Eq(
                parse_math(left),
                parse_math(right)
            )
        )

    variables = set()
    for equation in parsed:
        variables.update(equation.free_symbols)
    return sp.solve(
        parsed,
        list(variables)
    )
def differentiate(expression, variable="x", order=1):
    expr = parse_math(expression)
    symbol = sp.Symbol(variable)
    return sp.diff(
        expr,
        symbol,
        order
    )
def integrate(expression, variable="x"):
    expr = parse_math(expression)
    symbol = sp.Symbol(variable)
    return sp.integrate(
        expr,
        symbol
    )
def calculate_limit(
    expression,
    variable="x",
    point=0
    ):
    expr = parse_math(expression)
    symbol = sp.Symbol(variable)
    return sp.limit(
        expr,
        symbol,
        point
    )
def factor_expression(expression):
    expr = parse_math(expression)
    return sp.factor(expr)
def expand_expression(expression):
    expr = parse_math(expression)
    return sp.expand(expr)
def simplify_expression(expression):
    expr = parse_math(expression)
    return sp.simplify(expr)
def numerical_value(expression, digits=10):
    expr = parse_math(expression)
    return sp.N(
        expr,
        digits
    )
def complex_calculate(expression):
    expr = parse_math(expression)
    return sp.expand_complex(expr)
def complex_real(expression):
    expr = parse_math(expression)
    return sp.re(expr)
def complex_imaginary(expression):
    expr = parse_math(expression)
    return sp.im(expr)
def complex_conjugate(expression):
    expr = parse_math(expression)
    return sp.conjugate(expr)
def complex_magnitude(expression):
    expr = parse_math(expression)
    return sp.Abs(expr)
def complex_argument(expression):
    expr = parse_math(expression)
    return sp.arg(expr)
def matrix_create(values):
    return sp.Matrix(values)
def matrix_determinant(values):
    matrix = sp.Matrix(values)
    return matrix.det()
def matrix_inverse(values):
    matrix = sp.Matrix(values)
    return matrix.inv()
def matrix_transpose(values):
    matrix = sp.Matrix(values)
    return matrix.T
def matrix_rank(values):
    matrix = sp.Matrix(values)
    return matrix.rank()
def matrix_eigenvalues(values):
    matrix = sp.Matrix(values)
    return matrix.eigenvals()
def matrix_multiply(a_matrix, b_matrix):
    A = sp.Matrix(a_matrix)
    B = sp.Matrix(b_matrix)
    return A * B
def vector_dot(a_vector, b_vector):
    A = sp.Matrix(a_vector)
    B = sp.Matrix(b_vector)
    return A.dot(B)
def vector_cross(a_vector, b_vector):
    A = sp.Matrix(a_vector)
    B = sp.Matrix(b_vector)
    return A.cross(B)
def vector_magnitude(vector):
    V = sp.Matrix(vector)
    return sp.sqrt(
        V.dot(V)
    )
def circle_area(radius):
    return sp.pi * radius ** 2
def circle_circumference(radius):
    return 2 * sp.pi * radius
def sphere_volume(radius):
    return sp.Rational(4, 3) * sp.pi * radius ** 3
def sphere_surface_area(radius):
    return 4 * sp.pi * radius ** 2
def cylinder_volume(radius, height):
    return sp.pi * radius ** 2 * height
def cylinder_surface_area(radius, height):
    return 2 * sp.pi * radius * (
        radius + height
    )
def cone_volume(radius, height):
    return sp.Rational(1, 3) * sp.pi * radius ** 2 * height
def cone_surface_area(radius, slant_height):
    return sp.pi * radius * (
        radius + slant_height
    )
def rectangle_area(length, width):
    return length * width
def rectangle_perimeter(length, width):
    return 2 * (
        length + width
    )
def square_area(side):
    return side ** 2
def square_perimeter(side):
    return 4 * side
def triangle_area(base, height):
    return sp.Rational(1, 2) * base * height
def triangle_area_heron(a, b, c):
    s = (a + b + c) / 2
    return sp.sqrt(
        s * (s - a) * (s - b) * (s - c)
    )
def distance_2d(x1, y1, x2, y2):
    return sp.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )
def distance_3d(
    x1, y1, z1,
    x2, y2, z2
    ):
    return sp.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2 +
        (z2 - z1) ** 2
    )
def midpoint_2d(x1, y1, x2, y2):
    return (
        sp.Rational(x1 + x2, 2),
        sp.Rational(y1 + y2, 2)
    )
def slope(x1, y1, x2, y2):
    if x2 == x1:
        raise ValueError(
            "FlexAI: Vertical line has undefined slope."
        )
    return sp.Rational(
        y2 - y1,
        x2 - x1
    )
def mean(values):
    return sp.Rational(
        sum(values),
        len(values)
    )
def median(values):
    return sp.median(values)
def mode(values):
    return sp.modes(values)
def variance(values):
    return sp.variance(values)
def standard_deviation(values):
    return sp.std(values)
def permutation(n, r):
    return sp.factorial(n) / sp.factorial(n - r)
def combination(n, r):
    return sp.binomial(n, r)
def calculate(expression):
    expression = expression.strip()

    if expression.startswith("system:"):
        equations = expression[
            len("system:"):
        ].split(";")
        return solve_system(equations)

    if "=" in expression:

        return solve_equation(expression)
    try:
        result = parse_math(expression)
        return sp.simplify(result)
    except Exception:
        pass
    try:
        return basic_calculate(expression)
    except Exception as error:

        raise ValueError(
            f"FlexAI: Could not understand mathematical expression: "
            f"{expression}"
        ) from error

MATH_TOOLS = {
    "calculate": calculate,
    "solve_equation": solve_equation,
    "solve_system": solve_system,
    "differentiate": differentiate,
    "integrate": integrate,
    "limit": calculate_limit,
    "factor": factor_expression,
    "expand": expand_expression,
    "simplify": simplify_expression,
    "numerical_value": numerical_value,

    # Complex numbers
    "complex_calculate": complex_calculate,
    "complex_real": complex_real,
    "complex_imaginary": complex_imaginary,
    "complex_conjugate": complex_conjugate,
    "complex_magnitude": complex_magnitude,
    "complex_argument": complex_argument,

    # Matrices
    "matrix_create": matrix_create,
    "matrix_determinant": matrix_determinant,
    "matrix_inverse": matrix_inverse,
    "matrix_transpose": matrix_transpose,
    "matrix_rank": matrix_rank,
    "matrix_eigenvalues": matrix_eigenvalues,
    "matrix_multiply": matrix_multiply,

    # Vectors
    "vector_dot": vector_dot,
    "vector_cross": vector_cross,
    "vector_magnitude": vector_magnitude,

    # Geometry
    "circle_area": circle_area,
    "circle_circumference": circle_circumference,
    "sphere_volume": sphere_volume,
    "sphere_surface_area": sphere_surface_area,
    "cylinder_volume": cylinder_volume,
    "cylinder_surface_area": cylinder_surface_area,
    "cone_volume": cone_volume,
    "cone_surface_area": cone_surface_area,
    "rectangle_area": rectangle_area,
    "rectangle_perimeter": rectangle_perimeter,
    "square_area": square_area,
    "square_perimeter": square_perimeter,
    "triangle_area": triangle_area,
    "triangle_area_heron": triangle_area_heron,

    # Coordinate geometry
    "distance_2d": distance_2d,
    "distance_3d": distance_3d,
    "midpoint_2d": midpoint_2d,
    "slope": slope,

    # Statistics
    "mean": mean,
    "median": median,
    "mode": mode,
    "variance": variance,
    "standard_deviation": standard_deviation,

    # Combinatorics
    "permutation": permutation,
    "combination": combination,
}

if __name__ == "__main__":
    print("========== TRIGONOMETRY ==========")
    print(calculate("sin(pi/2)"))
    print(calculate("log(100, 10)"))
    print("\n========== ALGEBRA ==========")
    print(factor_expression("x^2 - 9"))
    print(expand_expression("(x + 2)^3"))
    print(simplify_expression("(x^2 - 1)/(x - 1)"))
    print("\n========== CALCULUS ==========")
    print(differentiate("x^3 + 2*x", "x"))
    print(integrate("x^2", "x"))
    print("\n========== COMPLEX NUMBERS ==========")
    print(complex_calculate("3 + 4*i"))
    print(complex_magnitude("3 + 4*i"))
    print("\n========== GEOMETRY ==========")
    print(circle_area(10))
    print(triangle_area_heron(3, 4, 5))
    print(distance_2d(0, 0, 3, 4))
    print("\n========== MATRICES ==========")
    print(matrix_determinant([
        [1, 2],
        [3, 4]
    ]))
    print("\n========== STATISTICS ==========")
    print(mean([10, 20, 30, 40]))
    print("\n========== COMBINATORICS ==========")
    print(combination(10, 3))