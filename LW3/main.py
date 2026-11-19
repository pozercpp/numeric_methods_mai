from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence


EPS = 1e-12


def solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Решение СЛАУ методом Гаусса с выбором главного элемента."""
    a = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < EPS:
            raise ValueError("Вырожденная система")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [value / divisor for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[row][-1] for row in range(n)]


def poly_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    n = max(len(left), len(right))
    result = [0.0] * n
    for i, value in enumerate(left):
        result[i] += value
    for i, value in enumerate(right):
        result[i] += value
    return result


def poly_mul(left: Sequence[float], right: Sequence[float]) -> list[float]:
    result = [0.0] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[i + j] += x * y
    return result


def poly_eval(coefficients: Sequence[float], x: float) -> float:
    """Значение полинома; коэффициенты расположены от x^0 к старшим степеням."""
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def format_polynomial(coefficients: Sequence[float], variable: str = "x") -> str:
    terms: list[str] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if abs(coefficient) < 5e-11:
            continue
        magnitude = abs(coefficient)
        if power == 0:
            body = f"{magnitude:.10g}"
        elif power == 1:
            body = f"{magnitude:.10g}*{variable}"
        else:
            body = f"{magnitude:.10g}*{variable}^{power}"
        if not terms:
            terms.append(("-" if coefficient < 0 else "") + body)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + body)
    return "".join(terms) or "0"


def lagrange_polynomial(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    result = [0.0]
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        basis = [1.0]
        denominator = 1.0
        for j, xj in enumerate(xs):
            if i != j:
                basis = poly_mul(basis, [-xj, 1.0])
                denominator *= xi - xj
        result = poly_add(result, [value * yi / denominator for value in basis])
    return result


def divided_differences(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    column = list(map(float, ys))
    coefficients = [column[0]]
    for order in range(1, len(xs)):
        column = [
            (column[i + 1] - column[i]) / (xs[i + order] - xs[i])
            for i in range(len(column) - 1)
        ]
        coefficients.append(column[0])
    return coefficients


def newton_to_standard(xs: Sequence[float], coefficients: Sequence[float]) -> list[float]:
    result = [coefficients[0]]
    factor = [1.0]
    for i in range(1, len(coefficients)):
        factor = poly_mul(factor, [-xs[i - 1], 1.0])
        result = poly_add(result, [coefficients[i] * value for value in factor])
    return result


def task_3_1() -> None:
    print("\n3.1. Интерполяция Лагранжа и Ньютона")
    function = lambda x: 1.0 / x**2 + x**2
    x_star = 0.8
    node_sets = {
        "а": [0.1, 0.5, 0.9, 1.3],
        "б": [0.1, 0.5, 1.1, 1.3],
    }
    for label, xs in node_sets.items():
        ys = [function(x) for x in xs]
        lagrange = lagrange_polynomial(xs, ys)
        newton_coefficients = divided_differences(xs, ys)
        newton = newton_to_standard(xs, newton_coefficients)
        approximation = poly_eval(lagrange, x_star)
        error = abs(approximation - function(x_star))
        print(f"  {label}) x = {xs}")
        print(f"     y = {[round(value, 10) for value in ys]}")
        print(f"     L3(x) = {format_polynomial(lagrange)}")
        print(f"     коэффициенты разделённых разностей = "
              f"{[round(value, 10) for value in newton_coefficients]}")
        print(f"     P3(x) = {format_polynomial(newton)}")
        print(f"     P3({x_star}) = {approximation:.10f}, "
              f"f({x_star}) = {function(x_star):.10f}, |ошибка| = {error:.10f}")


def natural_cubic_spline(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float, float, float]]:
    """Коэффициенты a,b,c,d для S_i=a+b*t+c*t^2+d*t^3, t=x-x_i."""
    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]
    matrix = [[0.0] * (n - 1) for _ in range(n - 1)]
    rhs = [0.0] * (n - 1)
    for row, i in enumerate(range(1, n)):
        matrix[row][row] = 2.0 * (h[i - 1] + h[i])
        if row > 0:
            matrix[row][row - 1] = h[i - 1]
        if row + 1 < n - 1:
            matrix[row][row + 1] = h[i]
        rhs[row] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    c = [0.0] + solve_linear_system(matrix, rhs) + [0.0]
    pieces = []
    for i in range(n):
        a = ys[i]
        b = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d = (c[i + 1] - c[i]) / (3.0 * h[i])
        pieces.append((a, b, c[i], d))
    return pieces


def spline_eval(xs: Sequence[float], pieces: Sequence[Sequence[float]], x: float) -> float:
    if not xs[0] <= x <= xs[-1]:
        raise ValueError("Точка находится вне интервала сплайна")
    interval = min(len(pieces) - 1, next((i for i in range(len(pieces)) if x <= xs[i + 1]), len(pieces) - 1))
    t = x - xs[interval]
    return poly_eval(pieces[interval], t)


def task_3_2() -> None:
    print("\n3.2. Естественный кубический сплайн")
    xs = [0.1, 0.5, 0.9, 1.3, 1.7]
    ys = [100.01, 4.2500, 2.0446, 2.2817, 3.2360]
    pieces = natural_cubic_spline(xs, ys)
    for i, coefficients in enumerate(pieces):
        print(f"  [{xs[i]:.1f}, {xs[i + 1]:.1f}]: "
              f"S{i}(x) = {format_polynomial(coefficients, f'(x-{xs[i]:.1f})')}")
    print(f"  S(0.8) = {spline_eval(xs, pieces, 0.8):.10f}")


def normal_equations(xs: Sequence[float], ys: Sequence[float], degree: int) -> tuple[list[list[float]], list[float]]:
    matrix = [
        [sum(x ** (row + col) for x in xs) for col in range(degree + 1)]
        for row in range(degree + 1)
    ]
    rhs = [sum(y * x**power for x, y in zip(xs, ys)) for power in range(degree + 1)]
    return matrix, rhs


def save_least_squares_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    polynomials: Sequence[Sequence[float]],
    function: Callable[[float], float],
    path: Path,
) -> bool:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/lw3-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    left, right = min(xs), max(xs)
    grid = [left + (right - left) * i / 400 for i in range(401)]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(xs, ys, color="black", label="Табличные значения", zorder=3)
    axis.plot(grid, [function(x) for x in grid], "--", label="f(x) = 1/x² + x²")
    for degree, polynomial in enumerate(polynomials, start=1):
        axis.plot(grid, [poly_eval(polynomial, x) for x in grid], label=f"МНК, степень {degree}")
    axis.set(xlabel="x", ylabel="y", title="Задание 3.3, вариант 26")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def task_3_3(plot_path: Path | None) -> None:
    print("\n3.3. Метод наименьших квадратов")
    xs = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    ys = [100.01, 4.250, 2.0446, 2.2817, 3.236, 4.6368]
    function = lambda x: 1.0 / x**2 + x**2
    polynomials = []
    for degree in (1, 2):
        matrix, rhs = normal_equations(xs, ys, degree)
        polynomial = solve_linear_system(matrix, rhs)
        polynomials.append(polynomial)
        sse = sum((poly_eval(polynomial, x) - y) ** 2 for x, y in zip(xs, ys))
        print(f"  Степень {degree}:")
        print(f"    нормальная матрица = {[[round(v, 9) for v in row] for row in matrix]}")
        print(f"    правая часть = {[round(value, 9) for value in rhs]}")
        print(f"    P{degree}(x) = {format_polynomial(polynomial)}")
        print(f"    сумма квадратов ошибок = {sse:.10f}")
    if plot_path is not None:
        if save_least_squares_plot(xs, ys, polynomials, function, plot_path):
            print(f"  График сохранён: {plot_path}")
        else:
            print("  График не построен: matplotlib не установлен")


def task_3_4() -> None:
    print("\n3.4. Численное дифференцирование")
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.0, 0.86603, 1.0, 0.0, -2.0]
    x_star = 2.0
    center = xs.index(x_star)
    h = xs[center] - xs[center - 1]
    first = (ys[center + 1] - ys[center - 1]) / (2.0 * h)
    second = (ys[center + 1] - 2.0 * ys[center] + ys[center - 1]) / h**2
    local_polynomial = lagrange_polynomial(xs[center - 1:center + 2], ys[center - 1:center + 2])
    print(f"  Локальный интерполяционный полином: P2(x) = {format_polynomial(local_polynomial)}")
    print(f"  f'({x_star:g})  = {first:.10f}")
    print(f"  f''({x_star:g}) = {second:.10f}")


def composite_midpoint(function: Callable[[float], float], start: float, end: float, h: float) -> float:
    n = round((end - start) / h)
    return h * sum(function(start + (i + 0.5) * h) for i in range(n))


def composite_trapezoid(function: Callable[[float], float], start: float, end: float, h: float) -> float:
    n = round((end - start) / h)
    return h * (0.5 * function(start) + sum(function(start + i * h) for i in range(1, n)) + 0.5 * function(end))


def composite_simpson(function: Callable[[float], float], start: float, end: float, h: float) -> float:
    n = round((end - start) / h)
    if n % 2:
        raise ValueError("Для метода Симпсона число интервалов должно быть чётным")
    odd = sum(function(start + i * h) for i in range(1, n, 2))
    even = sum(function(start + i * h) for i in range(2, n, 2))
    return h / 3.0 * (function(start) + function(end) + 4.0 * odd + 2.0 * even)


def runge_romberg(coarse: float, fine: float, order: int, ratio: float = 2.0) -> tuple[float, float]:
    correction = (fine - coarse) / (ratio**order - 1.0)
    return fine + correction, abs(correction)


def task_3_5() -> None:
    print("\n3.5. Численное интегрирование")
    function = lambda x: x**2 * math.sqrt(36.0 - x**2)
    start, end, h1, h2 = 1.0, 5.0, 1.0, 0.5
    methods = [
        ("средних прямоугольников", composite_midpoint, 2),
        ("трапеций", composite_trapezoid, 2),
        ("Симпсона", composite_simpson, 4),
    ]
    for name, method, order in methods:
        coarse = method(function, start, end, h1)
        fine = method(function, start, end, h2)
        refined, estimate = runge_romberg(coarse, fine, order)
        print(f"  Метод {name}:")
        print(f"    F(h={h1:g}) = {coarse:.10f}")
        print(f"    F(h={h2:g}) = {fine:.10f}")
        print(f"    оценка Рунге--Ромберга = {estimate:.10f}")
        print(f"    уточнённое значение = {refined:.10f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ЛР 3, вариант 26")
    parser.add_argument(
        "--task", choices=["all", "3.1", "3.2", "3.3", "3.4", "3.5"], default="all",
        help="выполнить одно задание (по умолчанию все)",
    )
    parser.add_argument("--no-plot", action="store_true", help="не сохранять график задания 3.3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {
        "3.1": task_3_1,
        "3.2": task_3_2,
        "3.3": lambda: task_3_3(None if args.no_plot else Path("task_3_3.png")),
        "3.4": task_3_4,
        "3.5": task_3_5,
    }
    selected: Iterable[str] = tasks if args.task == "all" else [args.task]
    print("Лабораторная работа №3, вариант 26")
    for name in selected:
        tasks[name]()


if __name__ == "__main__":
    main()
