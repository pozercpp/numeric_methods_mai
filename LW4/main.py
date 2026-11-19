from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence


State = tuple[float, float]
SystemFunction = Callable[[float, State], State]


def make_grid(left: float, right: float, step: float) -> list[float]:
    """Create a uniform grid and reject a step that does not divide the interval."""
    intervals = round((right - left) / step)
    if intervals <= 0 or not math.isclose(
        left + intervals * step, right, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("The step must divide the integration interval")
    return [left + i * step for i in range(intervals + 1)]


def add_scaled(state: State, increment: State, scale: float) -> State:
    return state[0] + scale * increment[0], state[1] + scale * increment[1]


def rk4_step(function: SystemFunction, x: float, state: State, step: float) -> State:
    k1 = function(x, state)
    k2 = function(x + step / 2, add_scaled(state, k1, step / 2))
    k3 = function(x + step / 2, add_scaled(state, k2, step / 2))
    k4 = function(x + step, add_scaled(state, k3, step))
    return (
        state[0] + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
        state[1] + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
    )


def euler_method(
    function: SystemFunction,
    left: float,
    right: float,
    initial: State,
    step: float,
) -> tuple[list[float], list[State]]:
    xs = make_grid(left, right, step)
    values = [initial]
    for x in xs[:-1]:
        values.append(add_scaled(values[-1], function(x, values[-1]), step))
    return xs, values


def runge_kutta_method(
    function: SystemFunction,
    left: float,
    right: float,
    initial: State,
    step: float,
) -> tuple[list[float], list[State]]:
    xs = make_grid(left, right, step)
    values = [initial]
    for x in xs[:-1]:
        values.append(rk4_step(function, x, values[-1], step))
    return xs, values


def adams_method(
    function: SystemFunction,
    left: float,
    right: float,
    initial: State,
    step: float,
) -> tuple[list[float], list[State]]:
    """Explicit four-step Adams-Bashforth method, started with RK4."""
    xs = make_grid(left, right, step)
    if len(xs) < 4:
        raise ValueError("The Adams method needs at least three grid intervals")

    values = [initial]
    for i in range(min(3, len(xs) - 1)):
        values.append(rk4_step(function, xs[i], values[i], step))

    derivatives = [function(x, value) for x, value in zip(xs[:4], values)]
    for i in range(3, len(xs) - 1):
        next_value = tuple(
            values[i][component]
            + step
            * (
                55 * derivatives[i][component]
                - 59 * derivatives[i - 1][component]
                + 37 * derivatives[i - 2][component]
                - 9 * derivatives[i - 3][component]
            )
            / 24
            for component in range(2)
        )
        values.append(next_value)  # type: ignore[arg-type]
        derivatives.append(function(xs[i + 1], values[-1]))
    return xs, values


def runge_romberg(
    coarse: Sequence[float], fine: Sequence[float], order: int
) -> tuple[list[float], list[float]]:
    """Extrapolated values and error estimates at the nodes of the coarse grid."""
    if len(fine) != 2 * len(coarse) - 1:
        raise ValueError("The fine grid must use half the coarse step")
    denominator = 2**order - 1
    estimates = [(fine[2 * i] - coarse[i]) / denominator for i in range(len(coarse))]
    improved = [fine[2 * i] + estimates[i] for i in range(len(coarse))]
    return improved, [abs(value) for value in estimates]


def cauchy_system(x: float, state: State) -> State:
    y, derivative = state
    return derivative, -2 * derivative / x - y / x**4


def cauchy_exact(x: float) -> float:
    return ((math.sin(1) + math.cos(1)) * math.cos(1 / x)
            + (math.sin(1) - math.cos(1)) * math.sin(1 / x))


def print_cauchy_method(
    name: str,
    solver: Callable[..., tuple[list[float], list[State]]],
    order: int,
    step: float,
) -> None:
    xs, coarse_states = solver(cauchy_system, 1.0, 2.0, (1.0, 1.0), step)
    _, fine_states = solver(cauchy_system, 1.0, 2.0, (1.0, 1.0), step / 2)
    coarse = [state[0] for state in coarse_states]
    fine = [state[0] for state in fine_states]
    improved, estimates = runge_romberg(coarse, fine, order)

    print(f"\n{name} (order {order})")
    print("      x          y_h        y_h/2         RR y       exact      |err_h|       RR est")
    for i, x in enumerate(xs):
        exact = cauchy_exact(x)
        print(
            f"  {x:5.2f}  {coarse[i]:11.7f}  {fine[2*i]:11.7f}  "
            f"{improved[i]:11.7f}  {exact:11.7f}  "
            f"{abs(coarse[i] - exact):11.3e}  {estimates[i]:11.3e}"
        )
    print(
        f"  max |y_h-y| = {max(abs(y - cauchy_exact(x)) for x, y in zip(xs, coarse)):.6e}; "
        f"max RR estimate = {max(estimates):.6e}"
    )


def task_4_1(step: float = 0.1) -> None:
    print("4.1. Cauchy problem, variant 26")
    print("x^4*y'' + 2*x^3*y' + y = 0, y(1)=1, y'(1)=1, x in [1,2]")
    print_cauchy_method("Euler method", euler_method, 1, step)
    print_cauchy_method("Runge-Kutta method", runge_kutta_method, 4, step)
    print_cauchy_method("Adams-Bashforth method", adams_method, 4, step)


def boundary_system(x: float, state: State) -> State:
    y, derivative = state
    second_derivative = (x + 1 / x - (x + 2) * derivative + y) / (x * (x + 1))
    return derivative, second_derivative


def boundary_exact(x: float) -> float:
    return x + 3.5 + 1 / x + (x / 2 + 1) * math.log(abs(x))


def shooting_method(
    step: float,
    first_guess: float = 5.0,
    second_guess: float = 7.0,
    tolerance: float = 1e-12,
    max_iterations: int = 30,
) -> tuple[list[float], list[float], int, float]:
    """Shooting with y(1) as the unknown parameter and the secant method."""
    target = 13 + 4 * math.log(2)

    def trajectory(initial_y: float) -> tuple[list[float], list[State], float]:
        xs, states = runge_kutta_method(
            boundary_system, 1.0, 2.0, (initial_y, 1.5), step
        )
        residual = states[-1][0] + 4 * states[-1][1] - target
        return xs, states, residual

    previous_guess, current_guess = first_guess, second_guess
    _, _, previous_residual = trajectory(previous_guess)
    xs, states, current_residual = trajectory(current_guess)
    iterations = 0
    while abs(current_residual) > tolerance and iterations < max_iterations:
        difference = current_residual - previous_residual
        if abs(difference) < 1e-15:
            raise ArithmeticError("The secant method encountered equal residuals")
        next_guess = current_guess - current_residual * (
            current_guess - previous_guess
        ) / difference
        previous_guess, previous_residual = current_guess, current_residual
        current_guess = next_guess
        xs, states, current_residual = trajectory(current_guess)
        iterations += 1
    if abs(current_residual) > tolerance:
        raise ArithmeticError("The shooting method did not converge")
    return xs, [state[0] for state in states], iterations, current_residual


def solve_linear_system(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    augmented = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("Singular linear system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, size + 1):
            augmented[column][j] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for j in range(column, size + 1):
                augmented[row][j] -= factor * augmented[column][j]
    return [augmented[row][-1] for row in range(size)]


def finite_difference_method(step: float) -> tuple[list[float], list[float]]:
    """Second-order central differences and second-order one-sided Robin BCs."""
    xs = make_grid(1.0, 2.0, step)
    size = len(xs)
    if size < 3:
        raise ValueError("At least two intervals are required")
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size

    # (-3*y0 + 4*y1 - y2) / (2*h) = 3/2
    matrix[0][0], matrix[0][1], matrix[0][2] = -3, 4, -1
    rhs[0] = 3 * step

    for i in range(1, size - 1):
        x = xs[i]
        a, b, c = x * (x + 1), x + 2, -1.0
        matrix[i][i - 1] = a / step**2 - b / (2 * step)
        matrix[i][i] = -2 * a / step**2 + c
        matrix[i][i + 1] = a / step**2 + b / (2 * step)
        rhs[i] = x + 1 / x

    # 4*(3*yn - 4*y(n-1) + y(n-2))/(2*h) + yn = 13 + 4*ln(2)
    matrix[-1][-3] = 2 / step
    matrix[-1][-2] = -8 / step
    matrix[-1][-1] = 6 / step + 1
    rhs[-1] = 13 + 4 * math.log(2)
    return xs, solve_linear_system(matrix, rhs)


def print_boundary_method(
    name: str,
    coarse_xs: Sequence[float],
    coarse: Sequence[float],
    fine: Sequence[float],
    order: int,
) -> None:
    improved, estimates = runge_romberg(coarse, fine, order)
    print(f"\n{name} (order {order})")
    print("      x          y_h        y_h/2         RR y       exact      |err_h|       RR est")
    for i, x in enumerate(coarse_xs):
        exact = boundary_exact(x)
        print(
            f"  {x:5.2f}  {coarse[i]:11.7f}  {fine[2*i]:11.7f}  "
            f"{improved[i]:11.7f}  {exact:11.7f}  "
            f"{abs(coarse[i] - exact):11.3e}  {estimates[i]:11.3e}"
        )
    print(
        f"  max |y_h-y| = {max(abs(y - boundary_exact(x)) for x, y in zip(coarse_xs, coarse)):.6e}; "
        f"max RR estimate = {max(estimates):.6e}"
    )


def task_4_2(step: float = 0.1) -> None:
    print("4.2. Boundary-value problem, variant 26")
    print("x(x+1)y''+(x+2)y'-y=x+1/x, y'(1)=3/2,")
    print("4y'(2)+y(2)=13+4ln(2), x in [1,2]")

    coarse_xs, shooting_coarse, iterations, residual = shooting_method(step)
    _, shooting_fine, _, _ = shooting_method(step / 2)
    print(f"Shooting secant iterations: {iterations}; boundary residual: {residual:.3e}")
    print_boundary_method(
        "Shooting method (RK4 integration)",
        coarse_xs,
        shooting_coarse,
        shooting_fine,
        4,
    )

    finite_xs, finite_coarse = finite_difference_method(step)
    _, finite_fine = finite_difference_method(step / 2)
    print_boundary_method(
        "Finite-difference method", finite_xs, finite_coarse, finite_fine, 2
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laboratory work 4, variant 26")
    parser.add_argument(
        "--task", choices=("4.1", "4.2", "all"), default="all", help="task to run"
    )
    parser.add_argument("--step", type=float, default=0.1, help="coarse grid step")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.step <= 0:
        raise SystemExit("The step must be positive")
    if arguments.task in ("4.1", "all"):
        task_4_1(arguments.step)
    if arguments.task == "all":
        print()
    if arguments.task in ("4.2", "all"):
        task_4_2(arguments.step)


if __name__ == "__main__":
    main()
