#!/usr/bin/env python3
"""Графическое решение задач 2.1 и 2.2 для варианта 26."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# В среде без домашнего каталога matplotlib нужен доступный для записи кеш.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-lw2")
)

import matplotlib.pyplot as plt
import numpy as np


def bisection(function, left: float, right: float, tolerance: float = 1e-12) -> float:
    """Находит корень на отрезке, где функция меняет знак."""
    f_left = function(left)
    f_right = function(right)
    if f_left * f_right > 0:
        raise ValueError("На концах отрезка функция должна иметь разные знаки")

    while right - left > tolerance:
        middle = (left + right) / 2
        f_middle = function(middle)
        if f_left * f_middle <= 0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2


def plot_task_2_1(output_dir: Path) -> Path:
    """Строит y = lg(x + 1) + 0.5 и y = x для задачи 2.1."""
    equation = lambda x: np.log10(x + 1) - x + 0.5
    roots = [
        bisection(equation, -0.999, -0.5),
        bisection(equation, 0.5, 1.0),
    ]

    x = np.linspace(-0.999, 1.6, 1400)
    logarithm = np.log10(x + 1) + 0.5

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, logarithm, linewidth=2, label=r"$y=\lg(x+1)+0.5$")
    ax.plot(x, x, linewidth=2, label=r"$y=x$")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    for index, root in enumerate(roots):
        ax.scatter(root, root, s=65, color="crimson", zorder=5)
        offset = (10, -24) if index == 0 else (10, 10)
        ax.annotate(
            rf"$x\approx {root:.6f}$",
            (root, root),
            xytext=offset,
            textcoords="offset points",
            color="crimson",
        )

    # В task_2_1.cpp метод начинает работу из x0 = 1.
    ax.scatter(1.0, 1.0, marker="x", s=85, color="darkorange", label=r"$x_0=1$")
    ax.set(xlim=(-1.05, 1.6), ylim=(-1.3, 1.65), xlabel="$x$", ylabel="$y$")
    ax.set_title(r"Задача 2.1: $\lg(x+1)-x+0.5=0$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = output_dir / "task_2_1_graph.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_task_2_2(output_dir: Path) -> Path:
    """Строит линии уровня F1=0 и F2=0 для системы задачи 2.2."""
    # Из первого уравнения: y = (1 +- sqrt(9 - 8*x^2)) / 2.
    x_limit = np.sqrt(9 / 8)
    x_first = np.linspace(-x_limit, x_limit, 1200)
    discriminant = np.maximum(0.0, 9 - 8 * x_first**2)
    y_upper = (1 + np.sqrt(discriminant)) / 2
    y_lower = (1 - np.sqrt(discriminant)) / 2

    # Из второго уравнения: y = (x + 1)^2 - 2, причем x >= -1.
    x_second = np.linspace(-1.0, 1.25, 1200)
    y_second = (x_second + 1) ** 2 - 2

    # После подстановки получаем x^4 + 4*x^3 + 3*x^2 - 6*x = 0.
    polynomial_roots = np.roots([1.0, 4.0, 3.0, -6.0, 0.0])
    real_x = sorted(
        root.real
        for root in polynomial_roots
        if abs(root.imag) < 1e-10 and root.real >= -1.0
    )
    solutions = [(x, (x + 1) ** 2 - 2) for x in real_x]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        x_first,
        y_upper,
        color="royalblue",
        linewidth=2,
        label=r"$2x^2-y+y^2-2=0$",
    )
    ax.plot(x_first, y_lower, color="royalblue", linewidth=2)
    ax.plot(
        x_second,
        y_second,
        color="seagreen",
        linewidth=2,
        label=r"$x-\sqrt{y+2}+1=0$",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    for index, (x_solution, y_solution) in enumerate(solutions):
        ax.scatter(x_solution, y_solution, s=70, color="crimson", zorder=5)
        offset = (10, -25) if index == 0 else (10, 10)
        ax.annotate(
            rf"$({x_solution:.6f};\ {y_solution:.6f})$",
            (x_solution, y_solution),
            xytext=offset,
            textcoords="offset points",
            color="crimson",
        )

    ax.scatter(
        0.8,
        1.4,
        marker="x",
        s=90,
        color="darkorange",
        label=r"$(x_0,y_0)=(0.8,1.4)$",
    )
    ax.set(xlim=(-1.15, 1.25), ylim=(-2.15, 2.35), xlabel="$x$", ylabel="$y$")
    ax.set_title("Задача 2.2: графическое решение системы, $a=2$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = output_dir / "task_2_2_graph.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "graphs",
        help="каталог для PNG-файлов (по умолчанию: ./graphs)",
    )
    parser.add_argument(
        "--show", action="store_true", help="также показать графики в окнах"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [plot_task_2_1(args.output_dir), plot_task_2_2(args.output_dir)]
    for path in paths:
        print(f"Сохранен график: {path}")

    if args.show:
        # Повторное построение не требуется: показываем сохраненные изображения.
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for axis, path in zip(axes, paths):
            axis.imshow(plt.imread(path))
            axis.axis("off")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
