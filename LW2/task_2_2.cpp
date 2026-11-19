#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kA = 2.0;
constexpr double kInitialX = 0.8; // Начальная точка выбрана по графику.
constexpr double kInitialY = 1.4;
constexpr int kMaxIterations = 10000;

struct Point {
    double x;
    double y;
};

struct Iteration {
    int number;
    Point point;
    double step;
    double residual;
};

Point systemValue(const Point& p) {
    return {
        kA * p.x * p.x - p.y + p.y * p.y - kA,
        p.x - std::sqrt(p.y + kA) + 1.0
    };
}

double norm(const Point& p) {
    return std::max(std::abs(p.x), std::abs(p.y));
}

Point difference(const Point& lhs, const Point& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}

// Эквивалентная форма системы для положительного решения:
// x = sqrt(y + a) - 1, y = sqrt(a - a*x^2 + y).
Point phi(const Point& p) {
    const double xRadicand = p.y + kA;
    const double yRadicand = kA - kA * p.x * p.x + p.y;
    if (xRadicand < 0.0 || yRadicand < 0.0) {
        throw std::domain_error("Итерация вышла из области определения");
    }
    return {
        std::sqrt(xRadicand) - 1.0,
        std::sqrt(yRadicand)
    };
}

std::vector<Iteration> simpleIteration(double eps) {
    std::vector<Iteration> history;
    Point current{kInitialX, kInitialY};

    for (int i = 1; i <= kMaxIterations; ++i) {
        const Point next = phi(current);
        const double step = norm(difference(next, current));
        const double residual = norm(systemValue(next));
        history.push_back({i, next, step, residual});
        current = next;

        if (step <= eps && residual <= eps) {
            return history;
        }
    }
    throw std::runtime_error("Метод простой итерации не сошелся");
}

Point newtonStep(const Point& p) {
    const Point value = systemValue(p);
    if (p.y + kA <= 0.0) {
        throw std::domain_error("Метод Ньютона вышел из области определения");
    }

    // Якобиан: [ 2*a*x,             2*y-1 ]
    //           [     1, -1/(2*sqrt(y+a)) ]
    const double j11 = 2.0 * kA * p.x;
    const double j12 = 2.0 * p.y - 1.0;
    const double j21 = 1.0;
    const double j22 = -1.0 / (2.0 * std::sqrt(p.y + kA));
    const double determinant = j11 * j22 - j12 * j21;
    if (std::abs(determinant) < 1e-15) {
        throw std::runtime_error("Якобиан вырожден");
    }

    // Решаем J*delta = -F по формулам Крамера.
    const double dx = (-value.x * j22 + j12 * value.y) / determinant;
    const double dy = (-j11 * value.y + value.x * j21) / determinant;
    return {p.x + dx, p.y + dy};
}

std::vector<Iteration> newton(double eps) {
    std::vector<Iteration> history;
    Point current{kInitialX, kInitialY};

    for (int i = 1; i <= kMaxIterations; ++i) {
        const Point next = newtonStep(current);
        const double step = norm(difference(next, current));
        const double residual = norm(systemValue(next));
        history.push_back({i, next, step, residual});
        current = next;

        if (step <= eps && residual <= eps) {
            return history;
        }
    }
    throw std::runtime_error("Метод Ньютона не сошелся");
}

Point referenceRoot() {
    Point p{kInitialX, kInitialY};
    for (int i = 0; i < 30; ++i) {
        p = newtonStep(p);
    }
    return p;
}

void printHistory(const std::string& title,
                  const std::vector<Iteration>& history,
                  const Point& exact) {
    std::cout << "\n" << title << "\n"
              << std::setw(4) << "k"
              << std::setw(18) << "x_k"
              << std::setw(18) << "y_k"
              << std::setw(18) << "||delta||"
              << std::setw(18) << "||F||"
              << std::setw(18) << "ошибка" << '\n';

    for (const auto& row : history) {
        const double error = norm(difference(row.point, exact));
        std::cout << std::setw(4) << row.number
                  << std::setw(18) << row.point.x
                  << std::setw(18) << row.point.y
                  << std::setw(18) << row.step
                  << std::setw(18) << row.residual
                  << std::setw(18) << error << '\n';
    }

    const Point answer = history.back().point;
    std::cout << "Решение: (" << answer.x << ", " << answer.y
              << "), итераций: " << history.size() << "\n";
}

double readEps() {
    double eps = 0.0;
    std::cout << "Введите точность eps (например, 1e-8): ";
    if (!(std::cin >> eps) || !std::isfinite(eps) || eps <= 0.0 || eps >= 1.0) {
        throw std::invalid_argument("eps должна быть числом из интервала (0, 1)");
    }
    return eps;
}

} // namespace

int main() {
    try {
        std::cout << std::scientific << std::setprecision(10);
        const double eps = readEps();
        const Point exact = referenceRoot();

        std::cout << "Система варианта 26 (a = 2):\n"
                  << "  2*x^2 - y + y^2 - 2 = 0\n"
                  << "  x - sqrt(y + 2) + 1 = 0\n"
                  << "Начальное приближение: (" << kInitialX << ", "
                  << kInitialY << ")\n"
                  << "Эталонное решение для оценки погрешности: ("
                  << exact.x << ", " << exact.y << ")\n";

        printHistory("Метод простой итерации", simpleIteration(eps), exact);
        printHistory("Метод Ньютона", newton(eps), exact);
    } catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
