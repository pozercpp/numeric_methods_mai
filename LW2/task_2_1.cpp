#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kInitialX = 1.0; // Начальное приближение выбрано по графику.
constexpr int kMaxIterations = 10000;

struct Iteration {
    int number;
    double x;
    double step;
    double residual;
};

double f(double x) {
    if (x + 1.0 <= 0.0) {
        throw std::domain_error("Аргумент логарифма должен быть положительным");
    }
    return std::log10(x + 1.0) - x + 0.5;
}

double derivative(double x) {
    return 1.0 / ((x + 1.0) * std::log(10.0)) - 1.0;
}

// lg(x + 1) - x + 0.5 = 0  <=>  x = lg(x + 1) + 0.5.
double phi(double x) {
    if (x + 1.0 <= 0.0) throw std::domain_error("Итерация вышла из области определения");
    return std::log10(x + 1.0) + 0.5;
}

std::vector<Iteration> simpleIteration(double eps) {
    std::vector<Iteration> history;
    double x = kInitialX;
    for (int i = 1; i <= kMaxIterations; ++i) {
        const double next = phi(x);
        const double step = std::abs(next - x);
        const double residual = std::abs(f(next));
        history.push_back({i, next, step, residual});
        x = next;
        if (step <= eps && residual <= eps) return history;
    }
    throw std::runtime_error("Метод простой итерации не сошелся");
}

std::vector<Iteration> newton(double eps) {
    std::vector<Iteration> history;
    double x = kInitialX;
    for (int i = 1; i <= kMaxIterations; ++i) {
        const double df = derivative(x);
        if (std::abs(df) < 1e-15) throw std::runtime_error("В методе Ньютона производная близка к нулю");
        const double next = x - f(x) / df;
        const double step = std::abs(next - x);
        const double residual = std::abs(f(next));
        history.push_back({i, next, step, residual});
        x = next;
        if (step <= eps && residual <= eps) return history;
    }
    throw std::runtime_error("Метод Ньютона не сошелся");
}

double referenceRoot() {
    double x = kInitialX;
    for (int i = 0; i < 30; ++i) x -= f(x) / derivative(x);
    return x;
}

void printHistory(const std::string& title,
                  const std::vector<Iteration>& history,
                  double exact) {
    std::cout << "\n" << title << "\n"
              << std::setw(4) << "k"
              << std::setw(18) << "x_k"
              << std::setw(18) << "|dx|"
              << std::setw(18) << "|f(x_k)|"
              << std::setw(18) << "|x_k-x*|" << '\n';

    for (const auto& row : history) {
        std::cout << std::setw(4) << row.number
                  << std::setw(18) << row.x
                  << std::setw(18) << row.step
                  << std::setw(18) << row.residual
                  << std::setw(18) << std::abs(row.x - exact) << '\n';
    }

    std::cout << "Корень: " << history.back().x
              << ", итераций: " << history.size() << "\n";
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
        const double exact = referenceRoot();

        std::cout << "Уравнение варианта 26: lg(x + 1) - x + 0.5 = 0\n"
                  << "Начальное приближение x0 = " << kInitialX << '\n'
                  << "Эталонный корень для оценки погрешности: " << exact << '\n';

        printHistory("Метод простой итерации", simpleIteration(eps), exact);
        printHistory("Метод Ньютона", newton(eps), exact);
    } catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
