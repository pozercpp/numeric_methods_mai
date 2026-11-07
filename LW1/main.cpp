#include <iostream>

#include "/home/pozercpp/numeric_methods/LW1/include/Iterative.hpp"
#include "/home/pozercpp/numeric_methods/LW1/include/JacobiEigen.hpp"
#include "/home/pozercpp/numeric_methods/LW1/include/QREigen.hpp"
#include "/home/pozercpp/numeric_methods/LW1/include/lu.hpp"
#include "/home/pozercpp/numeric_methods/LW1/include/tridiagonal.hpp"
#include "/home/pozercpp/numeric_methods/include/matrix.hpp"

constexpr double EPS = 1e-10;
constexpr size_t MAX_ITERATIONS = 100000;

template <typename T> void print_matrix(const numerical::Matrix<T>& A) {
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      std::cout << std::setw(15) << A[i, j] << ' ';
    }
    std::cout << '\n';
  }
}

[[nodiscard]] std::pair<numerical::Matrix<double>, numerical::Matrix<double>> read_system(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);
  size_t n, m;
  if (!(file >> n >> m)) throw std::runtime_error("Cannot read matrix dimensions");
  if (n == 0 || n != m) throw std::invalid_argument("Matrix must be square and n must be greater than 0");
  numerical::Matrix<double> A(n, m);
  numerical::Matrix<double> b(n, 1);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (!(file >> A[i, j])) throw std::runtime_error("Cannot read matrix");
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (!(file >> b[i, 0])) throw std::runtime_error("Cannot read RHS");
  }
  return {A, b};
}

void run(const std::string& name, auto&& func) {
  std::cout << "\n========== " << name << " ==========\n";
  try {
    func();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}

auto main() -> signed {
  std::cout << std::fixed << std::setprecision(10);
  run("LU decomposition", [] {
    auto [A, b] = read_system("/home/pozercpp/numeric_methods/LW1/data/lu.txt");
    numerical::LU<double> lu(A);
    numerical::Matrix<double> x = lu.solve(b);
    std::cout << "Solution:\n";
    print_matrix(x);
    std::cout << "\nDeterminant: " << lu.determinant() << '\n';
    std::cout << "\nInverse matrix:\n";
    print_matrix(lu.inverse());
  });
  run("QR decomposition", [] {
    numerical::Matrix<double> A;
    A.read_from_file("/home/pozercpp/numeric_methods/LW1/data/qr.txt");
    auto result = numerical::qr_decomposition(A);
    std::cout << "Q:\n";
    print_matrix(result.Q);
    std::cout << "\nR:\n";
    print_matrix(result.R);

    auto eigen_result = numerical::qr_eigenvalues(A, EPS, MAX_ITERATIONS);
    std::cout << "\nEigenvalues:\n";
    for (const auto& value : eigen_result.eigenvalues) {
      std::cout << value << '\n';
    }
    std::cout << "Iterations: " << eigen_result.iterations << '\n';
    std::cout << "Converged: " << std::boolalpha
              << eigen_result.converged << '\n';
  });
  run("Simple iterations", [] {
    auto [A, b] = read_system("/home/pozercpp/numeric_methods/LW1/data/Iterative.txt");
    auto result = numerical::simple_iterations(A, b, EPS, MAX_ITERATIONS);
    std::cout << "Solution:\n";
    print_matrix(result.x);
    std::cout << "Iterations: " << result.iterations << '\n';
    std::cout << "Residual: " << result.residual << '\n';
    std::cout << "Last change: " << result.last_change << '\n';
    std::cout << "Converged: " << std::boolalpha << result.converged << '\n';
  });
  run("Gauss-Seidel", [] {
    auto [A, b] = read_system("/home/pozercpp/numeric_methods/LW1/data/Iterative.txt");
    auto result = numerical::gauss_seidel(A, b, EPS, MAX_ITERATIONS);
    std::cout << "Solution:\n";
    print_matrix(result.x);
    std::cout << "Iterations: " << result.iterations << '\n';
    std::cout << "Residual: " << result.residual << '\n';
    std::cout << "Last change: " << result.last_change << '\n';
    std::cout << "Converged: " << std::boolalpha << result.converged << '\n';
  });
  run("Jacobi eigenvalues", [] {
    numerical::Matrix<double> A;
    A.read_from_file("/home/pozercpp/numeric_methods/LW1/data/jacobi.txt");
    auto result = numerical::jacobi_eigen(A, EPS);
    std::cout << "Eigenvalues:\n";
    for (const auto& value : result.eigenvalues) std::cout << value << '\n';
    std::cout << "\nEigenvectors:\n";
    print_matrix(result.eigenvectors);
    std::cout << "Rotations: " << result.rotations << '\n';
    std::cout << "Off-diagonal norm: " << result.off_diagonal_norm << '\n';
    std::cout << "Converged: " << std::boolalpha << result.converged << '\n';
  });
  run("Tridiagonal solver", [] {
    auto [A, b] = read_system("/home/pozercpp/numeric_methods/LW1/data/tridiagonal.txt");
    numerical::Matrix<double> x = numerical::solve_tridiagonal(A, b);
    std::cout << "Solution:\n";
    print_matrix(x);
  });
  return 0;
}
