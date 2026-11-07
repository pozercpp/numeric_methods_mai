#pragma once

#include "/home/pozercpp/numeric_methods/include/matrix.hpp"
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>

namespace numerical {
template <std::floating_point T>
Matrix<T> solve_tridiagonal(const Matrix<T> &A, const Matrix<T> &b) {
  const size_t n = A.rows();
  if (n == 0 || A.cols() != n)
    throw std::invalid_argument("A must be a nonempty square matrix");
  if (b.rows() != n || b.cols() != 1)
    throw std::invalid_argument("b must be an n x 1 column vector");
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if ((i > j + 1 || j > i + 1) && A[i, j] != T{})
        throw std::invalid_argument("A is not tridiagonal");
    }
  }
  std::vector<T> alpha(n, T{}), beta(n, T{});
  const auto check_pivot = [](T d) {
    if (!std::isfinite(d) || std::abs(d) <= std::numeric_limits<T>::epsilon())
      throw std::runtime_error("Zero or invalid pivot in tridiagonal solver");
  };
  T denominator = A[0, 0];
  check_pivot(denominator);
  if (n > 1)
    alpha[0] = A[0, 1] / denominator;
  beta[0] = b[0, 0] / denominator;
  for (size_t i = 1; i < n; ++i) {
    denominator = A[i, i] - A[i, i - 1] * alpha[i - 1];
    check_pivot(denominator);
    if (i + 1 < n)
      alpha[i] = A[i, i + 1] / denominator;
    beta[i] = (b[i, 0] - A[i, i - 1] * beta[i - 1]) / denominator;
  }
  Matrix<T> x(n, 1);
  x[n - 1, 0] = beta[n - 1];
  for (size_t i = n - 1; i-- > 0;)
    x[i, 0] = beta[i] - alpha[i] * x[i + 1, 0];
  return x;
}
} // namespace numerical