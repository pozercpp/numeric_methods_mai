#pragma once

#include "/home/pozercpp/numeric_methods/include/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace numerical {

template <std::floating_point T> struct IterativeResult {
  Matrix<T> x;
  size_t iterations = 0;
  T residual = T{};
  T last_change = T{};
  bool converged = false;
};

namespace detail {
template <std::floating_point T> void validate(const Matrix<T> &A, const Matrix<T> &b, T eps, size_t max_iterations) {
  const size_t n = A.rows();
  if (n == 0 || A.cols() != n || b.rows() != n || b.cols() != 1) throw std::invalid_argument("Expected square A and n x 1 vector b");
  if (!std::isfinite(eps) || eps <= T{} || max_iterations == 0) throw std::invalid_argument("eps must be positive and max_iterations nonzero");
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(b[i, 0])) throw std::invalid_argument("Non-finite rhs");
    for (size_t j = 0; j < n; ++j)
      if (!std::isfinite(A[i, j])) throw std::invalid_argument("Non-finite matrix entry");
    if (A[i, i] == T{}) throw std::invalid_argument("Zero diagonal element");
  }
}

template <std::floating_point T> T residual_norm(const Matrix<T> &A, const Matrix<T> &x, const Matrix<T> &b) {
  T norm = T{};
  for (size_t i = 0; i < A.rows(); ++i) {
    T s = -b[i, 0];
    for (size_t j = 0; j < A.cols(); ++j)
      s += A[i, j] * x[j, 0];
    norm = std::max(norm, std::abs(s));
  }
  return norm;
}

template <std::floating_point T, bool seidel> IterativeResult<T> iterate(const Matrix<T> &A, const Matrix<T> &b, T eps, size_t max_iterations) {
  validate(A, b, eps, max_iterations);
  const size_t n = A.rows();
  Matrix<T> x(n, 1); // x(0) = 0
  Matrix<T> next(n, 1);
  IterativeResult<T> result;
  for (size_t k = 1; k <= max_iterations; ++k) {
    T change = T{};
    for (size_t i = 0; i < n; ++i) {
      T s = b[i, 0];
      for (size_t j = 0; j < n; ++j) {
        if (j == i) continue;
        if constexpr (seidel) {
          s -= A[i, j] * (j < i ? next[j, 0] : x[j, 0]);
        } else {
          s -= A[i, j] * x[j, 0];
        }
      }
      next[i, 0] = s / A[i, i];
      change = std::max(change, std::abs(next[i, 0] - x[i, 0]));
    }
    T residual = residual_norm(A, next, b);
    result.iterations = k;
    result.last_change = change;
    result.residual = residual;
    if (!std::isfinite(change) || !std::isfinite(residual)) throw std::runtime_error("Iteration diverged (non-finite values)");
    if (change <= eps && residual <= eps) {
      result.x = std::move(next);
      result.converged = true;
      return result;
    }
    std::swap(x, next);
  }
  result.x = std::move(x);
  return result;
}
} // namespace detail
template <std::floating_point T> IterativeResult<T> simple_iterations(const Matrix<T> &A, const Matrix<T> &b, T eps, size_t max_iterations = 100000) {
  return detail::iterate<T, false>(A, b, eps, max_iterations);
}
template <std::floating_point T> IterativeResult<T> gauss_seidel(const Matrix<T> &A, const Matrix<T> &b, T eps, size_t max_iterations = 100000) {
  return detail::iterate<T, true>(A, b, eps, max_iterations);
}
} // namespace numerical