#pragma once

#include "/home/pozercpp/numeric_methods/include/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace numerical {

template <std::floating_point T> struct EigenResult {
  std::vector<T> eigenvalues;
  Matrix<T> eigenvectors;
  size_t rotations = 0;
  T off_diagonal_norm = 0;
  bool converged = false;
};

template <std::floating_point T>
EigenResult<T> jacobi_eigen(const Matrix<T> &input, T eps,
                            size_t max_rotations = 100000) {
  const size_t n = input.rows();
  if (n == 0 || input.cols() != n)
    throw std::invalid_argument("Matrix must be nonempty and square");
  if (!std::isfinite(eps) || eps <= T{0})
    throw std::invalid_argument("eps must be positive and finite");
  Matrix<T> a = input;
  Matrix<T> v(n, n);
  for (size_t i = 0; i < n; ++i) {
    v[i, i] = T{1};
    for (size_t j = 0; j < n; ++j) {
      if (!std::isfinite(a[i, j]))
        throw std::invalid_argument("Matrix must contain finite values");
      const T scale = std::max({T{1}, std::abs(a[i, j]), std::abs(a[j, i])});
      if (std::abs(a[i, j] - a[j, i]) >
          T{16} * std::numeric_limits<T>::epsilon() * scale)
        throw std::invalid_argument("Matrix must be symmetric");
    }
  }
  auto off_norm = [&]() {
    T norm = 0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j)
        norm = std::hypot(norm, a[i, j]);
    }
    return norm;
  };
  size_t rotations = 0;
  T norm = off_norm();
  while (norm > eps && rotations < max_rotations) {
    size_t p = 0, q = 1;
    T largest = 0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(a[i, j]) > largest) {
          largest = std::abs(a[i, j]);
          p = i;
          q = j;
        }
      }
    }
    if (largest == T{0})
      break;
    const T apq = a[p, q];
    const T tau = (a[q, q] - a[p, p]) / (T{2} * apq);
    const T t =
        std::copysign(T{1}, tau) / (std::abs(tau) + std::hypot(T{1}, tau));
    const T c = T{1} / std::hypot(T{1}, t);
    const T s = t * c;
    const T app = a[p, p], aqq = a[q, q];
    a[p, p] = app - t * apq;
    a[q, q] = aqq + t * apq;
    a[p, q] = a[q, p] = T{0};
    for (size_t k = 0; k < n; ++k) {
      if (k != p && k != q) {
        const T akp = a[k, p], akq = a[k, q];
        a[k, p] = a[p, k] = c * akp - s * akq;
        a[k, q] = a[q, k] = s * akp + c * akq;
      }
      const T vkp = v[k, p], vkq = v[k, q];
      v[k, p] = c * vkp - s * vkq;
      v[k, q] = s * vkp + c * vkq;
    }
    ++rotations;
    norm = off_norm();
  }
  std::vector<T> values(n);
  for (size_t i = 0; i < n; ++i)
    values[i] = a[i, i];
  return {std::move(values), std::move(v), rotations, norm, norm <= eps};
}

} // namespace numerical