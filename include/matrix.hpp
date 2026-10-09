#pragma once

#include <utility>
#include <vector>
#include <type_traits>
#include <stdexcept>

template<typename T>
concept arithmetic = std::is_arithmetic<T>;

template<arithmetic T>
class Matrix {
private:
    size_t n, m;
    std::vector<T> v;
    size_t get_ind(size_t i, size_t j) { return i * m + j; }
public:
    Matrix() = default;
    Matrix(size_t n_, size_t m_) : n(n_), m(m_), v(n_ * m_) {}
    T& operator[] (size_t i, size_t j) { return v[get_ind(i, j)]; }
    const T& operator[] (size_t i, size_t j) const { return v[get_ind(i, j)]; }
    T& at(size_t i, size_t j) { return v.at(get_ind(i, j)); }
    const T& at(size_t i, size_t j) const { return v.at(get_ind(i, j)); }
    Matrix operator+(const Matrix& other) const {
        if (n != other.n || m != other.m) throw std::invalid_argument("Matrix dimensions mismatch");
        Matrix res(n, m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                res.v[get_ind(i, j)] = v[get_ind(i, j)] + other.v[get_ind(i, j)];
            }
        }
        return res;
    }
    Matrix& operator+=(const Matrix& other) {
        if (n != other.n || m != other.m) throw std::invalid_argument("Matrix dimensions mismatch");
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                v[get_ind(i, j)] += other.v[get_ind(i, j)];
            }
        }
        return *this;
    }
    Matrix operator-(const Matrix& other) const {
        if (n != other.n || m != other.m) throw std::invalid_argument("Matrix dimensions mismatch");
        Matrix res(n, m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                res.v[get_ind(i, j)] = v[get_ind(i, j)] - other.v[get_ind(i, j)];
            }
        }
        return res;
    }
    Matrix& operator-=(const Matrix& other) {
        if (n != other.n || m != other.m) throw std::invalid_argument("Matrix dimensions mismatch");
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                v[get_ind(i, j)] -= other.v[get_ind(i, j)];
            }
        }
        return *this;
    }
    Matrix operator*(const Matrix& other) const {
        if (m != other.n) throw std::invalid_argument("Matrix dimensions mismatch");
        Matrix res(n, other.m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < other.m; ++j) {
                for (size_t p = 0; p < m; ++p) {
                    res.v[get_ind(i, j)] += v[i][p] * other.v[p][j];
                }
            }
        }
        return res;
    }
    Matrix& operator *=(const Matrix& other) { 
        *this = *this * other;
        return *this;
    }
    Matrix& operator*=(const T& val) const {
        for (size_t i{0}; i < n; ++i) {
            for (size_t j{0}; j < m; ++j) {
                v[get_ind(i, j)] *= val;
            }
        }
    }
    Matrix& operator/=(const T& val) const {
        for (size_t i{0}; i < n; ++i) {
            for (size_t j{0}; j < m; ++j) {
                v[get_ind(i, j)] /= val;
            }
        }
        return *this;
    }
    Matrix operator*(const T& val) const {
        Matrix res = *this;
        res *= val;
        return res;
    }
    Matrix operator/(const T& val) const {
        Matrix res = *this;
        res /= val;
        return res;
    }
};