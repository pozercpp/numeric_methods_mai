#pragma once

#include "/home/pozercpp/numeric_methods/include/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iomanip>

namespace numerical {

template<std::floating_point T>
struct QRResult {
    Matrix<T> Q, R;
};

// QR decomposition using Householder reflections: A = Q*R, Q^T Q = I.
template<std::floating_point T>
QRResult<T> qr_decomposition(const Matrix<T>& input) {
    const size_t rows = input.rows(), cols = input.cols();
    if (!rows || !cols || rows < cols)
        throw std::invalid_argument("QR requires rows >= columns > 0");
    Matrix<T> Q(rows, rows), R = input;
    for (size_t i = 0; i < rows; ++i) Q[i,i] = T{1};
    for (size_t k = 0; k < cols; ++k) {
        T norm = 0;
        for (size_t i = k; i < rows; ++i) norm = std::hypot(norm, R[i,k]);
        if (norm == T{}) continue;
        const T alpha = -std::copysign(norm, R[k,k]);
        std::vector<T> v(rows-k);
        for (size_t i = k; i < rows; ++i) v[i-k] = R[i,k];
        v[0] -= alpha;
        T vv = 0;
        for (T z : v) vv += z*z;
        if (vv == T{}) continue;
        const T beta = T{2}/vv;
        for (size_t j = k; j < cols; ++j) {
            T dot = 0;
            for (size_t i = k; i < rows; ++i) dot += v[i-k]*R[i,j];
            for (size_t i = k; i < rows; ++i) R[i,j] -= beta*v[i-k]*dot;
        }
        for (size_t i = 0; i < rows; ++i) {
            T dot = 0;
            for (size_t j = k; j < rows; ++j) dot += Q[i,j]*v[j-k];
            for (size_t j = k; j < rows; ++j) Q[i,j] -= beta*dot*v[j-k];
        }
    }
    return {Q,R};
}

template<std::floating_point T>
struct QREigenResult {
    std::vector<std::complex<T>> eigenvalues;
    size_t iterations = 0;
    bool converged = false;
};

namespace detail {
template<std::floating_point T>
using C = std::complex<T>;

template<std::floating_point T>
struct ComplexMatrix {
    size_t n;
    std::vector<C<T>> data;
    explicit ComplexMatrix(size_t size) : n(size), data(size*size) {}
    C<T>& operator()(size_t i, size_t j) { return data[i*n+j]; }
    const C<T>& operator()(size_t i, size_t j) const { return data[i*n+j]; }
};

template<std::floating_point T>
void householder_qr(const ComplexMatrix<T>& a, ComplexMatrix<T>& q, ComplexMatrix<T>& r) {
    const size_t n = a.n;
    r = a;
    q = ComplexMatrix<T>(n);
    for (size_t i=0;i<n;++i) q(i,i)=T{1};
    for (size_t k=0;k+1<n;++k) {
        T norm = 0;
        for (size_t i=k;i<n;++i) norm=std::hypot(norm,std::abs(r(i,k)));
        if (norm==T{}) continue;
        C<T> phase = std::abs(r(k,k))==T{} ? C<T>{1} : r(k,k)/std::abs(r(k,k));
        C<T> alpha=-phase*norm;
        std::vector<C<T>> v(n-k);
        for (size_t i=k;i<n;++i) v[i-k]=r(i,k);
        v[0]-=alpha;
        T vv=0;
        for (auto z:v) vv+=std::norm(z);
        if (vv==T{}) continue;
        T beta=T{2}/vv;
        for(size_t j=k;j<n;++j) {
            C<T> dot=0;
            for(size_t i=k;i<n;++i) dot+=std::conj(v[i-k])*r(i,j);
            for(size_t i=k;i<n;++i) r(i,j)-=beta*v[i-k]*dot;
        }
        for(size_t i=0;i<n;++i) {
            C<T> dot=0;
            for(size_t j=k;j<n;++j) dot+=q(i,j)*v[j-k];
            for(size_t j=k;j<n;++j) q(i,j)-=beta*dot*std::conj(v[j-k]);
        }
    }
}

template<std::floating_point T>
void hessenberg(ComplexMatrix<T>& a) {
    size_t n=a.n;
    for(size_t k=0;k+2<n;++k) {
        T norm=0;
        for(size_t i=k+1;i<n;++i) norm=std::hypot(norm,std::abs(a(i,k)));
        if(norm==T{}) continue;
        C<T> phase=std::abs(a(k+1,k))==T{} ? C<T>{1} : a(k+1,k)/std::abs(a(k+1,k));
        std::vector<C<T>> v(n-k-1);
        for(size_t i=k+1;i<n;++i) v[i-k-1]=a(i,k);
        v[0]+=phase*norm;
        T vv=0;
        for(auto z:v) vv+=std::norm(z);
        if(vv==T{}) continue;
        T beta=T{2}/vv;
        for(size_t j=k;j<n;++j) {
            C<T> dot=0;
            for(size_t i=k+1;i<n;++i) dot+=std::conj(v[i-k-1])*a(i,j);
            for(size_t i=k+1;i<n;++i) a(i,j)-=beta*v[i-k-1]*dot;
        }
        for(size_t i=0;i<n;++i) {
            C<T> dot=0;
            for(size_t j=k+1;j<n;++j) dot+=a(i,j)*v[j-k-1];
            for(size_t j=k+1;j<n;++j) a(i,j)-=beta*dot*std::conj(v[j-k-1]);
        }
        for(size_t i=k+2;i<n;++i) a(i,k)=0;
    }
}

template<std::floating_point T>
C<T> wilkinson_shift(const ComplexMatrix<T>& a, size_t m) {
    auto aa=a(m-2,m-2), bb=a(m-2,m-1), cc=a(m-1,m-2), dd=a(m-1,m-1);
    C<T> trace=(aa+dd)/T{2};
    C<T> delta=std::sqrt((aa-dd)*(aa-dd)/T{4}+bb*cc);
    C<T> l1=trace+delta, l2=trace-delta;
    return std::abs(l1-dd)<std::abs(l2-dd)?l1:l2;
}
} // namespace detail

// Shifted QR iterations in complex arithmetic; supports non-symmetric real matrices.
template<std::floating_point T>
QREigenResult<T> qr_eigenvalues(const Matrix<T>& input, T eps, size_t max_iterations=100000) {
    using C=std::complex<T>;
    size_t n=input.rows();
    if(n==0 || input.cols()!=n) throw std::invalid_argument("A must be a nonempty square matrix");
    if(!std::isfinite(eps) || eps<=0) throw std::invalid_argument("eps must be finite and positive");
    detail::ComplexMatrix<T> a(n);
    for(size_t i=0;i<n;++i)
        for(size_t j=0;j<n;++j) {
            if(!std::isfinite(input[i,j])) throw std::invalid_argument("A contains non-finite values");
            a(i,j)=input[i,j];
        }
    detail::hessenberg(a);
    QREigenResult<T> result;
    result.eigenvalues.resize(n);
    size_t m=n;
    while(m>1 && result.iterations < max_iterations) {
        T scale=std::abs(a(m-2,m-2))+std::abs(a(m-1,m-1));
        if(std::abs(a(m-1,m-2))<=eps*std::max(T{1},scale)) {
            result.eigenvalues[m-1]=a(m-1,m-1);
            --m;
            continue;
        }
        C mu=detail::wilkinson_shift(a,m);
        detail::ComplexMatrix<T> block(m),q(m),r(m);
        for(size_t i=0;i<m;++i)
            for(size_t j=0;j<m;++j) block(i,j)=a(i,j)-(i==j?mu:C{});
        detail::householder_qr(block,q,r);
        for(size_t i=0;i<m;++i)
            for(size_t j=0;j<m;++j) {
                C sum=0;
                for(size_t k=0;k<m;++k) sum+=r(i,k)*q(k,j);
                a(i,j)=sum+(i==j?mu:C{});
            }
        // Exact Hessenberg structure is preserved by QR in exact arithmetic.
        for(size_t i=0;i<m;++i)
            for(size_t j=0;j+1<i;++j) a(i,j)=0;
        ++result.iterations;
    }
    if(m==1) { result.eigenvalues[0]=a(0,0); m=0; }
    result.converged=(m==0);
    if(!result.converged)
        for(size_t i=0;i<m;++i) result.eigenvalues[i]=a(i,i); // provisional only
    return result;
}
} // namespace numerical
