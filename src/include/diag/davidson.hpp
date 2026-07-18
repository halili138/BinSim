#pragma once
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <complex>
#include <type_traits>

#define FORCE_INLINE inline __attribute__((always_inline))

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

using complexf64 = std::complex<double>;

template <typename T>
using RealT = decltype(std::abs(T{}));

template <typename T>
FORCE_INLINE T math_conj(const T &x)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        return x;
    }
    else
    {
        return std::conj(x);
    }
}

template <typename T>
RealT<T> build_ritz_and_residual(
    const int64_t N,
    const int dim,
    const T *__restrict__ coeffs,
    const RealT<T> min_e,
    const T *const *__restrict__ V,
    const T *const *__restrict__ AV,
    T *__restrict__ axt)
{
    const T *local_V[64];
    const T *local_AV[64];
    T local_c[64];
    for (int j = 0; j < dim; ++j)
    {
        local_V[j] = *(V + j);
        local_AV[j] = *(AV + j);
        local_c[j] = *(coeffs + j);
    }
    RealT<T> norm_r_sq = {};
#pragma omp parallel for schedule(static) reduction(+ : norm_r_sq)
    for (int64_t i = 0; i < N; ++i)
    {
        T val_x = {};
        T val_ax = {};
#pragma GCC unroll 8
        for (int j = 0; j < dim; ++j)
        {
            const T c = local_c[j];
            val_x += c * *(local_V[j] + i);
            val_ax += c * *(local_AV[j] + i);
        }
        T val_r = val_ax - min_e * val_x;
        *(axt + i) = val_r;

        norm_r_sq += std::norm(val_r);
    }
    return std::sqrt(norm_r_sq);
}

template <typename T>
void build_ritz_vector(
    const int64_t N,
    const int dim,
    const T *__restrict__ coeffs,
    const T *const *__restrict__ V,
    T *__restrict__ xt)
{
    const T *local_V[64];
    T local_c[64];
    for (int j = 0; j < dim; ++j)
    {
        local_V[j] = *(V + j);
        local_c[j] = *(coeffs + j);
    }
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i)
    {
        T val_x = {};
#pragma GCC unroll 8
        for (int j = 0; j < dim; ++j)
            val_x += local_c[j] * *(local_V[j] + i);
        *(xt + i) = val_x;
    }
}

template <typename T>
void apply_preconditioner_inplace(
    const int64_t N,
    T *__restrict__ axt,
    const T *__restrict__ diags,
    const RealT<T> min_e,
    const RealT<T> shift)
{
    const RealT<T> abs_shift = std::abs(shift);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i)
    {
        RealT<T> diff_real = std::real(*(diags + i)) - min_e;
        RealT<T> abs_diff = std::abs(diff_real);
        if (abs_diff < abs_shift)
        {
            diff_real = (diff_real >= 0) ? abs_shift : -abs_shift;
        }
        *(axt + i) = *(axt + i) / diff_real;
    }
}

template <typename T>
RealT<T> mgs_orthogonalize(
    const int64_t N,
    const int dim,
    const T *const *__restrict__ V,
    T *__restrict__ vt)
{
    for (int j = 0; j < dim; ++j)
    {
        const T *__restrict__ vj = *(V + j);
        T proj = {};
#pragma omp parallel for schedule(static) reduction(+ : proj)
        for (int64_t i = 0; i < N; ++i)
        {
            proj += math_conj(*(vj + i)) * *(vt + i);
        }
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
            *(vt + i) -= proj * *(vj + i);
        }
    }
    RealT<T> norm_sq = {};
#pragma omp parallel for schedule(static) reduction(+ : norm_sq)
    for (int64_t i = 0; i < N; ++i)
    {
        norm_sq += std::norm(*(vt + i));
    }
    return std::sqrt(norm_sq);
}

template <typename T>
RealT<T> inplace_normalize(
    const int64_t N,
    T *__restrict__ vec)
{
    RealT<T> norm_sq = {};
#pragma omp parallel for schedule(static) reduction(+ : norm_sq)
    for (int64_t i = 0; i < N; ++i)
    {
        norm_sq += std::norm(*(vec + i));
    }
    const RealT<T> nrm = std::sqrt(norm_sq);
    const T inv_nrm = static_cast<T>(1.0 / nrm);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i)
    {
        *(vec + i) *= inv_nrm;
    }
    return nrm;
}

template <typename T>
T fast_dot(
    const int64_t N,
    const T *__restrict__ v1,
    const T *__restrict__ v2)
{
    T res = {};
#pragma omp parallel for schedule(static) reduction(+ : res)
    for (int64_t i = 0; i < N; ++i)
    {
        res += math_conj(*(v1 + i)) * *(v2 + i);
    }
    return res;
}

template <typename T>
void compute_heff_col(
    const int64_t N, const int dim,
    const T *const *__restrict__ V, const T *__restrict__ AV_dim,
    T *__restrict__ out_col)
{
    const T *local_V[64];
    for (int j = 0; j < dim; ++j)
    {
        local_V[j] = *(V + j);
        *(out_col + j) = {};
    }
#pragma omp parallel
    {
        T local_acc[64] = {};
#pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
            const T av_val = *(AV_dim + i);
#pragma GCC unroll 8
            for (int j = 0; j < dim; ++j)
            {
                local_acc[j] += math_conj(*(local_V[j] + i)) * av_val;
            }
        }
#pragma omp critical
        {
            for (int j = 0; j < dim; ++j)
            {
                *(out_col + j) += local_acc[j];
            }
        }
    }
}

template <typename T>
void rebuild_heff(
    const int64_t N,
    const int dim,
    const T *const *__restrict__ V,
    const T *const *__restrict__ AV,
    T *__restrict__ heff,
    const int ldh)
{
    const T *local_V[64];
    const T *local_AV[64];
    for (int j = 0; j < dim; ++j)
    {
        local_V[j] = *(V + j);
        local_AV[j] = *(AV + j);
    }
    for (int j = 0; j < dim; ++j)
    {
        for (int i = 0; i < dim; ++i)
        {
            *(heff + i + j * ldh) = {};
        }
    }
#pragma omp parallel
    {
        T local_heff[64][64] = {};
#pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
#pragma GCC unroll 4
            for (int col = 0; col < dim; ++col)
            {
                const T av_val = *(local_AV[col] + i);
#pragma GCC unroll 4
                for (int row = 0; row < dim; ++row)
                {
                    local_heff[row][col] += math_conj(*(local_V[row] + i)) * av_val;
                }
            }
        }
#pragma omp critical
        {
            for (int col = 0; col < dim; ++col)
            {
                for (int row = 0; row < dim; ++row)
                {
                    *(heff + row + col * ldh) += local_heff[row][col];
                }
            }
        }
    }
}
