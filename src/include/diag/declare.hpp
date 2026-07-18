#pragma once
#include "diag/davidson.hpp"

#define DECLARE_DIAG_INTERFACES(Tv, SUFFIX)                                        \
    extern "C"                                                                     \
    {                                                                              \
        double build_ritz_and_residual_##SUFFIX(                                   \
            const int64_t N, const int dim,                                        \
            const Tv *__restrict__ coeffs, const double min_e,                     \
            const Tv **__restrict__ V, const Tv **__restrict__ AV,                 \
            Tv *__restrict__ axt)                                                  \
        {                                                                          \
            return build_ritz_and_residual<Tv>(N, dim, coeffs, min_e, V, AV, axt); \
        }                                                                          \
                                                                                   \
        void build_ritz_vector_##SUFFIX(                                           \
            const int64_t N, const int dim,                                        \
            const Tv *__restrict__ coeffs,                                         \
            const Tv **__restrict__ V,                                             \
            Tv *__restrict__ xt)                                                   \
        {                                                                          \
            build_ritz_vector<Tv>(N, dim, coeffs, V, xt);                          \
        }                                                                          \
                                                                                   \
        void apply_preconditioner_inplace_##SUFFIX(                                \
            const int64_t N,                                                       \
            Tv *__restrict__ axt,                                                  \
            const Tv *__restrict__ diags,                                          \
            const double min_e, const double shift)                                \
        {                                                                          \
            apply_preconditioner_inplace<Tv>(N, axt, diags, min_e, shift);         \
        }                                                                          \
                                                                                   \
        double mgs_orthogonalize_##SUFFIX(                                         \
            const int64_t N, const int dim,                                        \
            const Tv **__restrict__ V,                                             \
            Tv *__restrict__ vt)                                                   \
        {                                                                          \
            return mgs_orthogonalize<Tv>(N, dim, V, vt);                           \
        }                                                                          \
                                                                                   \
        double inplace_normalize_##SUFFIX(                                         \
            const int64_t N,                                                       \
            Tv *__restrict__ vec)                                                  \
        {                                                                          \
            return inplace_normalize<Tv>(N, vec);                                  \
        }                                                                          \
                                                                                   \
        Tv fast_dot_##SUFFIX(                                                      \
            const int64_t N,                                                       \
            const Tv *__restrict__ v1,                                             \
            const Tv *__restrict__ v2)                                             \
        {                                                                          \
            return fast_dot<Tv>(N, v1, v2);                                        \
        }                                                                          \
                                                                                   \
        void compute_heff_col_##SUFFIX(                                            \
            const int64_t N, const int dim,                                        \
            const Tv **__restrict__ V,                                             \
            const Tv *__restrict__ AV_dim,                                         \
            Tv *__restrict__ out_col)                                              \
        {                                                                          \
            compute_heff_col<Tv>(N, dim, V, AV_dim, out_col);                      \
        }                                                                          \
                                                                                   \
        void rebuild_heff_##SUFFIX(                                                \
            const int64_t N, const int dim,                                        \
            const Tv **__restrict__ V,                                             \
            const Tv **__restrict__ AV,                                            \
            Tv *__restrict__ heff,                                                 \
            const int ldh)                                                         \
        {                                                                          \
            rebuild_heff<Tv>(N, dim, V, AV, heff, ldh);                            \
        }                                                                          \
    }
