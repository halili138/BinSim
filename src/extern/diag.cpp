#include "diag/davidson.hpp"

extern "C"
{
    double build_ritz_and_residual_f64(
        const int64_t N,
        const int dim,
        const double *__restrict__ coeffs,
        const double min_e,
        const double **__restrict__ V,
        const double **__restrict__ AV,
        double *__restrict__ axt)
    {
        return build_ritz_and_residual<double>(N, dim, coeffs, min_e, V, AV, axt);
    }

    void build_ritz_vector_f64(
        const int64_t N,
        const int dim,
        const double *__restrict__ coeffs,
        const double **__restrict__ V,
        double *__restrict__ xt)
    {
        build_ritz_vector<double>(N, dim, coeffs, V, xt);
    }

    void apply_preconditioner_inplace_f64(
        const int64_t N,
        double *__restrict__ axt,
        const double *__restrict__ diags,
        const double min_e,
        const double shift)
    {
        apply_preconditioner_inplace<double>(N, axt, diags, min_e, shift);
    }

    double mgs_orthogonalize_f64(
        const int64_t N,
        const int dim,
        const double **__restrict__ V,
        double *__restrict__ vt)
    {
        return mgs_orthogonalize<double>(N, dim, V, vt);
    }

    double inplace_normalize_f64(
        const int64_t N,
        double *__restrict__ vec)
    {
        return inplace_normalize<double>(N, vec);
    }

    double fast_dot_f64(
        const int64_t N,
        const double *__restrict__ v1,
        const double *__restrict__ v2)
    {
        return fast_dot<double>(N, v1, v2);
    }

    void compute_heff_col_f64(
        const int64_t N,
        const int dim,
        const double **__restrict__ V,
        const double *__restrict__ AV_dim,
        double *__restrict__ out_col)
    {
        compute_heff_col<double>(N, dim, V, AV_dim, out_col);
    }

    void rebuild_heff_f64(
        const int64_t N,
        const int dim,
        const double **__restrict__ V,
        const double **__restrict__ AV,
        double *__restrict__ heff,
        const int ldh)
    {
        rebuild_heff<double>(N, dim, V, AV, heff, ldh);
    }
}

extern "C"
{
    double build_ritz_and_residual_c64(
        const int64_t N,
        const int dim,
        const complexf64 *__restrict__ coeffs,
        const double min_e,
        const complexf64 **__restrict__ V,
        const complexf64 **__restrict__ AV,
        complexf64 *__restrict__ axt)
    {
        return build_ritz_and_residual<complexf64>(N, dim, coeffs, min_e, V, AV, axt);
    }

    void build_ritz_vector_c64(
        const int64_t N,
        const int dim,
        const complexf64 *__restrict__ coeffs,
        const complexf64 **__restrict__ V,
        complexf64 *__restrict__ xt)
    {
        build_ritz_vector<complexf64>(N, dim, coeffs, V, xt);
    }

    void apply_preconditioner_inplace_c64(
        const int64_t N,
        complexf64 *__restrict__ axt,
        const complexf64 *__restrict__ diags,
        const double min_e,
        const double shift)
    {
        apply_preconditioner_inplace<complexf64>(N, axt, diags, min_e, shift);
    }

    double mgs_orthogonalize_c64(
        const int64_t N,
        const int dim,
        const complexf64 **__restrict__ V,
        complexf64 *__restrict__ vt)
    {
        return mgs_orthogonalize<complexf64>(N, dim, V, vt);
    }

    double inplace_normalize_c64(
        const int64_t N,
        complexf64 *__restrict__ vec)
    {
        return inplace_normalize<complexf64>(N, vec);
    }

    complexf64 fast_dot_c64(
        const int64_t N,
        const complexf64 *__restrict__ v1,
        const complexf64 *__restrict__ v2)
    {
        return fast_dot<complexf64>(N, v1, v2);
    }

    void compute_heff_col_c64(
        const int64_t N,
        const int dim,
        const complexf64 **__restrict__ V,
        const complexf64 *__restrict__ AV_dim,
        complexf64 *__restrict__ out_col)
    {
        compute_heff_col<complexf64>(N, dim, V, AV_dim, out_col);
    }

    void rebuild_heff_c64(
        const int64_t N,
        const int dim,
        const complexf64 **__restrict__ V,
        const complexf64 **__restrict__ AV,
        complexf64 *__restrict__ heff,
        const int ldh)
    {
        rebuild_heff<complexf64>(N, dim, V, AV, heff, ldh);
    }
}
