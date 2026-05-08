#include "common.hpp"

extern "C"
{
    int64 get_subspace_dim(void *basis_ptr)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        return get_subspace_dim_tmpl<uint32>(basis);
    }

    void destroy_basis_manager(void *basis_ptr)
    {
        BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);
        destroy_basis_manager_tmpl<uint32>(basis);
    }

    void *create_basis_manager(
        const int64 norb,
        const int64 na,
        const int64 nb,
        const int64 total_sym,
        const int64 *__restrict__ orbsym,
        const int64 num_irreps)
    {
        return create_basis_manager_tmpl<uint32>(norb, na, nb, total_sym, orbsym, num_irreps);
    }

    void set_det_coeff_f64(
        void *__restrict__ basis_ptr,
        const uint32 target_astr,
        const uint32 target_bstr,
        const double coeff,
        double *vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        set_det_coeff<uint32, double>(basis, target_astr, target_bstr, coeff, vec);
    }

    void set_det_coeff_c64(
        void *__restrict__ basis_ptr,
        const uint32 target_astr,
        const uint32 target_bstr,
        const complexf64 coeff,
        complexf64 *vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        set_det_coeff<uint32, complexf64>(basis, target_astr, target_bstr, coeff, vec);
    }

    void compute_diagonal_elements_raw_f64(
        void *__restrict__ basis_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 nterms,
        double *__restrict__ diags)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);

        compute_diagonal_elements_raw<uint32, double>(
            basis, azs, bzs, cs, nterms, diags);
    }

    void compute_diagonal_elements_raw_c64(
        void *__restrict__ basis_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const complexf64 *__restrict__ cs,
        const int64 nterms,
        complexf64 *__restrict__ diags)
    {
        const BasisManager<uint32> *basis = static_cast<BasisManager<uint32> *>(basis_ptr);

        compute_diagonal_elements_raw<uint32, complexf64>(
            basis, azs, bzs, cs, nterms, diags);
    }

    void *create_custom_basis_manager(
        int64 norb,
        const uint32 *input_astrs, int64 num_astrs,
        const uint32 *input_bstrs, int64 num_bstrs,
        const int64 *orbsym, int64 total_sym, int64 num_irreps)
    {
        return create_custom_basis_manager_tmpl<uint32>(
            norb,
            input_astrs, num_astrs,
            input_bstrs, num_bstrs,
            orbsym, total_sym, num_irreps);
    }
}
