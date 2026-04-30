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
        const int64 *__restrict__ orbsym,
        double *vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        set_det_coeff<uint32, double>(basis, target_astr, target_bstr, coeff, orbsym, vec);
    }

    void set_det_coeff_c64(
        void *__restrict__ basis_ptr,
        const uint32 target_astr,
        const uint32 target_bstr,
        const complexf64 coeff,
        const int64 *__restrict__ orbsym,
        complexf64 *vec)
    {
        const BasisManager<uint32> *basis = static_cast<const BasisManager<uint32> *>(basis_ptr);
        set_det_coeff<uint32, complexf64>(basis, target_astr, target_bstr, coeff, orbsym, vec);
    }
}
