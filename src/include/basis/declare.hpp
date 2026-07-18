#pragma once
#include "basis/basis.hpp"

#define DECLARE_BASIS_INTERFACES(Ti, SUFFIX)                                                    \
    extern "C"                                                                                  \
    {                                                                                           \
        int64 get_subspace_dim##SUFFIX(void *basis_ptr)                                         \
        {                                                                                       \
            return get_subspace_dim_tmpl<Ti>(static_cast<const BasisManager<Ti> *>(basis_ptr)); \
        }                                                                                       \
                                                                                                \
        int64 get_num_symmetry_blocks##SUFFIX(void *basis_ptr)                                  \
        {                                                                                       \
            return static_cast<const BasisManager<Ti> *>(basis_ptr)->num_blocks;                \
        }                                                                                       \
                                                                                                \
        void destroy_basis_manager##SUFFIX(void *basis_ptr)                                     \
        {                                                                                       \
            auto *basis = static_cast<BasisManager<Ti> *>(basis_ptr);                           \
            basis->clear();                                                                     \
            delete basis;                                                                       \
        }                                                                                       \
                                                                                                \
        void *create_basis_manager##SUFFIX(                                                     \
            const int64 norb, const int64 na, const int64 nb,                                   \
            const int64 total_sym,                                                              \
            const int64 *__restrict__ orbsym,                                                   \
            const int64 num_irreps)                                                             \
        {                                                                                       \
            return create_basis_manager_tmpl<Ti>(                                               \
                norb, na, nb, total_sym, orbsym, num_irreps);                                   \
        }                                                                                       \
                                                                                                \
        void set_det_coeff_f64##SUFFIX(                                                         \
            void *__restrict__ basis_ptr,                                                       \
            const Ti target_astr,                                                               \
            const Ti target_bstr,                                                               \
            const double coeff,                                                                 \
            double *vec)                                                                        \
        {                                                                                       \
            set_det_coeff<Ti, double>(                                                          \
                static_cast<const BasisManager<Ti> *>(basis_ptr),                               \
                target_astr, target_bstr, coeff, vec);                                          \
        }                                                                                       \
                                                                                                \
        void set_det_coeff_c64##SUFFIX(                                                         \
            void *__restrict__ basis_ptr,                                                       \
            const Ti target_astr,                                                               \
            const Ti target_bstr,                                                               \
            const complexf64 coeff,                                                             \
            complexf64 *vec)                                                                    \
        {                                                                                       \
            set_det_coeff<Ti, complexf64>(                                                      \
                static_cast<const BasisManager<Ti> *>(basis_ptr),                               \
                target_astr, target_bstr, coeff, vec);                                          \
        }                                                                                       \
                                                                                                \
        void *create_custom_basis_manager##SUFFIX(                                              \
            int64 norb,                                                                         \
            const Ti *input_astrs, int64 num_astrs,                                             \
            const Ti *input_bstrs, int64 num_bstrs,                                             \
            const int64 *orbsym, int64 total_sym, int64 num_irreps)                             \
        {                                                                                       \
            return create_custom_basis_manager_tmpl<Ti>(                                        \
                norb, input_astrs, num_astrs, input_bstrs, num_bstrs,                           \
                orbsym, total_sym, num_irreps);                                                 \
        }                                                                                       \
                                                                                                \
        void *create_partitioned_basis_manager##SUFFIX(                                         \
            const int64 norb, const int64 na, const int64 nb,                                   \
            const int64 physical_total_sym,                                                     \
            const int64 *__restrict__ physical_orbsym,                                          \
            const int64 *__restrict__ virtual_orbsym,                                           \
            const int64 physical_num_irreps, const int64 virtual_num_irreps)                    \
        {                                                                                       \
            return create_partitioned_basis_manager_tmpl<Ti>(                                   \
                norb, na, nb, physical_total_sym,                                               \
                physical_orbsym, virtual_orbsym,                                                \
                physical_num_irreps, virtual_num_irreps);                                       \
        }                                                                                       \
    }
