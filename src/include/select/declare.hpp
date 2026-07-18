#pragma once
#include <cstdlib>
#include "core/bit.hpp"
#include "select/forward.hpp"
#include "select/nosym.hpp"

#define DECLARE_SCI_INTERFACES(Ti, SUFFIX)                                                                \
    extern "C"                                                                                            \
    {                                                                                                     \
        void *build_network_otf_sci_bitstr_##SUFFIX(                                                      \
            const int64 *orbsym, int64 norb, int64 ngs,                                                   \
            const Ti *axs, const Ti *bxs,                                                                 \
            const int64 *ranks, const int64 *num_zas, const int64 *num_zbs,                               \
            const Ti *flat_zas, const Ti *flat_zbs,                                                       \
            const double *flat_wa, const double *flat_wb)                                                 \
        {                                                                                                 \
            return build_network_otf<Ti, double>(                                                         \
                orbsym, norb, ngs, axs, bxs, ranks, num_zas, num_zbs,                                     \
                flat_zas, flat_zbs, flat_wa, flat_wb);                                                    \
        }                                                                                                 \
                                                                                                          \
        void *create_sci_basis_manager_bitstr_##SUFFIX(                                                   \
            const Ti *astrs, int64 na,                                                                    \
            const Ti *bstrs, int64 nb,                                                                    \
            int64 norb, const int64 *orbsym,                                                              \
            int64 total_sym, int64 num_irreps)                                                            \
        {                                                                                                 \
            return create_sci_basis_manager<Ti>(                                                          \
                astrs, na, bstrs, nb, norb, orbsym, total_sym, num_irreps);                               \
        }                                                                                                 \
                                                                                                          \
        void remap_wavefunction_sci_bitstr_##SUFFIX(                                                      \
            void *old_ptr, const double *old_psi, void *new_ptr, double *new_psi,                         \
            const Ti *new_a, const Ti *new_b, const double *new_v, int64 num_new)                         \
        {                                                                                                 \
            auto *old = static_cast<SciBasisManager<Ti> *>(old_ptr);                                      \
            auto *nw = static_cast<SciBasisManager<Ti> *>(new_ptr);                                       \
            std::vector<BufferedEntry<Ti, double>> entries;                                               \
            entries.reserve(num_new);                                                                     \
            for (int64 i = 0; i < num_new; ++i)                                                           \
                entries.push_back({new_a[i], new_b[i], new_v[i]});                                        \
            remap_wavefunction<Ti, double>(old, old_psi, nw, new_psi, num_new > 0 ? &entries : nullptr);  \
        }                                                                                                 \
                                                                                                          \
        void get_diags_elements_sci_bitstr_##SUFFIX(                                                      \
            void *basis, void *net, double *diags)                                                        \
        {                                                                                                 \
            get_diags_elements_sci<Ti, double>(                                                           \
                static_cast<SciBasisManager<Ti> *>(basis),                                                \
                static_cast<Network_OTF<Ti, double> *>(net),                                              \
                diags);                                                                                   \
        }                                                                                                 \
                                                                                                          \
        void hvec_sci_full_bitstr_##SUFFIX(void *basis, void *net, const double *src, double *dst)        \
        {                                                                                                 \
            auto *bs = static_cast<SciBasisManager<Ti> *>(basis);                                         \
            auto *n = static_cast<Network_OTF<Ti, double> *>(net);                                        \
            std::fill_n(dst, bs->dim, 0.0);                                                               \
            for (int64 blk = 0; blk < bs->num_blocks; ++blk)                                              \
                contract_hvec_sci<Ti, double>(bs->blocks[blk], bs, n, src, dst + bs->blocks[blk].offset); \
        }                                                                                                 \
                                                                                                          \
        int64 sci_basis_dim_bitstr_##SUFFIX(void *ptr)                                                    \
        {                                                                                                 \
            return static_cast<SciBasisManager<Ti> *>(ptr)->dim;                                          \
        }                                                                                                 \
                                                                                                          \
        int64 sci_basis_num_blocks_bitstr_##SUFFIX(void *ptr)                                             \
        {                                                                                                 \
            return static_cast<SciBasisManager<Ti> *>(ptr)->num_blocks;                                   \
        }                                                                                                 \
                                                                                                          \
        void sci_select_bitstr_##SUFFIX(                                                                  \
            const Ti *new_a, int64 n_new_a,                                                               \
            const Ti *new_b, int64 n_new_b,                                                               \
            const Ti *old_a, int64 n_old_a,                                                               \
            const Ti *old_b, int64 n_old_b,                                                               \
            void *src_basis,                                                                              \
            void *net,                                                                                    \
            double *src_psi, double E_var, double eps,                                                    \
            Ti **out_a, Ti **out_b, int64 *n_pairs)                                                       \
        {                                                                                                 \
            sci_select_bitstr_impl<Ti, double>(                                                           \
                new_a, n_new_a, new_b, n_new_b,                                                           \
                old_a, n_old_a, old_b, n_old_b,                                                           \
                src_basis, net, src_psi, E_var, eps,                                                      \
                out_a, out_b, n_pairs);                                                                   \
        }                                                                                                 \
                                                                                                          \
        void destroy_sci_basis_manager_bitstr_##SUFFIX(void *ptr)                                         \
        {                                                                                                 \
            destroy_sci_basis_manager<Ti>(static_cast<SciBasisManager<Ti> *>(ptr));                       \
        }                                                                                                 \
    }

#define DECLARE_SCI_NOSYM_INTERFACES(Ti, SUFFIX)                                               \
    extern "C"                                                                                 \
    {                                                                                          \
        void *create_sci_basis_manager_nosym_##SUFFIX(                                         \
            const Ti *astrs, int64 na,                                                         \
            const Ti *bstrs, int64 nb,                                                         \
            int64 norb)                                                                        \
        {                                                                                      \
            return create_sci_basis_manager_nosym<Ti>(                                         \
                astrs, na, bstrs, nb, norb);                                                   \
        }                                                                                      \
                                                                                               \
        void remap_wavefunction_sci_nosym_##SUFFIX(                                            \
            void *old_ptr, const double *old_psi,                                              \
            void *new_ptr, double *new_psi)                                                    \
        {                                                                                      \
            auto *old = static_cast<SciBasisManagerNosym<Ti> *>(old_ptr);                      \
            auto *nw = static_cast<SciBasisManagerNosym<Ti> *>(new_ptr);                       \
            remap_wavefunction_nosym<Ti, double>(old, old_psi, nw, new_psi);                   \
        }                                                                                      \
                                                                                               \
        void get_diags_elements_sci_nosym_##SUFFIX(                                            \
            void *basis, void *net, double *diags)                                             \
        {                                                                                      \
            get_diags_elements_sci_nosym<Ti, double>(                                          \
                static_cast<SciBasisManagerNosym<Ti> *>(basis),                                \
                static_cast<Network_OTF<Ti, double> *>(net),                                   \
                diags);                                                                        \
        }                                                                                      \
                                                                                               \
        void hvec_sci_nosym_##SUFFIX(void *basis, void *net, const double *src, double *dst)   \
        {                                                                                      \
            contract_hvec_sci_nosym<Ti, double>(                                               \
                static_cast<SciBasisManagerNosym<Ti> *>(basis),                                \
                static_cast<Network_OTF<Ti, double> *>(net),                                   \
                src, dst);                                                                     \
        }                                                                                      \
                                                                                               \
        int64 sci_basis_nosym_dim_##SUFFIX(void *ptr)                                          \
        {                                                                                      \
            return static_cast<SciBasisManagerNosym<Ti> *>(ptr)->dim;                          \
        }                                                                                      \
                                                                                               \
        void sci_select_nosym_##SUFFIX(                                                        \
            const Ti *new_a, int64 n_new_a,                                                    \
            const Ti *new_b, int64 n_new_b,                                                    \
            const Ti *old_a, int64 n_old_a,                                                    \
            const Ti *old_b, int64 n_old_b,                                                    \
            void *src_basis,                                                                   \
            void *net,                                                                         \
            double *src_psi, double E_var, double eps,                                         \
            Ti **out_a, Ti **out_b, int64 *n_pairs)                                            \
        {                                                                                      \
            sci_select_nosym_impl<Ti, double>(                                                 \
                new_a, n_new_a, new_b, n_new_b,                                                \
                old_a, n_old_a, old_b, n_old_b,                                                \
                src_basis, net, src_psi, E_var, eps,                                           \
                out_a, out_b, n_pairs);                                                        \
        }                                                                                      \
                                                                                               \
        void destroy_sci_basis_manager_nosym_##SUFFIX(void *ptr)                               \
        {                                                                                      \
            destroy_sci_basis_manager_nosym<Ti>(static_cast<SciBasisManagerNosym<Ti> *>(ptr)); \
        }                                                                                      \
    }
