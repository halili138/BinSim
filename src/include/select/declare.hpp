#pragma once
#include <cstdlib>
#include "core/bit.hpp"
#include "basis/sci_basis.hpp"
#include "select/forward.hpp"
#include "basis/nosym_basis.hpp"
#include "select/nosym.hpp"
#include "ham/otf.hpp"

// ===== Symm (block-based) template impls =====

template <typename Ti, typename Tv>
static void *build_network_otf_sci_bitstr_impl(
    const int64 *orbsym, int64 norb, int64 ngs,
    const Ti *axs, const Ti *bxs,
    const int64 *ranks, const int64 *num_zas, const int64 *num_zbs,
    const Ti *flat_zas, const Ti *flat_zbs,
    const Tv *flat_wa, const Tv *flat_wb)
{
    return build_network_otf<Ti, Tv>(
        orbsym, norb, ngs, axs, bxs, ranks, num_zas, num_zbs,
        flat_zas, flat_zbs, flat_wa, flat_wb);
}

template <typename Ti>
static void *create_sci_basis_manager_bitstr_impl(
    const Ti *astrs, int64 na,
    const Ti *bstrs, int64 nb,
    int64 norb, const int64 *orbsym,
    int64 total_sym, int64 num_irreps)
{
    return static_cast<void *>(
        create_sci_basis_manager<Ti>(astrs, na, bstrs, nb, norb,
                                     orbsym, total_sym, num_irreps));
}

template <typename Ti, typename Tv>
static void remap_wavefunction_sci_bitstr_impl(
    void *old_ptr, const Tv *old_psi, void *new_ptr, Tv *new_psi,
    const Ti *new_a, const Ti *new_b, const Tv *new_v, int64 num_new)
{
    auto *old = static_cast<SciBasisManager<Ti> *>(old_ptr);
    auto *nw = static_cast<SciBasisManager<Ti> *>(new_ptr);
    std::vector<BufferedEntry<Ti, Tv>> entries;
    entries.reserve(num_new);
    for (int64 i = 0; i < num_new; ++i)
        entries.push_back({new_a[i], new_b[i], new_v[i]});
    remap_wavefunction<Ti, Tv>(old, old_psi, nw, new_psi, num_new > 0 ? &entries : nullptr);
}

template <typename Ti, typename Tv>
static void get_diags_elements_sci_bitstr_impl(void *basis, void *net, Tv *diags)
{
    auto *bs = static_cast<SciBasisManager<Ti> *>(basis);
    auto *n = static_cast<Network_OTF<Ti, Tv> *>(net);
    get_diags_elements_sci<Ti, Tv>(bs, n, diags);
}

template <typename Ti, typename Tv>
static void hvec_sci_full_bitstr_impl(void *basis, void *net, const Tv *src, Tv *dst)
{
    auto *bs = static_cast<SciBasisManager<Ti> *>(basis);
    auto *n = static_cast<Network_OTF<Ti, Tv> *>(net);
    std::fill_n(dst, bs->dim, 0.0);
    for (int64 blk = 0; blk < bs->num_blocks; ++blk)
        contract_hvec_sci<Ti, Tv>(bs->blocks[blk], bs, n, src, dst + bs->blocks[blk].offset);
}

template <typename Ti>
static int64 sci_basis_dim_bitstr_impl(void *ptr)
{
    return static_cast<SciBasisManager<Ti> *>(ptr)->dim;
}

template <typename Ti>
static int64 sci_basis_num_blocks_bitstr_impl(void *ptr)
{
    return static_cast<SciBasisManager<Ti> *>(ptr)->num_blocks;
}

template <typename Ti, typename Tv>
static void sci_select_bitstr_impl(
    const Ti *new_a, int64 n_new_a,
    const Ti *new_b, int64 n_new_b,
    const Ti *old_a, int64 n_old_a,
    const Ti *old_b, int64 n_old_b,
    void *src_basis,
    void *net,
    Tv *src_psi, Tv E_var, Tv eps,
    Ti **out_a, Ti **out_b, int64 *n_pairs)
{
    auto *basis = static_cast<SciBasisManager<Ti> *>(src_basis);
    auto *otf = static_cast<Network_OTF<Ti, Tv> *>(net);

    auto all_groups = flatten_groups<Ti, Tv>(otf);

    auto old_a_idx = build_idx_map<Ti, true>(basis);
    auto old_b_idx = build_idx_map<Ti, false>(basis);

    const auto &dg = otf->diag_groups;
    int diag_rank = 0;
    std::vector<Tv> pa_d, pb_do, pb_dn, pa_do;

    if (!dg.empty() && dg[0].rank > 0)
    {
        diag_rank = dg[0].rank;
        const auto &g = dg[0];

        auto alloc_diag = [&](int64 n)
        {
            return std::vector<Tv>((size_t)(n * diag_rank), Tv{});
        };

        pa_d = alloc_diag(n_new_a);
        pb_do = alloc_diag(n_old_b);
        pb_dn = alloc_diag(n_new_b);
        pa_do = alloc_diag(n_old_a);

        precompute_diag_phases<Ti, Tv>(new_a, n_new_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_d.data());
        precompute_diag_phases<Ti, Tv>(old_b, n_old_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_do.data());
        precompute_diag_phases<Ti, Tv>(new_b, n_new_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_dn.data());
        precompute_diag_phases<Ti, Tv>(old_a, n_old_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_do.data());
    }

    std::vector<std::pair<Ti, Ti>> p1, p2, p3;
    select_pass_a<Ti, Tv>(
        new_a, n_new_a, old_b, n_old_b, new_b, n_new_b,
        old_a_idx, old_b_idx,
        basis->num_blocks,
        all_groups, src_psi, basis->blocks,
        pa_d.data(), pb_do.data(), pb_dn.data(),
        diag_rank,
        E_var, eps, p1, p3);
    select_pass_b<Ti, Tv>(
        new_b, n_new_b, old_a, n_old_a,
        old_b_idx, old_a_idx,
        basis->num_blocks,
        all_groups, src_psi, basis->blocks,
        pb_dn.data(), pa_do.data(),
        diag_rank,
        E_var, eps, p2);

    *n_pairs = (int64)(p1.size() + p2.size() + p3.size());
    if (*n_pairs == 0)
    {
        *out_a = nullptr;
        *out_b = nullptr;
        return;
    }

    *out_a = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));
    *out_b = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));

    int64 idx = 0;
    for (const auto &[a, b] : p1)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p2)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p3)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
}

template <typename Ti>
static void destroy_sci_basis_manager_bitstr_impl(void *ptr)
{
    destroy_sci_basis_manager<Ti>(static_cast<SciBasisManager<Ti> *>(ptr));
}

// ===== Nosym (flat) template impls =====

template <typename Ti>
static void *create_sci_basis_manager_nosym_impl(
    const Ti *astrs, int64 na,
    const Ti *bstrs, int64 nb,
    int64 norb)
{
    return create_sci_basis_manager_nosym<Ti>(astrs, na, bstrs, nb, norb);
}

template <typename Ti, typename Tv>
static void remap_wavefunction_sci_nosym_impl(
    void *old_ptr, const Tv *old_psi,
    void *new_ptr, Tv *new_psi)
{
    auto *old = static_cast<SciBasisManagerNosym<Ti> *>(old_ptr);
    auto *nw = static_cast<SciBasisManagerNosym<Ti> *>(new_ptr);
    remap_wavefunction_nosym<Ti, Tv>(old, old_psi, nw, new_psi);
}

template <typename Ti, typename Tv>
static void get_diags_elements_sci_nosym_impl(void *basis, void *net, Tv *diags)
{
    auto *bs = static_cast<SciBasisManagerNosym<Ti> *>(basis);
    auto *otf = static_cast<Network_OTF<Ti, Tv> *>(net);
    get_diags_elements_sci_nosym<Ti, Tv>(bs, otf, diags);
}

template <typename Ti, typename Tv>
static void hvec_sci_nosym_impl(void *basis, void *net, const Tv *src, Tv *dst)
{
    auto *bs = static_cast<SciBasisManagerNosym<Ti> *>(basis);
    auto *otf = static_cast<Network_OTF<Ti, Tv> *>(net);
    contract_hvec_sci_nosym<Ti, Tv>(bs, otf, src, dst);
}

template <typename Ti>
static int64 sci_basis_nosym_dim_impl(void *ptr)
{
    return static_cast<SciBasisManagerNosym<Ti> *>(ptr)->dim;
}

template <typename Ti, typename Tv>
static void sci_select_nosym_impl(
    const Ti *new_a, int64 n_new_a,
    const Ti *new_b, int64 n_new_b,
    const Ti *old_a, int64 n_old_a,
    const Ti *old_b, int64 n_old_b,
    void *src_basis,
    void *net,
    Tv *src_psi, Tv E_var, Tv eps,
    Ti **out_a, Ti **out_b, int64 *n_pairs)
{
    auto *basis = static_cast<SciBasisManagerNosym<Ti> *>(src_basis);
    auto *otf = static_cast<Network_OTF<Ti, Tv> *>(net);

    auto all_groups = flatten_groups<Ti, Tv>(otf);

    auto old_a_idx = build_idx_map_nosym<Ti, true>(basis);
    auto old_b_idx = build_idx_map_nosym<Ti, false>(basis);

    const auto &dg = otf->diag_groups;
    int diag_rank = 0;
    std::vector<Tv> pa_d, pb_do, pb_dn, pa_do;

    if (!dg.empty() && dg[0].rank > 0)
    {
        diag_rank = dg[0].rank;
        const auto &g = dg[0];
        auto alloc = [&](int64 n)
        {
            return std::vector<Tv>((size_t)(n * diag_rank), Tv{});
        };

        pa_d = alloc(n_new_a);
        pb_do = alloc(n_old_b);
        pb_dn = alloc(n_new_b);
        pa_do = alloc(n_old_a);

        precompute_diag_phases<Ti, Tv>(new_a, n_new_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_d.data());
        precompute_diag_phases<Ti, Tv>(old_b, n_old_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_do.data());
        precompute_diag_phases<Ti, Tv>(new_b, n_new_b, g.unique_zbs, g.wb, g.num_zb, g.rank, pb_dn.data());
        precompute_diag_phases<Ti, Tv>(old_a, n_old_a, g.unique_zas, g.wa, g.num_za, g.rank, pa_do.data());
    }

    std::vector<std::pair<Ti, Ti>> p1, p2, p3;
    select_pass_a_nosym<Ti, Tv>(
        new_a, n_new_a, old_b, n_old_b, new_b, n_new_b,
        old_a_idx, old_b_idx,
        basis->num_b,
        all_groups, src_psi,
        pa_d.data(), pb_do.data(), pb_dn.data(),
        diag_rank,
        E_var, eps, p1, p3);
    select_pass_b_nosym<Ti, Tv>(
        new_b, n_new_b, old_a, n_old_a,
        old_b_idx, old_a_idx,
        basis->num_b,
        all_groups, src_psi,
        pb_dn.data(), pa_do.data(),
        diag_rank,
        E_var, eps, p2);

    *n_pairs = (int64)(p1.size() + p2.size() + p3.size());
    if (*n_pairs == 0)
    {
        *out_a = nullptr;
        *out_b = nullptr;
        return;
    }

    *out_a = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));
    *out_b = (Ti *)malloc((size_t)(*n_pairs) * sizeof(Ti));

    int64 idx = 0;
    for (const auto &[a, b] : p1)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p2)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
    for (const auto &[a, b] : p3)
    {
        (*out_a)[idx] = a;
        (*out_b)[idx] = b;
        ++idx;
    }
}

template <typename Ti>
static void destroy_sci_basis_manager_nosym_impl(void *ptr)
{
    destroy_sci_basis_manager_nosym<Ti>(static_cast<SciBasisManagerNosym<Ti> *>(ptr));
}

// ===== Symm extern-C macro =====

#define DECLARE_SCI_INTERFACES(Ti, SUFFIX)                                                                           \
    extern "C"                                                                                                       \
    {                                                                                                                \
        void *build_network_otf_sci_bitstr_##SUFFIX(                                                                 \
            const int64 *orbsym, int64 norb, int64 ngs,                                                              \
            const Ti *axs, const Ti *bxs,                                                                            \
            const int64 *ranks, const int64 *num_zas, const int64 *num_zbs,                                           \
            const Ti *flat_zas, const Ti *flat_zbs,                                                                  \
            const double *flat_wa, const double *flat_wb)                                                            \
        {                                                                                                            \
            return build_network_otf_sci_bitstr_impl<Ti, double>(                                                    \
                orbsym, norb, ngs, axs, bxs, ranks, num_zas, num_zbs,                                                \
                flat_zas, flat_zbs, flat_wa, flat_wb);                                                               \
        }                                                                                                            \
                                                                                                                     \
        void *create_sci_basis_manager_bitstr_##SUFFIX(                                                              \
            const Ti *astrs, int64 na,                                                                               \
            const Ti *bstrs, int64 nb,                                                                               \
            int64 norb, const int64 *orbsym,                                                                         \
            int64 total_sym, int64 num_irreps)                                                                       \
        {                                                                                                            \
            return create_sci_basis_manager_bitstr_impl<Ti>(                                                          \
                astrs, na, bstrs, nb, norb, orbsym, total_sym, num_irreps);                                         \
        }                                                                                                            \
                                                                                                                     \
        void remap_wavefunction_sci_bitstr_##SUFFIX(                                                                 \
            void *old_ptr, const double *old_psi, void *new_ptr, double *new_psi,                                     \
            const Ti *new_a, const Ti *new_b, const double *new_v, int64 num_new)                                   \
        {                                                                                                            \
            remap_wavefunction_sci_bitstr_impl<Ti, double>(                                                           \
                old_ptr, old_psi, new_ptr, new_psi, new_a, new_b, new_v, num_new);                                  \
        }                                                                                                            \
                                                                                                                     \
        void get_diags_elements_sci_bitstr_##SUFFIX(                                                                 \
            void *basis, void *net, double *diags)                                                                   \
        {                                                                                                            \
            get_diags_elements_sci_bitstr_impl<Ti, double>(basis, net, diags);                                       \
        }                                                                                                            \
                                                                                                                     \
        void hvec_sci_full_bitstr_##SUFFIX(void *basis, void *net, const double *src, double *dst)                    \
        {                                                                                                            \
            hvec_sci_full_bitstr_impl<Ti, double>(basis, net, src, dst);                                             \
        }                                                                                                            \
                                                                                                                     \
        int64 sci_basis_dim_bitstr_##SUFFIX(void *ptr)                                                               \
        {                                                                                                            \
            return sci_basis_dim_bitstr_impl<Ti>(ptr);                                                                \
        }                                                                                                            \
                                                                                                                     \
        int64 sci_basis_num_blocks_bitstr_##SUFFIX(void *ptr)                                                        \
        {                                                                                                            \
            return sci_basis_num_blocks_bitstr_impl<Ti>(ptr);                                                         \
        }                                                                                                            \
                                                                                                                     \
        void sci_select_bitstr_##SUFFIX(                                                                              \
            const Ti *new_a, int64 n_new_a,                                                                          \
            const Ti *new_b, int64 n_new_b,                                                                          \
            const Ti *old_a, int64 n_old_a,                                                                          \
            const Ti *old_b, int64 n_old_b,                                                                          \
            void *src_basis,                                                                                         \
            void *net,                                                                                               \
            double *src_psi, double E_var, double eps,                                                               \
            Ti **out_a, Ti **out_b, int64 *n_pairs)                                                                  \
        {                                                                                                            \
            sci_select_bitstr_impl<Ti, double>(                                                                       \
                new_a, n_new_a, new_b, n_new_b,                                                                      \
                old_a, n_old_a, old_b, n_old_b,                                                                      \
                src_basis, net, src_psi, E_var, eps,                                                                 \
                out_a, out_b, n_pairs);                                                                              \
        }                                                                                                            \
                                                                                                                     \
        void destroy_sci_basis_manager_bitstr_##SUFFIX(void *ptr)                                                     \
        {                                                                                                            \
            destroy_sci_basis_manager_bitstr_impl<Ti>(ptr);                                                           \
        }                                                                                                            \
    }

// ===== Nosym extern-C macro =====

#define DECLARE_SCI_NOSYM_INTERFACES(Ti, SUFFIX)                                                                     \
    extern "C"                                                                                                       \
    {                                                                                                                \
        void *create_sci_basis_manager_nosym_##SUFFIX(                                                               \
            const Ti *astrs, int64 na,                                                                               \
            const Ti *bstrs, int64 nb,                                                                               \
            int64 norb)                                                                                              \
        {                                                                                                            \
            return create_sci_basis_manager_nosym_impl<Ti>(                                                           \
                astrs, na, bstrs, nb, norb);                                                                         \
        }                                                                                                            \
                                                                                                                     \
        void remap_wavefunction_sci_nosym_##SUFFIX(                                                                  \
            void *old_ptr, const double *old_psi,                                                                    \
            void *new_ptr, double *new_psi)                                                                          \
        {                                                                                                            \
            remap_wavefunction_sci_nosym_impl<Ti, double>(                                                            \
                old_ptr, old_psi, new_ptr, new_psi);                                                                 \
        }                                                                                                            \
                                                                                                                     \
        void get_diags_elements_sci_nosym_##SUFFIX(                                                                  \
            void *basis, void *net, double *diags)                                                                   \
        {                                                                                                            \
            get_diags_elements_sci_nosym_impl<Ti, double>(basis, net, diags);                                        \
        }                                                                                                            \
                                                                                                                     \
        void hvec_sci_nosym_##SUFFIX(void *basis, void *net, const double *src, double *dst)                          \
        {                                                                                                            \
            hvec_sci_nosym_impl<Ti, double>(basis, net, src, dst);                                                   \
        }                                                                                                            \
                                                                                                                     \
        int64 sci_basis_nosym_dim_##SUFFIX(void *ptr)                                                                \
        {                                                                                                            \
            return sci_basis_nosym_dim_impl<Ti>(ptr);                                                                 \
        }                                                                                                            \
                                                                                                                     \
        void sci_select_nosym_##SUFFIX(                                                                               \
            const Ti *new_a, int64 n_new_a,                                                                          \
            const Ti *new_b, int64 n_new_b,                                                                          \
            const Ti *old_a, int64 n_old_a,                                                                          \
            const Ti *old_b, int64 n_old_b,                                                                          \
            void *src_basis,                                                                                         \
            void *net,                                                                                               \
            double *src_psi, double E_var, double eps,                                                               \
            Ti **out_a, Ti **out_b, int64 *n_pairs)                                                                  \
        {                                                                                                            \
            sci_select_nosym_impl<Ti, double>(                                                                        \
                new_a, n_new_a, new_b, n_new_b,                                                                      \
                old_a, n_old_a, old_b, n_old_b,                                                                      \
                src_basis, net, src_psi, E_var, eps,                                                                 \
                out_a, out_b, n_pairs);                                                                              \
        }                                                                                                            \
                                                                                                                     \
        void destroy_sci_basis_manager_nosym_##SUFFIX(void *ptr)                                                      \
        {                                                                                                            \
            destroy_sci_basis_manager_nosym_impl<Ti>(ptr);                                                            \
        }                                                                                                            \
    }
