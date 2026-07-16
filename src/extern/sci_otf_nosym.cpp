#include "sci_basis_nosym.hpp"
#include "sci_select_nosym.hpp"
#include "otf.hpp"

extern "C"
{
    void *build_network_otf_sci_bitstr_f64(
        const int64 *orbsym, int64 norb, int64 ngs,
        const uint32 *axs, const uint32 *bxs,
        const int64 *ranks, const int64 *num_zas, const int64 *num_zbs,
        const uint32 *flat_zas, const uint32 *flat_zbs,
        const double *flat_wa, const double *flat_wb)
    {
        return build_network_otf<uint32, double>(
            orbsym, norb, ngs, axs, bxs, ranks, num_zas, num_zbs,
            flat_zas, flat_zbs, flat_wa, flat_wb);
    }

    void *create_sci_basis_manager_nosym_f64(
        const uint32 *astrs, int64 na,
        const uint32 *bstrs, int64 nb,
        int64 norb)
    {
        return create_sci_basis_manager_nosym<uint32>(astrs, na, bstrs, nb, norb);
    }

    void remap_wavefunction_sci_nosym_f64(
        void *old_ptr, const double *old_psi,
        void *new_ptr, double *new_psi)
    {
        auto *old = static_cast<SciBasisManagerNosym<uint32> *>(old_ptr);
        auto *nw  = static_cast<SciBasisManagerNosym<uint32> *>(new_ptr);
        remap_wavefunction_nosym<uint32, double>(old, old_psi, nw, new_psi);
    }

    void get_diags_elements_sci_nosym_f64(void *basis, void *net, double *diags)
    {
        get_diags_elements_sci_nosym<uint32, double>(
            static_cast<SciBasisManagerNosym<uint32> *>(basis),
            static_cast<Network_OTF<uint32, double> *>(net), diags);
    }

    void hvec_sci_nosym_f64(void *basis, void *net, const double *src, double *dst)
    {
        auto *bs = static_cast<SciBasisManagerNosym<uint32> *>(basis);
        auto *otf = static_cast<Network_OTF<uint32, double> *>(net);
        contract_hvec_sci_nosym<uint32, double>(bs, otf, src, dst);
    }

    int64 sci_basis_nosym_dim_f64(void *ptr)
    {
        return static_cast<SciBasisManagerNosym<uint32> *>(ptr)->dim;
    }

    void sci_select_nosym_f64(
        const uint32_t *new_a, int64 n_new_a,
        const uint32_t *new_b, int64 n_new_b,
        const uint32_t *old_a, int64 n_old_a,
        const uint32_t *old_b, int64 n_old_b,
        void *src_basis,
        void *net,
        double *src_psi, double E_var, double eps,
        uint32_t **out_a, uint32_t **out_b, int64 *n_pairs)
    {
        auto *basis = static_cast<SciBasisManagerNosym<uint32> *>(src_basis);
        auto *otf = static_cast<Network_OTF<uint32, double> *>(net);

        auto all_groups = flatten_groups<uint32, double>(otf);

        auto old_a_idx = build_idx_map_nosym<uint32, true>(basis);
        auto old_b_idx = build_idx_map_nosym<uint32, false>(basis);

        const auto &dg = otf->diag_groups;
        int diag_rank = 0;
        std::vector<double> pa_d, pb_do, pb_dn, pa_do;

        if (!dg.empty() && dg[0].rank > 0) {
            diag_rank = dg[0].rank;
            const auto &g = dg[0];
            auto alloc = [&](int64 n) { return std::vector<double>((size_t)(n * diag_rank), 0.0); };
            pa_d  = alloc(n_new_a); pb_do = alloc(n_old_b);
            pb_dn = alloc(n_new_b); pa_do = alloc(n_old_a);
            precompute_diag_phases_nosym<uint32,double,true> (new_a,n_new_a,g.unique_zas,g.wa,g.num_za,g.rank,pa_d.data());
            precompute_diag_phases_nosym<uint32,double,false>(old_b,n_old_b,g.unique_zbs,g.wb,g.num_zb,g.rank,pb_do.data());
            precompute_diag_phases_nosym<uint32,double,false>(new_b,n_new_b,g.unique_zbs,g.wb,g.num_zb,g.rank,pb_dn.data());
            precompute_diag_phases_nosym<uint32,double,true> (old_a,n_old_a,g.unique_zas,g.wa,g.num_za,g.rank,pa_do.data());
        }

        std::vector<std::pair<uint32_t, uint32_t>> p1, p2, p3;
        select_pass_a_nosym<uint32, double>(
            new_a, n_new_a, old_b, n_old_b, new_b, n_new_b,
            old_a_idx, old_b_idx,
            basis->num_b,
            all_groups, src_psi,
            pa_d.data(), pb_do.data(), pb_dn.data(),
            diag_rank,
            E_var, eps, p1, p3);
        select_pass_b_nosym<uint32, double>(
            new_b, n_new_b, old_a, n_old_a,
            old_b_idx, old_a_idx,
            basis->num_b,
            all_groups, src_psi,
            pb_dn.data(), pa_do.data(),
            diag_rank,
            E_var, eps, p2);

        *n_pairs = (int64)(p1.size() + p2.size() + p3.size());
        if (*n_pairs == 0) { *out_a = nullptr; *out_b = nullptr; return; }

        *out_a = (uint32_t *)malloc((size_t)(*n_pairs) * sizeof(uint32_t));
        *out_b = (uint32_t *)malloc((size_t)(*n_pairs) * sizeof(uint32_t));

        int64 idx = 0;
        for (const auto &[a,b] : p1) { (*out_a)[idx]=a; (*out_b)[idx]=b; ++idx; }
        for (const auto &[a,b] : p2) { (*out_a)[idx]=a; (*out_b)[idx]=b; ++idx; }
        for (const auto &[a,b] : p3) { (*out_a)[idx]=a; (*out_b)[idx]=b; ++idx; }
    }

    void destroy_sci_basis_manager_nosym_f64(void *ptr)
    {
        destroy_sci_basis_manager_nosym<uint32>(
            static_cast<SciBasisManagerNosym<uint32> *>(ptr));
    }
}
