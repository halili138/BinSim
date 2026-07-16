#include "sci_basis.hpp"
#include "sci_select.hpp"
#include "sci_select_test.hpp"
#include "sci3_select.hpp"
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

    void *create_sci_basis_manager_bitstr_f64(
        const uint32 *astrs, int64 na,
        const uint32 *bstrs, int64 nb,
        int64 norb, const int64 *orbsym,
        int64 total_sym, int64 num_irreps)
    {
        return static_cast<void *>(
            create_sci_basis_manager<uint32>(
                astrs, na, bstrs, nb,
                norb, orbsym, total_sym, num_irreps));
    }

    int64 sci_hvec_select_external_bitstr_f64(
        void *tgt, void *src, void *net,
        const bool *is_new_a, const bool *is_new_b,
        int64 blk, const double *src_vec, const double *candidate_diags,
        double variational_energy, int chunk_size, double eps,
        uint32 *out_a, uint32 *out_b, double *out_v, int64 max_entries)
    {
        auto *a = static_cast<SciBasisManager<uint32> *>(tgt);
        auto *b = static_cast<SciBasisManager<uint32> *>(src);
        auto *c = static_cast<Network_OTF<uint32, double> *>(net);
        auto *entries = new BufferedEntry<uint32, double>[max_entries];
        int64 n = sci_select_external_block<uint32, double>(
            a, b, c, is_new_a, is_new_b, blk, src_vec, candidate_diags,
            variational_energy, chunk_size, eps, entries, max_entries);
        for (int64 i = 0; i < n; ++i)
        {
            out_a[i] = entries[i].astr;
            out_b[i] = entries[i].bstr;
            out_v[i] = entries[i].val;
        }
        delete[] entries;
        return n;
    }

    void sci_select_instant_bitstr_f64(
        void *tgt, void *src, void *net,
        const bool *is_new_a, const bool *is_new_b,
        int64 blk, const double *src_vec,
        double variational_energy, int a_chunk_size, int b_chunk_size, double eps,
        bool *selected_a, bool *selected_b)
    {
        auto *a = static_cast<SciBasisManager<uint32> *>(tgt);
        auto *b = static_cast<SciBasisManager<uint32> *>(src);
        auto *c = static_cast<Network_OTF<uint32, double> *>(net);
        sci_select_external_block<uint32, double>(
            a, b, c, is_new_a, is_new_b, blk, src_vec,
            variational_energy, a_chunk_size, b_chunk_size, eps,
            selected_a, selected_b);
    }

    void remap_wavefunction_sci_bitstr_f64(
        void *old_ptr, const double *old_psi, void *new_ptr, double *new_psi,
        const uint32 *new_a, const uint32 *new_b, const double *new_v, int64 num_new)
    {
        auto *old = static_cast<SciBasisManager<uint32> *>(old_ptr);
        auto *nw = static_cast<SciBasisManager<uint32> *>(new_ptr);
        std::vector<BufferedEntry<uint32, double>> entries;
        entries.reserve(num_new);
        for (int64 i = 0; i < num_new; ++i)
            entries.push_back({new_a[i], new_b[i], new_v[i]});
        remap_wavefunction<uint32, double>(old, old_psi, nw, new_psi, num_new > 0 ? &entries : nullptr);
    }

    void get_diags_elements_sci_bitstr_f64(void *basis, void *net, double *diags)
    {
        get_diags_elements_sci<uint32, double>(
            static_cast<SciBasisManager<uint32> *>(basis),
            static_cast<Network_OTF<uint32, double> *>(net), diags);
    }

    void hvec_sci_full_bitstr_f64(void *basis, void *net, const double *src, double *dst)
    {
        auto *bs = static_cast<SciBasisManager<uint32> *>(basis);
        auto *n = static_cast<Network_OTF<uint32, double> *>(net);
        std::fill_n(dst, bs->dim, 0.0);
        for (int64 blk = 0; blk < bs->num_blocks; ++blk)
            contract_hvec_sci<uint32, double>(
                bs->blocks[blk], bs, n, src, dst + bs->blocks[blk].offset);
    }

    int64 sci_basis_dim_bitstr(void *ptr) { return static_cast<SciBasisManager<uint32> *>(ptr)->dim; }

    int64 sci_basis_num_blocks_bitstr(void *ptr) { return static_cast<SciBasisManager<uint32> *>(ptr)->num_blocks; }

    void sci3_select_bitstr_f64(
        const uint32_t *new_a, int64 n_new_a,
        const uint32_t *new_b, int64 n_new_b,
        const uint32_t *old_a, int64 n_old_a,
        const uint32_t *old_b, int64 n_old_b,
        void *src_basis,
        void *net,
        double *src_psi, double E_var, double eps,
        uint32_t **out_a, uint32_t **out_b, int64 *n_pairs)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(src_basis);
        auto *otf = static_cast<Network_OTF<uint32, double> *>(net);

        auto all_groups = flatten_groups<uint32, double>(otf);

        auto old_a_idx = build_idx_map<uint32, true>(basis);
        auto old_b_idx = build_idx_map<uint32, false>(basis);

        std::vector<std::pair<uint32_t, uint32_t>> p1, p2, p3;
        select_pass_a<uint32, double>(
            new_a, n_new_a, old_b, n_old_b, new_b, n_new_b,
            old_a_idx, old_b_idx,
            basis->num_blocks,
            all_groups, src_psi, basis->blocks,
            E_var, eps, p1, p3);
        select_pass_b<uint32, double>(
            new_b, n_new_b, old_a, n_old_a,
            old_b_idx, old_a_idx,
            basis->num_blocks,
            all_groups, src_psi, basis->blocks,
            E_var, eps, p2);

        *n_pairs = (int64)(p1.size() + p2.size() + p3.size());
        if (*n_pairs == 0)
        {
            *out_a = nullptr;
            *out_b = nullptr;
            return;
        }

        *out_a = (uint32_t *)malloc((size_t)(*n_pairs) * sizeof(uint32_t));
        *out_b = (uint32_t *)malloc((size_t)(*n_pairs) * sizeof(uint32_t));

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

    void destroy_sci_basis_manager_bitstr_f64(void *ptr)
    {
        destroy_sci_basis_manager<uint32>(static_cast<SciBasisManager<uint32> *>(ptr));
    }
}
