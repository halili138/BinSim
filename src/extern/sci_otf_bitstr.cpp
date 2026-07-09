#include "sci_basis.hpp"
#include "sci_select.hpp"
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

    void destroy_sci_basis_manager_bitstr_f64(void *ptr) { destroy_sci_basis_manager<uint32>(static_cast<SciBasisManager<uint32> *>(ptr)); }
}
