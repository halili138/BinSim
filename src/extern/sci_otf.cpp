#include "sci_basis.hpp"
#include "sci_select.hpp"
#include "sci_hvec.hpp"
#include "otf.hpp"

extern "C"
{

    void *expand_and_build_sci_basis_f64(
        void *src_basis_ptr,
        const double *src_psi,
        void *net_ptr,
        int64 norb,
        const int64 *orbsym,
        int64 num_irreps,
        int mode_code,
        int strat_code)
    {
        auto *sb = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        std::vector<SVDGroup_OTF<uint32, double>> all_groups;
        all_groups.reserve(net->num_groups);
        all_groups.insert(all_groups.end(), net->diag_groups.begin(), net->diag_groups.end());
        all_groups.insert(all_groups.end(), net->pure_a_groups.begin(), net->pure_a_groups.end());
        all_groups.insert(all_groups.end(), net->pure_b_groups.begin(), net->pure_b_groups.end());
        all_groups.insert(all_groups.end(), net->mixed_groups.begin(), net->mixed_groups.end());

        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        std::vector<uint32> dst_astrs, dst_bstrs;
        std::vector<bool> is_new_a, is_new_b;
        std::vector<std::pair<uint32, uint32>> new_pairs_strs;

        expand_bitstrings<uint32, double>(
            sb, src_psi, all_groups.data(), (int64)all_groups.size(),
            norb, orbsym, num_irreps, mode,
            dst_astrs, is_new_a, dst_bstrs, is_new_b,
            (mode == SelectMode::Pair) ? &new_pairs_strs : nullptr);

        int64 total_sym = sb->total_sym;

        return static_cast<void *>(
            create_sci_basis_manager<uint32>(
                dst_astrs.data(), (int64)dst_astrs.size(), is_new_a,
                dst_bstrs.data(), (int64)dst_bstrs.size(), is_new_b,
                norb, orbsym, total_sym, num_irreps,
                mode, strat,
                (mode == SelectMode::Pair) ? &new_pairs_strs : nullptr,
                sb));
    }

    void *expand_and_build_sci_basis_c64(
        void *src_basis_ptr,
        const complexf64 *src_psi,
        void *net_ptr,
        int64 norb,
        const int64 *orbsym,
        int64 num_irreps,
        int mode_code,
        int strat_code)
    {
        auto *sb = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        std::vector<SVDGroup_OTF<uint32, complexf64>> all_groups;
        all_groups.reserve(net->num_groups);
        all_groups.insert(all_groups.end(), net->diag_groups.begin(), net->diag_groups.end());
        all_groups.insert(all_groups.end(), net->pure_a_groups.begin(), net->pure_a_groups.end());
        all_groups.insert(all_groups.end(), net->pure_b_groups.begin(), net->pure_b_groups.end());
        all_groups.insert(all_groups.end(), net->mixed_groups.begin(), net->mixed_groups.end());

        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        std::vector<uint32> dst_astrs, dst_bstrs;
        std::vector<bool> is_new_a, is_new_b;
        std::vector<std::pair<uint32, uint32>> new_pairs_strs;

        expand_bitstrings<uint32, complexf64>(
            sb, src_psi, all_groups.data(), (int64)all_groups.size(),
            norb, orbsym, num_irreps, mode,
            dst_astrs, is_new_a, dst_bstrs, is_new_b,
            (mode == SelectMode::Pair) ? &new_pairs_strs : nullptr);

        int64 total_sym = sb->total_sym;

        return static_cast<void *>(
            create_sci_basis_manager<uint32>(
                dst_astrs.data(), (int64)dst_astrs.size(), is_new_a,
                dst_bstrs.data(), (int64)dst_bstrs.size(), is_new_b,
                norb, orbsym, total_sym, num_irreps,
                mode, strat,
                (mode == SelectMode::Pair) ? &new_pairs_strs : nullptr,
                sb));
    }

    void destroy_sci_basis_manager_f64(void *ptr)
    {
        destroy_sci_basis_manager<uint32>(
            static_cast<SciBasisManager<uint32> *>(ptr));
    }

    void destroy_sci_basis_manager_c64(void *ptr)
    {
        destroy_sci_basis_manager<uint32>(
            static_cast<SciBasisManager<uint32> *>(ptr));
    }

    int64 sci_hvec_select_for_block_f64(
        void *tgt_basis_ptr,
        void *src_basis_ptr,
        void *net_ptr,
        int64 block_idx,
        const double *src_vec,
        int chunk_size,
        double eps,
        uint32 *out_a,
        uint32 *out_b,
        double *out_v,
        int64 max_entries,
        double *out_max_abs,
        int64 *out_count_gt_eps,
        int64 *out_count_gt_1e12)
    {
        auto *tgt = static_cast<SciBasisManager<uint32> *>(tgt_basis_ptr);
        auto *src = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        auto *entries = new BufferedEntry<uint32, double>[max_entries];
        SciSelectStats stats;
        int64 count = sci_hvec_select_for_block<uint32, double>(
            tgt, src, net, block_idx, src_vec,
            chunk_size, eps, entries, max_entries, &stats);

        *out_max_abs = stats.max_abs;
        *out_count_gt_eps = stats.count_gt_eps;
        *out_count_gt_1e12 = stats.count_gt_1e12;

        for (int64 i = 0; i < count; ++i)
        {
            out_a[i] = entries[i].astr;
            out_b[i] = entries[i].bstr;
            out_v[i] = entries[i].val;
        }
        delete[] entries;
        return count;
    }

    int64 sci_hvec_select_for_block_c64(
        void *tgt_basis_ptr,
        void *src_basis_ptr,
        void *net_ptr,
        int64 block_idx,
        const complexf64 *src_vec,
        int chunk_size,
        double eps,
        uint32 *out_a,
        uint32 *out_b,
        complexf64 *out_v,
        int64 max_entries,
        double *out_max_abs,
        int64 *out_count_gt_eps,
        int64 *out_count_gt_1e12)
    {
        auto *tgt = static_cast<SciBasisManager<uint32> *>(tgt_basis_ptr);
        auto *src = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        auto *entries = new BufferedEntry<uint32, complexf64>[max_entries];
        SciSelectStats stats;
        int64 count = sci_hvec_select_for_block<uint32, complexf64>(
            tgt, src, net, block_idx, src_vec,
            chunk_size, eps, entries, max_entries, &stats);

        *out_max_abs = stats.max_abs;
        *out_count_gt_eps = stats.count_gt_eps;
        *out_count_gt_1e12 = stats.count_gt_1e12;

        for (int64 i = 0; i < count; ++i)
        {
            out_a[i] = entries[i].astr;
            out_b[i] = entries[i].bstr;
            out_v[i] = entries[i].val;
        }
        delete[] entries;
        return count;
    }

    void *create_source_basis_from_merge_f64(
        void *src_basis_ptr,
        const uint32 *sel_a,
        const uint32 *sel_b,
        const double *sel_v,
        int64 num_sel,
        int64 norb,
        const int64 *orbsym,
        int64 total_sym,
        int64 num_irreps,
        int mode_code,
        int strat_code)
    {
        auto *sb = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        std::vector<BufferedEntry<uint32, double>> selected;
        selected.reserve(num_sel);
        for (int64 i = 0; i < num_sel; ++i)
            selected.push_back({sel_a[i], sel_b[i], sel_v[i]});

        return static_cast<void *>(
            create_source_basis_from_merge<uint32, double>(
                sb, selected, norb, orbsym, total_sym, num_irreps, mode, strat));
    }

    void *create_source_basis_from_merge_c64(
        void *src_basis_ptr,
        const uint32 *sel_a,
        const uint32 *sel_b,
        const complexf64 *sel_v,
        int64 num_sel,
        int64 norb,
        const int64 *orbsym,
        int64 total_sym,
        int64 num_irreps,
        int mode_code,
        int strat_code)
    {
        auto *sb = static_cast<SciBasisManager<uint32> *>(src_basis_ptr);
        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        std::vector<BufferedEntry<uint32, complexf64>> selected;
        selected.reserve(num_sel);
        for (int64 i = 0; i < num_sel; ++i)
            selected.push_back({sel_a[i], sel_b[i], sel_v[i]});

        return static_cast<void *>(
            create_source_basis_from_merge<uint32, complexf64>(
                sb, selected, norb, orbsym, total_sym, num_irreps, mode, strat));
    }

    void remap_wavefunction_sci_f64(
        void *old_src_ptr,
        const double *old_psi,
        void *new_src_ptr,
        double *new_psi,
        const uint32 *new_a,
        const uint32 *new_b,
        const double *new_v,
        int64 num_new)
    {
        auto *old_src = static_cast<SciBasisManager<uint32> *>(old_src_ptr);
        auto *new_src = static_cast<SciBasisManager<uint32> *>(new_src_ptr);

        std::vector<BufferedEntry<uint32, double>> entries;
        entries.reserve(num_new);
        for (int64 i = 0; i < num_new; ++i)
            entries.push_back({new_a[i], new_b[i], new_v[i]});

        remap_wavefunction<uint32, double>(
            old_src, old_psi, new_src, new_psi,
            num_new > 0 ? &entries : nullptr);
    }

    void remap_wavefunction_sci_c64(
        void *old_src_ptr,
        const complexf64 *old_psi,
        void *new_src_ptr,
        complexf64 *new_psi,
        const uint32 *new_a,
        const uint32 *new_b,
        const complexf64 *new_v,
        int64 num_new)
    {
        auto *old_src = static_cast<SciBasisManager<uint32> *>(old_src_ptr);
        auto *new_src = static_cast<SciBasisManager<uint32> *>(new_src_ptr);

        std::vector<BufferedEntry<uint32, complexf64>> entries;
        entries.reserve(num_new);
        for (int64 i = 0; i < num_new; ++i)
            entries.push_back({new_a[i], new_b[i], new_v[i]});

        remap_wavefunction<uint32, complexf64>(
            old_src, old_psi, new_src, new_psi,
            num_new > 0 ? &entries : nullptr);
    }

    void *build_network_otf_sci_f64(
        const int64 *orbsym, int64 norb, int64 ngs,
        const uint32 *axs, const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_zas, const int64 *num_zbs,
        const uint32 *flat_zas, const uint32 *flat_zbs,
        const double *flat_wa, const double *flat_wb)
    {
        BasisManager<uint32> tmp;
        tmp.norb = norb;
        tmp.orbsym = new int64[norb];
        std::copy(orbsym, orbsym + norb, tmp.orbsym);

        void *result = build_network_otf<uint32, double>(
            &tmp, norb, ngs, axs, bxs,
            ranks, num_zas, num_zbs,
            flat_zas, flat_zbs, flat_wa, flat_wb);

        delete[] tmp.orbsym;
        return result;
    }

    void *build_network_otf_sci_c64(
        const int64 *orbsym, int64 norb, int64 ngs,
        const uint32 *axs, const uint32 *bxs,
        const int64 *ranks,
        const int64 *num_zas, const int64 *num_zbs,
        const uint32 *flat_zas, const uint32 *flat_zbs,
        const complexf64 *flat_wa, const complexf64 *flat_wb)
    {
        BasisManager<uint32> tmp;
        tmp.norb = norb;
        tmp.orbsym = new int64[norb];
        std::copy(orbsym, orbsym + norb, tmp.orbsym);

        void *result = build_network_otf<uint32, complexf64>(
            &tmp, norb, ngs, axs, bxs,
            ranks, num_zas, num_zbs,
            flat_zas, flat_zbs, flat_wa, flat_wb);

        delete[] tmp.orbsym;
        return result;
    }

    void *create_sci_basis_from_strings_f64(
        const uint32 *astrs, int64 na,
        const uint32 *bstrs, int64 nb,
        int64 norb, const int64 *orbsym,
        int64 total_sym, int64 num_irreps,
        int mode_code, int strat_code)
    {
        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        std::vector<bool> is_new_a(na, false);
        std::vector<bool> is_new_b(nb, false);

        return static_cast<void *>(
            create_sci_basis_manager<uint32>(
                astrs, na, is_new_a,
                bstrs, nb, is_new_b,
                norb, orbsym, total_sym, num_irreps,
                mode, strat, nullptr, nullptr));
    }

    void *create_sci_basis_from_strings_c64(
        const uint32 *astrs, int64 na,
        const uint32 *bstrs, int64 nb,
        int64 norb, const int64 *orbsym,
        int64 total_sym, int64 num_irreps,
        int mode_code, int strat_code)
    {
        return create_sci_basis_from_strings_f64(
            astrs, na, bstrs, nb, norb, orbsym, total_sym, num_irreps, mode_code, strat_code);
    }

    int64 sci_basis_dim(void *ptr)
    {
        return static_cast<SciBasisManager<uint32> *>(ptr)->dim;
    }

    int64 sci_basis_num_blocks(void *ptr)
    {
        return static_cast<SciBasisManager<uint32> *>(ptr)->num_blocks;
    }

    int64 sci_basis_num_alpha_strings(void *ptr)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(ptr);
        int64 total = 0;
        for (int64 i = 0; i < basis->num_irreps; ++i)
            total += basis->num_astrs[i];
        return total;
    }

    int64 sci_basis_num_beta_strings(void *ptr)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(ptr);
        int64 total = 0;
        for (int64 i = 0; i < basis->num_irreps; ++i)
            total += basis->num_bstrs[i];
        return total;
    }

    void get_diags_elements_sci_f64(void *basis_ptr, void *net_ptr, double *diags)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);
        get_diags_elements_sci<uint32, double>(basis, net, diags);
    }

    void get_diags_elements_sci_c64(void *basis_ptr, void *net_ptr, complexf64 *diags)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);
        get_diags_elements_sci<uint32, complexf64>(basis, net, diags);
    }

    void hvec_sci_full_f64(void *basis_ptr, void *net_ptr,
                           const double *src, double *dst)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, double> *>(net_ptr);

        std::fill_n(dst, basis->dim, 0.0);

        for (int64 blk = 0; blk < basis->num_blocks; ++blk)
        {
            const BlockDesc<uint32> &block = basis->blocks[blk];
            double *dst_blk = dst + block.offset;
            contract_hvec_sci_for_block<uint32, double>(
                basis, basis, net, blk, src, dst_blk);
        }
    }

    void hvec_sci_full_c64(void *basis_ptr, void *net_ptr,
                           const complexf64 *src, complexf64 *dst)
    {
        auto *basis = static_cast<SciBasisManager<uint32> *>(basis_ptr);
        auto *net = static_cast<Network_OTF<uint32, complexf64> *>(net_ptr);

        std::fill_n(dst, basis->dim, complexf64{});

        for (int64 blk = 0; blk < basis->num_blocks; ++blk)
        {
            const BlockDesc<uint32> &block = basis->blocks[blk];
            complexf64 *dst_blk = dst + block.offset;
            contract_hvec_sci_for_block<uint32, complexf64>(
                basis, basis, net, blk, src, dst_blk);
        }
    }

    void *create_sci_basis_from_standard_f64(void *existing_ptr,
                                              int mode_code, int strat_code)
    {
        auto *bs = static_cast<BasisManager<uint32> *>(existing_ptr);
        SelectMode mode = (mode_code == 0) ? SelectMode::Bitstring : SelectMode::Pair;
        SelectStrategy strat = (strat_code == 0) ? SelectStrategy::GrowOnly : SelectStrategy::Recompete;

        int64 num_irreps = bs->num_irreps;
        int64 total_a = 0, total_b = 0;
        for (int64 s = 0; s < num_irreps; ++s)
        {
            total_a += bs->num_astrs[s];
            total_b += bs->num_bstrs[s];
        }

        std::vector<uint32> astrs(total_a), bstrs(total_b);
        int64 a_off = 0, b_off = 0;
        for (int64 s = 0; s < num_irreps; ++s)
        {
            if (bs->num_astrs[s] > 0)
            {
                std::copy(bs->astrs_vec[s], bs->astrs_vec[s] + bs->num_astrs[s],
                          astrs.data() + a_off);
                a_off += bs->num_astrs[s];
            }
            if (bs->num_bstrs[s] > 0)
            {
                std::copy(bs->bstrs_vec[s], bs->bstrs_vec[s] + bs->num_bstrs[s],
                          bstrs.data() + b_off);
                b_off += bs->num_bstrs[s];
            }
        }

        std::vector<bool> is_new_a(total_a, false);
        std::vector<bool> is_new_b(total_b, false);

        return static_cast<void *>(
            create_sci_basis_manager<uint32>(
                astrs.data(), total_a, is_new_a,
                bstrs.data(), total_b, is_new_b,
                bs->norb, bs->orbsym, bs->total_sym, bs->num_irreps,
                mode, strat, nullptr, nullptr));
    }
}
