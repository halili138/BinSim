#pragma once
#include <vector>
#include <ankerl/unordered_dense.h>
#include "select/utils.hpp"

template <typename Ti, typename Tv>
struct BufferedEntry
{
    Ti astr;
    Ti bstr;
    Tv val;
};

template <typename Ti>
struct SciBasisView
{
    const BlockDesc<Ti> *blocks;
    int64 num_blocks;
    int max_a_count, max_b_count;
    const int64 *block_map;
    int64 num_irreps;
    const ankerl::unordered_dense::map<Ti, int> *a_idx_map;
    const ankerl::unordered_dense::map<Ti, int> *b_idx_map;
    const int64 *src_offsets;
};

template <typename Ti>
struct SciBasisManager
{
    Ti *all_astrs = nullptr;
    Ti *all_bstrs = nullptr;
    Ti **astrs_vec = nullptr;
    Ti **bstrs_vec = nullptr;

    int64 *num_astrs = nullptr;
    int64 *num_bstrs = nullptr;

    BlockDesc<Ti> *blocks = nullptr;
    int64 num_blocks = {};

    int64 *orbsym = nullptr;
    int64 *block_map = nullptr;
    int64 num_irreps = {};
    int64 total_sym = {};

    int64 dim = {};
    int64 norb = {};
    int max_a_count = {};
    int max_b_count = {};

    ankerl::unordered_dense::map<Ti, int> a_idx_map;
    ankerl::unordered_dense::map<Ti, int> b_idx_map;

    std::vector<int64> _src_offsets;
    SciBasisView<Ti> view;

    void _init_view()
    {
        int64 total = num_irreps * num_irreps;
        _src_offsets.resize(total, -1);
        for (int64 i = 0; i < num_blocks; ++i)
        {
            const auto &blk = blocks[i];
            int64 bid = blk.asym * num_irreps + blk.bsym;
            _src_offsets[bid] = blk.offset;
        }
        view = SciBasisView<Ti>{blocks, num_blocks,
                                max_a_count, max_b_count,
                                block_map, num_irreps,
                                &a_idx_map, &b_idx_map,
                                _src_offsets.data()};
    }

    void clear()
    {
        delete[] all_astrs;
        all_astrs = nullptr;
        delete[] all_bstrs;
        all_bstrs = nullptr;
        delete[] astrs_vec;
        astrs_vec = nullptr;
        delete[] bstrs_vec;
        bstrs_vec = nullptr;
        delete[] num_astrs;
        num_astrs = nullptr;
        delete[] num_bstrs;
        num_bstrs = nullptr;
        delete[] blocks;
        blocks = nullptr;
        delete[] orbsym;
        orbsym = nullptr;
        delete[] block_map;
        block_map = nullptr;
        num_blocks = 0;
        num_irreps = 0;
        total_sym = 0;
        dim = 0;
        norb = 0;
        max_a_count = 0;
        max_b_count = 0;
        a_idx_map.clear();
        b_idx_map.clear();
        _src_offsets.clear();
    }

    ~SciBasisManager() { clear(); }
};

template <typename Ti>
SciBasisManager<Ti> *create_sci_basis_manager(
    const Ti *input_astrs, int64 num_a_total,
    const Ti *input_bstrs, int64 num_b_total,
    int64 norb, const int64 *orbsym,
    int64 total_sym, int64 num_irreps)
{
    SciBasisManager<Ti> *basis = new SciBasisManager<Ti>();

    try
    {
        basis->num_irreps = num_irreps;
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;
        basis->total_sym = total_sym;

        basis->num_astrs = new int64[num_irreps]();
        basis->num_bstrs = new int64[num_irreps]();

        for (int64 i = 0; i < num_a_total; ++i)
        {
            int64 sym = get_string_sym(input_astrs[i], orbsym);
            if (sym < num_irreps)
                basis->num_astrs[sym]++;
        }
        for (int64 i = 0; i < num_b_total; ++i)
        {
            int64 sym = get_string_sym(input_bstrs[i], orbsym);
            if (sym < num_irreps)
                basis->num_bstrs[sym]++;
        }

        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym < num_irreps && basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
                basis->num_blocks++;
        }

        basis->all_astrs = new Ti[num_a_total];
        basis->all_bstrs = new Ti[num_b_total];
        basis->astrs_vec = new Ti *[num_irreps];
        basis->bstrs_vec = new Ti *[num_irreps];
        basis->blocks = new BlockDesc<Ti>[basis->num_blocks];
        basis->orbsym = new int64[norb];
        std::copy(orbsym, orbsym + norb, basis->orbsym);
        basis->block_map = new int64[num_irreps * num_irreps];
        std::fill_n(basis->block_map, num_irreps * num_irreps, -1);

        int64 a_offset = 0, b_offset = 0;
        for (int64 i = 0; i < num_irreps; ++i)
        {
            basis->astrs_vec[i] = basis->all_astrs + a_offset;
            a_offset += basis->num_astrs[i];
            basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
            b_offset += basis->num_bstrs[i];
        }

        int64 *a_idx = new int64[num_irreps]();
        int64 *b_idx = new int64[num_irreps]();
        for (int64 i = 0; i < num_a_total; ++i)
        {
            int64 sym = get_string_sym(input_astrs[i], orbsym);
            if (sym < num_irreps)
                basis->astrs_vec[sym][a_idx[sym]++] = input_astrs[i];
        }
        for (int64 i = 0; i < num_b_total; ++i)
        {
            int64 sym = get_string_sym(input_bstrs[i], orbsym);
            if (sym < num_irreps)
                basis->bstrs_vec[sym][b_idx[sym]++] = input_bstrs[i];
        }
        delete[] a_idx;
        delete[] b_idx;

        for (int64 i = 0; i < num_irreps; ++i)
        {
            if (basis->num_astrs[i] > 0)
                std::sort(basis->astrs_vec[i], basis->astrs_vec[i] + basis->num_astrs[i]);
            if (basis->num_bstrs[i] > 0)
                std::sort(basis->bstrs_vec[i], basis->bstrs_vec[i] + basis->num_bstrs[i]);
        }

        int64 block_counter = 0;
        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym >= num_irreps)
                continue;
            if (basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
            {
                BlockDesc<Ti> &block = basis->blocks[block_counter];
                block.asym = asym;
                block.bsym = bsym;
                block.num_a = basis->num_astrs[asym];
                block.num_b = basis->num_bstrs[bsym];
                block.astrs = basis->astrs_vec[asym];
                block.bstrs = basis->bstrs_vec[bsym];
                block.offset = basis->dim;
                basis->block_map[asym * num_irreps + bsym] = block_counter;
                block_counter++;
                basis->dim += block.num_a * block.num_b;
            }
        }

        basis->max_a_count = 0;
        basis->max_b_count = 0;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            if (basis->blocks[i].num_a > basis->max_a_count)
                basis->max_a_count = basis->blocks[i].num_a;
            if (basis->blocks[i].num_b > basis->max_b_count)
                basis->max_b_count = basis->blocks[i].num_b;
        }

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            const BlockDesc<Ti> &blk = basis->blocks[i];
            for (int a = 0; a < blk.num_a; ++a)
                basis->a_idx_map[blk.astrs[a]] = a;
            for (int b = 0; b < blk.num_b; ++b)
                basis->b_idx_map[blk.bstrs[b]] = b;
        }

        basis->_init_view();
    }
    catch (...)
    {
        basis->clear();
        delete basis;
        throw;
    }

    return basis;
}

template <typename Ti, typename Tv>
void remap_wavefunction(
    const SciBasisManager<Ti> *old_src, const Tv *old_psi,
    const SciBasisManager<Ti> *new_src, Tv *new_psi,
    const std::vector<BufferedEntry<Ti, Tv>> *new_entries)
{
    std::fill_n(new_psi, new_src->dim, Tv{});
    const int64 nirp = new_src->num_irreps;
    const int64 *orsym = new_src->orbsym;

    for (int64 i = 0; i < old_src->num_blocks; ++i)
    {
        const BlockDesc<Ti> &oblk = old_src->blocks[i];
        for (int a = 0; a < oblk.num_a; ++a)
        {
            Ti astr = oblk.astrs[a];
            for (int b = 0; b < oblk.num_b; ++b)
            {
                Ti bstr = oblk.bstrs[b];
                int64 old_pos = oblk.offset + (int64)a * oblk.num_b + b;
                int64 sa = get_string_sym(astr, orsym);
                int64 sb = get_string_sym(bstr, orsym);
                if (sa >= nirp || sb >= nirp)
                    continue;
                int64 nbid = new_src->block_map[sa * nirp + sb];
                if (nbid == -1)
                    continue;
                auto it_a = new_src->a_idx_map.find(astr);
                auto it_b = new_src->b_idx_map.find(bstr);
                if (it_a == new_src->a_idx_map.end() || it_b == new_src->b_idx_map.end())
                    continue;
                const BlockDesc<Ti> &nblk = new_src->blocks[nbid];
                int64 np = nblk.offset + (int64)it_a->second * nblk.num_b + it_b->second;
                new_psi[np] = old_psi[old_pos];
            }
        }
    }

    if (new_entries)
    {
        for (const auto &e : *new_entries)
        {
            int64 sa = get_string_sym(e.astr, orsym);
            int64 sb = get_string_sym(e.bstr, orsym);
            if (sa >= nirp || sb >= nirp)
                continue;
            int64 nbid = new_src->block_map[sa * nirp + sb];
            if (nbid == -1)
                continue;
            auto it_a = new_src->a_idx_map.find(e.astr);
            auto it_b = new_src->b_idx_map.find(e.bstr);
            if (it_a == new_src->a_idx_map.end() || it_b == new_src->b_idx_map.end())
                continue;
            const BlockDesc<Ti> &nblk = new_src->blocks[nbid];
            int64 np = nblk.offset + (int64)it_a->second * nblk.num_b + it_b->second;
            new_psi[np] = e.val;
        }
    }
}

template <typename Ti, typename Tv>
void get_diags_elements_sci(
    const SciBasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *net, Tv *diags)
{
    const SVDGroup_OTF<Ti, Tv> &group = net->diag_groups[0];
    const int rank = group.rank;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int max_a_count = basis->max_a_count;
    const int max_b_count = basis->max_b_count;

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);
        for (int64 bi = 0; bi < num_blocks; ++bi)
        {
            const BlockDesc<Ti> &blk = blocks[bi];
            Tv *pa = local_a_phase.data();
            Tv *pb = local_b_phase.data();
            for (int i = 0; i < blk.num_a; ++i)
                precompute_phase<0, Ti, Tv>(blk.astrs[i], zas, num_za, wa0, pa + i, max_a_count, rank);
            for (int i = 0; i < blk.num_b; ++i)
                precompute_phase<0, Ti, Tv>(blk.bstrs[i], zbs, num_zb, wb0, pb + i, max_b_count, rank);

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < blk.num_a; ++a)
                for (int b = 0; b < blk.num_b; ++b)
                    diags[blk.offset + a * blk.num_b + b] +=
                        compute_coeff<0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
        }
    }
}

template <typename Ti>
void destroy_sci_basis_manager(SciBasisManager<Ti> *basis)
{
    basis->clear();
    delete basis;
}
