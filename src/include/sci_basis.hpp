#pragma once
#include "common.hpp"
#include "basis.hpp"
#include "otf.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename Ti, typename Tv>
struct BufferedEntry
{
    Ti astr;
    Ti bstr;
    Tv val;
};

struct NewPair {
    int64 block_id;
    int a_local;
    int b_local;
};

struct OldPair {
    int64 block_id;
    int a_local;
    int b_local;
    int64 src_offset;
};

enum class SelectMode { Bitstring, Pair };
enum class SelectStrategy { GrowOnly, Recompete };

template <typename Ti>
struct SciBasisView
{
    const BlockDesc<Ti> *blocks;
    int64 num_blocks;
    int max_a_count, max_b_count;
    const int64 *block_map;
    int64 num_irreps;
    const std::unordered_map<Ti, int> *a_idx_map;
    const std::unordered_map<Ti, int> *b_idx_map;
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

    std::unordered_map<Ti, int> a_idx_map;
    std::unordered_map<Ti, int> b_idx_map;

    std::vector<int64> _src_offsets;

    SelectMode mode = SelectMode::Bitstring;
    SelectStrategy strat = SelectStrategy::GrowOnly;

    std::vector<bool> is_new_a;
    std::vector<bool> is_new_b;

    std::vector<NewPair> new_pairs;
    std::vector<OldPair> old_pairs;

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
        is_new_a.clear();
        is_new_b.clear();
        new_pairs.clear();
        old_pairs.clear();
        _src_offsets.clear();
    }

    ~SciBasisManager() { clear(); }
};

template <typename Ti, typename Tv>
void expand_bitstrings(
    const SciBasisManager<Ti> *src_basis,
    const Tv *src_psi,
    const SVDGroup_OTF<Ti, Tv> *groups, int64 ngs,
    int64 norb, const int64 *orbsym,
    int64 num_irreps,
    SelectMode mode,
    std::vector<Ti> &dst_astrs,
    std::vector<bool> &is_new_a,
    std::vector<Ti> &dst_bstrs,
    std::vector<bool> &is_new_b,
    std::vector<std::pair<Ti, Ti>> *new_pairs_strs)
{
    std::unordered_map<Ti, int> a_map;
    std::unordered_map<Ti, int> b_map;

    // Insert source astrs/bstrs (mark 0 = old)
    for (int64 i = 0; i < src_basis->num_blocks; ++i)
    {
        const BlockDesc<Ti> &blk = src_basis->blocks[i];
        for (int a = 0; a < blk.num_a; ++a)
            a_map.try_emplace(blk.astrs[a], 0);
        for (int b = 0; b < blk.num_b; ++b)
            b_map.try_emplace(blk.bstrs[b], 0);
    }

    // XOR-expand: apply groups to source astrs/bstrs
    for (int64 g = 0; g < ngs; ++g)
    {
        const Ti ax = groups[g].ax;
        const Ti bx = groups[g].bx;

        if (ax != 0)
        {
            for (int64 i = 0; i < src_basis->num_blocks; ++i)
            {
                const BlockDesc<Ti> &blk = src_basis->blocks[i];
                for (int a = 0; a < blk.num_a; ++a)
                {
                    Ti da = blk.astrs[a] ^ ax;
                    a_map.try_emplace(da, 1);
                }
            }
        }

        if (bx != 0)
        {
            for (int64 i = 0; i < src_basis->num_blocks; ++i)
            {
                const BlockDesc<Ti> &blk = src_basis->blocks[i];
                for (int b = 0; b < blk.num_b; ++b)
                {
                    Ti db = blk.bstrs[b] ^ bx;
                    b_map.try_emplace(db, 1);
                }
            }
        }
    }

    // Group by symmetry and sort
    {
        std::vector<std::vector<std::pair<Ti,int>>> a_by_sym(num_irreps);

        for (auto &[astr, tag] : a_map)
        {
            int64 sym = get_string_sym(astr, orbsym);
            if (sym >= num_irreps)
                continue;
            a_by_sym[sym].push_back({astr, tag});
        }

        dst_astrs.clear();
        is_new_a.clear();
        for (int64 s = 0; s < num_irreps; ++s)
        {
            std::sort(a_by_sym[s].begin(), a_by_sym[s].end(),
                      [](auto &x, auto &y) { return x.first < y.first; });
            for (auto &[astr, tag] : a_by_sym[s])
            {
                dst_astrs.push_back(astr);
                is_new_a.push_back(tag != 0);
            }
        }
    }

    {
        std::vector<std::vector<std::pair<Ti,int>>> b_by_sym(num_irreps);

        for (auto &[bstr, tag] : b_map)
        {
            int64 sym = get_string_sym(bstr, orbsym);
            if (sym >= num_irreps)
                continue;
            b_by_sym[sym].push_back({bstr, tag});
        }

        dst_bstrs.clear();
        is_new_b.clear();
        for (int64 s = 0; s < num_irreps; ++s)
        {
            std::sort(b_by_sym[s].begin(), b_by_sym[s].end(),
                      [](auto &x, auto &y) { return x.first < y.first; });
            for (auto &[bstr, tag] : b_by_sym[s])
            {
                dst_bstrs.push_back(bstr);
                is_new_b.push_back(tag != 0);
            }
        }
    }

    // Mode B: collect new pairs from non-zero source states
    if (mode == SelectMode::Pair && new_pairs_strs != nullptr)
    {
        std::unordered_set<uint64_t> seen;
        new_pairs_strs->clear();

        for (int64 i = 0; i < src_basis->num_blocks; ++i)
        {
            const BlockDesc<Ti> &blk = src_basis->blocks[i];
            for (int a = 0; a < blk.num_a; ++a)
            {
                const Ti sa = blk.astrs[a];
                for (int b = 0; b < blk.num_b; ++b)
                {
                    const int64 src_pos = blk.offset + (int64)a * blk.num_b + b;
                    if (src_psi != nullptr && src_psi[src_pos] == Tv{})
                        continue;

                    const Ti sb = blk.bstrs[b];

                    for (int64 g = 0; g < ngs; ++g)
                    {
                        const Ti ax = groups[g].ax;
                        const Ti bx = groups[g].bx;

                        Ti da = sa ^ ax;
                        Ti db = sb ^ bx;

                        if (da == sa && db == sb)
                            continue;

                        auto it_a = a_map.find(da);
                        if (it_a == a_map.end())
                            continue;
                        auto it_b = b_map.find(db);
                        if (it_b == b_map.end())
                            continue;

                        if (it_a->second == 0 && it_b->second == 0)
                            continue;

                        uint64_t key = (uint64_t(da) << 32) | uint64_t(db);
                        if (seen.insert(key).second)
                            new_pairs_strs->push_back({da, db});
                    }
                }
            }
        }
    }
}

template <typename Ti>
SciBasisManager<Ti> *create_sci_basis_manager(
    const Ti *input_astrs, int64 num_a_total,
    const std::vector<bool> &is_new_a_in,
    const Ti *input_bstrs, int64 num_b_total,
    const std::vector<bool> &is_new_b_in,
    int64 norb, const int64 *orbsym,
    int64 total_sym, int64 num_irreps,
    SelectMode mode, SelectStrategy strat,
    const std::vector<std::pair<Ti,Ti>> *new_pairs_strs,
    const SciBasisManager<Ti> *src_basis)
{
    SciBasisManager<Ti> *basis = new SciBasisManager<Ti>();

    try
    {
        basis->num_irreps = num_irreps;
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;
        basis->total_sym = total_sym;
        basis->mode = mode;
        basis->strat = strat;

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

        int64 a_offset = 0;
        int64 b_offset = 0;
        for (int64 i = 0; i < num_irreps; ++i)
        {
            basis->astrs_vec[i] = basis->all_astrs + a_offset;
            a_offset += basis->num_astrs[i];

            basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
            b_offset += basis->num_bstrs[i];
        }

        // Fill per-symmetry arrays and is_new markers
        basis->is_new_a.resize(num_a_total);
        basis->is_new_b.resize(num_b_total);

        int64 *a_idx = new int64[num_irreps]();
        int64 *b_idx = new int64[num_irreps]();

        for (int64 i = 0; i < num_a_total; ++i)
        {
            Ti astr = input_astrs[i];
            int64 sym = get_string_sym(astr, orbsym);
            if (sym < num_irreps)
            {
                int64 pos = a_idx[sym]++;
                basis->astrs_vec[sym][pos] = astr;
                int64 flat_idx = basis->astrs_vec[sym] - basis->all_astrs + pos;
            }
        }

        for (int64 i = 0; i < num_b_total; ++i)
        {
            Ti bstr = input_bstrs[i];
            int64 sym = get_string_sym(bstr, orbsym);
            if (sym < num_irreps)
            {
                int64 pos = b_idx[sym]++;
                basis->bstrs_vec[sym][pos] = bstr;
            }
        }

        delete[] a_idx;
        delete[] b_idx;

        // Sort within symmetry blocks
        for (int64 i = 0; i < num_irreps; ++i)
        {
            if (basis->num_astrs[i] > 0)
                std::sort(basis->astrs_vec[i], basis->astrs_vec[i] + basis->num_astrs[i]);
            if (basis->num_bstrs[i] > 0)
                std::sort(basis->bstrs_vec[i], basis->bstrs_vec[i] + basis->num_bstrs[i]);
        }

        // Build is_new markers by matching sorted astrs/bstrs back
        {
            std::unordered_map<Ti, int> a_tag;
            for (int64 i = 0; i < num_a_total; ++i)
                a_tag.try_emplace(input_astrs[i], is_new_a_in[i] ? 1 : 0);
            for (int64 s = 0; s < num_irreps; ++s)
            {
                for (int a = 0; a < basis->num_astrs[s]; ++a)
                {
                    Ti astr = basis->astrs_vec[s][a];
                    int64 global_a = basis->astrs_vec[s] - basis->all_astrs + a;
                    auto it = a_tag.find(astr);
                    basis->is_new_a[global_a] = (it != a_tag.end() && it->second != 0);
                }
            }
        }
        {
            std::unordered_map<Ti, int> b_tag;
            for (int64 i = 0; i < num_b_total; ++i)
                b_tag.try_emplace(input_bstrs[i], is_new_b_in[i] ? 1 : 0);
            for (int64 s = 0; s < num_irreps; ++s)
            {
                for (int b = 0; b < basis->num_bstrs[s]; ++b)
                {
                    Ti bstr = basis->bstrs_vec[s][b];
                    int64 global_b = basis->bstrs_vec[s] - basis->all_bstrs + b;
                    auto it = b_tag.find(bstr);
                    basis->is_new_b[global_b] = (it != b_tag.end() && it->second != 0);
                }
            }
        }

        // Build blocks
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

        // Compute max counts
        basis->max_a_count = 0;
        basis->max_b_count = 0;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            if (basis->blocks[i].num_a > basis->max_a_count)
                basis->max_a_count = basis->blocks[i].num_a;
            if (basis->blocks[i].num_b > basis->max_b_count)
                basis->max_b_count = basis->blocks[i].num_b;
        }

        // Build dictionary idx_maps
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            const BlockDesc<Ti> &blk = basis->blocks[i];
            for (int a = 0; a < blk.num_a; ++a)
                basis->a_idx_map[blk.astrs[a]] = a;
            for (int b = 0; b < blk.num_b; ++b)
                basis->b_idx_map[blk.bstrs[b]] = b;
        }

        // Mode B: convert new_pair strings to local indices
        if (mode == SelectMode::Pair && new_pairs_strs != nullptr)
        {
            basis->new_pairs.reserve(new_pairs_strs->size());
            for (auto &[astr, bstr] : *new_pairs_strs)
            {
                int64 sym_a = get_string_sym(astr, orbsym);
                int64 sym_b = get_string_sym(bstr, orbsym);
                if (sym_a >= num_irreps || sym_b >= num_irreps)
                    continue;

                int64 block_id = basis->block_map[sym_a * num_irreps + sym_b];
                if (block_id == -1)
                    continue;

                auto it_a = basis->a_idx_map.find(astr);
                if (it_a == basis->a_idx_map.end())
                    continue;
                auto it_b = basis->b_idx_map.find(bstr);
                if (it_b == basis->b_idx_map.end())
                    continue;

                basis->new_pairs.push_back({block_id, it_a->second, it_b->second});
            }
        }

        // Strategy 2 + Mode B: build old_pairs by mapping source states to target positions
        if (strat == SelectStrategy::Recompete && mode == SelectMode::Pair && src_basis != nullptr)
        {
            basis->old_pairs.reserve(src_basis->dim);
            for (int64 i = 0; i < src_basis->num_blocks; ++i)
            {
                const BlockDesc<Ti> &sblk = src_basis->blocks[i];
                for (int a = 0; a < sblk.num_a; ++a)
                {
                    Ti astr = sblk.astrs[a];
                    for (int b = 0; b < sblk.num_b; ++b)
                    {
                        Ti bstr = sblk.bstrs[b];
                        int64 src_pos = sblk.offset + (int64)a * sblk.num_b + b;

                        int64 sym_a = get_string_sym(astr, orbsym);
                        int64 sym_b = get_string_sym(bstr, orbsym);
                        if (sym_a >= num_irreps || sym_b >= num_irreps)
                            continue;

                        int64 tgt_block_id = basis->block_map[sym_a * num_irreps + sym_b];
                        if (tgt_block_id == -1)
                            continue;

                        auto it_a = basis->a_idx_map.find(astr);
                        if (it_a == basis->a_idx_map.end())
                            continue;
                        auto it_b = basis->b_idx_map.find(bstr);
                        if (it_b == basis->b_idx_map.end())
                            continue;

                        basis->old_pairs.push_back({tgt_block_id, it_a->second, it_b->second, src_pos});
                    }
                }
            }
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
SciBasisManager<Ti> *create_source_basis_from_merge(
    const SciBasisManager<Ti> *src_basis,
    const std::vector<BufferedEntry<Ti, Tv>> &selected,
    int64 norb, const int64 *orbsym,
    int64 total_sym, int64 num_irreps,
    SelectMode mode, SelectStrategy strat)
{
    std::unordered_set<Ti> a_set, b_set;

    for (int64 i = 0; i < src_basis->num_blocks; ++i)
    {
        const BlockDesc<Ti> &blk = src_basis->blocks[i];
        for (int a = 0; a < blk.num_a; ++a)
            a_set.insert(blk.astrs[a]);
        for (int b = 0; b < blk.num_b; ++b)
            b_set.insert(blk.bstrs[b]);
    }

    for (const auto &e : selected)
    {
        a_set.insert(e.astr);
        b_set.insert(e.bstr);
    }

    std::vector<Ti> flat_a(a_set.begin(), a_set.end());
    std::vector<Ti> flat_b(b_set.begin(), b_set.end());

    std::vector<bool> is_new_a(flat_a.size(), false);
    std::vector<bool> is_new_b(flat_b.size(), false);

    return create_sci_basis_manager<Ti>(
        flat_a.data(), flat_a.size(), is_new_a,
        flat_b.data(), flat_b.size(), is_new_b,
        norb, orbsym, total_sym, num_irreps,
        mode, strat, nullptr, nullptr);
}

template <typename Ti, typename Tv>
void remap_wavefunction(
    const SciBasisManager<Ti> *old_src,
    const Tv *old_psi,
    const SciBasisManager<Ti> *new_src,
    Tv *new_psi,
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
                int64 nb_id = new_src->block_map[sa * nirp + sb];
                if (nb_id == -1)
                    continue;

                auto it_a = new_src->a_idx_map.find(astr);
                auto it_b = new_src->b_idx_map.find(bstr);
                if (it_a == new_src->a_idx_map.end() || it_b == new_src->b_idx_map.end())
                    continue;

                const BlockDesc<Ti> &nblk = new_src->blocks[nb_id];
                int64 new_pos = nblk.offset + (int64)it_a->second * nblk.num_b + it_b->second;
                new_psi[new_pos] = old_psi[old_pos];
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
            int64 nb_id = new_src->block_map[sa * nirp + sb];
            if (nb_id == -1)
                continue;

            auto it_a = new_src->a_idx_map.find(e.astr);
            auto it_b = new_src->b_idx_map.find(e.bstr);
            if (it_a == new_src->a_idx_map.end() || it_b == new_src->b_idx_map.end())
                continue;

            const BlockDesc<Ti> &nblk = new_src->blocks[nb_id];
            int64 new_pos = nblk.offset + (int64)it_a->second * nblk.num_b + it_b->second;
            new_psi[new_pos] = e.val;
        }
    }
}

template <typename Ti, typename Tv>
void get_diags_elements_sci(
    const SciBasisManager<Ti> *basis,
    const Network_OTF<Ti, Tv> *net,
    Tv *diags)
{
    const SVDGroup_OTF<Ti, Tv> &group = net->diag_groups[0];
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int max_a_count = basis->max_a_count;
    const int max_b_count = basis->max_b_count;

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int64 block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            for (int i = 0; i < block.num_a; ++i)
                precompute_phase<0, Ti, Tv>(block.astrs[i], zas, num_za, wa0,
                                            pa0 + i, max_a_count, rank);
            for (int i = 0; i < block.num_b; ++i)
                precompute_phase<0, Ti, Tv>(block.bstrs[i], zbs, num_zb, wb0,
                                            pb0 + i, max_b_count, rank);

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < block.num_a; ++a)
            {
                for (int b = 0; b < block.num_b; ++b)
                {
                    const Tv vt = compute_coeff<0, Tv>(a, b, pa, pb,
                                                       max_a_count, max_b_count, rank);
                    diags[block.offset + (int64)a * block.num_b + b] += vt;
                }
            }
        }
    }
}

template <typename Ti>
void destroy_sci_basis_manager(SciBasisManager<Ti> *basis)
{
    basis->clear();
    delete basis;
}
