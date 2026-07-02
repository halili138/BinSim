#pragma once
#include "sci_hvec.hpp"
#include <cmath>

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>)
        return v * v;
    else
        return v.real() * v.real() + v.imag() * v.imag();
}

template <typename Ti>
inline void build_block_new_set(
    const SciBasisManager<Ti> *tgt_basis,
    int64 block_idx,
    std::unordered_set<uint64_t> &out)
{
    out.clear();
    for (const auto &p : tgt_basis->new_pairs)
    {
        if (p.block_id == block_idx)
        {
            uint64_t key = (uint64_t(p.a_local) << 32) | uint64_t(p.b_local);
            out.insert(key);
        }
    }
}

template <typename Ti, typename Tv>
int64 sci_hvec_select_for_block(
    const SciBasisManager<Ti> *tgt_basis,
    const SciBasisManager<Ti> *src_basis,
    const Network_OTF<Ti, Tv> *net,
    int64 block_idx,
    const Tv *src_vec,
    int chunk_size,
    double eps,
    BufferedEntry<Ti, Tv> *out_entries,
    int64 max_entries)
{
    const BlockDesc<Ti> &full_block = tgt_basis->blocks[block_idx];
    const int64 num_a_total = full_block.num_a;
    const int64 num_b = full_block.num_b;
    const SelectMode mode = tgt_basis->mode;

    const int64 ga_base = full_block.astrs - tgt_basis->all_astrs;
    const int64 gb_base = full_block.bstrs - tgt_basis->all_bstrs;

    std::unordered_set<uint64_t> block_new_set;
    if (mode == SelectMode::Pair)
        build_block_new_set(tgt_basis, block_idx, block_new_set);

    int64 out_count = 0;

    for (int64 a_start = 0; a_start < num_a_total; a_start += chunk_size)
    {
        const int64 a_end = std::min(a_start + (int64)chunk_size, num_a_total);
        const int64 cur_num_a = a_end - a_start;

        BlockDesc<Ti> chunk_desc = full_block;
        chunk_desc.astrs = full_block.astrs + a_start;
        chunk_desc.num_a = cur_num_a;
        chunk_desc.offset = 0;

        Tv *chunk_acc = new Tv[cur_num_a * num_b];
        std::fill_n(chunk_acc, cur_num_a * num_b, Tv{});

        dispatch_chunks_for_block<0>(chunk_desc, src_basis, net->diag_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<1>(chunk_desc, src_basis, net->pure_a_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<2>(chunk_desc, src_basis, net->pure_b_groups, src_vec, chunk_acc);
        dispatch_chunks_for_block<3>(chunk_desc, src_basis, net->mixed_groups, src_vec, chunk_acc);

        for (int a = 0; a < cur_num_a && out_count < max_entries; ++a)
        {
            const int64 a_global = a_start + a;
            const Tv *row = chunk_acc + a * num_b;

            if (mode == SelectMode::Bitstring)
            {
                bool a_new = tgt_basis->is_new_a[ga_base + a_global];
                for (int b = 0; b < num_b && out_count < max_entries; ++b)
                {
                    if (row[b] == Tv{})
                        continue;
                    if (!a_new && !tgt_basis->is_new_b[gb_base + b])
                        continue;
                    if (sqnorm(row[b]) <= eps * eps)
                        continue;

                    out_entries[out_count++] = {full_block.astrs[a_global],
                                                full_block.bstrs[b], row[b]};
                }
            }
            else
            {
                for (int b = 0; b < num_b && out_count < max_entries; ++b)
                {
                    if (row[b] == Tv{})
                        continue;
                    if (sqnorm(row[b]) <= eps * eps)
                        continue;

                    uint64_t key = (uint64_t(a_global) << 32) | uint64_t(b);
                    if (block_new_set.find(key) == block_new_set.end())
                        continue;

                    out_entries[out_count++] = {full_block.astrs[a_global],
                                                full_block.bstrs[b], row[b]};
                }
            }
        }

        delete[] chunk_acc;
    }

    return out_count;
}
