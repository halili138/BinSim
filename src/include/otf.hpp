#pragma once
#include "common.hpp"
#include <vector>

template <typename Ti,
          typename Tv>
struct SVDGroup_OTF
{
    Ti ax;
    Ti bx;
    int rank;

    int num_za;
    Ti *unique_zas;
    int num_zb;
    Ti *unique_zbs;

    Tv *wa;
    Tv *wb;

    inline Tv *ptr_wa(int r) { return wa + r * num_za; }
    inline Tv *ptr_wb(int r) { return wb + r * num_zb; }
};

struct IndexMap
{
    const int *a_idx_map;
    const int *b_idx_map;
};

template <typename Ti,
          typename Tv>
struct Network_OTF
{
    int *excit_types;
    IndexMap map;
    int64 num_groups;
    SVDGroup_OTF<Ti, Tv> *groups;
};

template <typename Tv>
struct SharedBatchBuffer
{
    std::vector<int> src_b_idxs;
    std::vector<int> dst_b_idxs;
    std::vector<Tv> batch_phase;

    std::vector<int> valid_b_counts;
    std::vector<int> src_block_idxs;

    int max_b_count;
    int max_rank;

    SharedBatchBuffer(int64 batch_size, int mb, int mr)
        : max_b_count(mb), max_rank(mr)
    {
        src_b_idxs.resize(batch_size * mb);
        dst_b_idxs.resize(batch_size * mb);
        batch_phase.resize(batch_size * mb * mr);
        valid_b_counts.resize(batch_size);
        src_block_idxs.resize(batch_size);
    }

    inline int *ptr_src_b(int64 batch_idx) { return src_b_idxs.data() + batch_idx * max_b_count; }
    inline int *ptr_dst_b(int64 batch_idx) { return dst_b_idxs.data() + batch_idx * max_b_count; }
    inline Tv *ptr_phase(int64 batch_idx) { return batch_phase.data() + batch_idx * max_b_count * max_rank; }
};

template <typename Ti, typename Tv>
void destroy_network_otf(Network_OTF<Ti, Tv> *net)
{
    if (!net)
        return;

    delete[] net->excit_types;
    delete[] net->map.a_idx_map;
    delete[] net->map.b_idx_map;

    for (int64 g = 0; g < net->num_groups; ++g)
    {
        delete[] net->groups[g].unique_zas;
        delete[] net->groups[g].unique_zbs;
        delete[] net->groups[g].wa;
        delete[] net->groups[g].wb;
    }

    delete[] net->groups;
    delete net;
}

template <typename Ti, typename Tv>
void *build_network_otf(
    const BasisManager<Ti> *basis,
    int64 norb,
    int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_zas,
    const int64 *num_zbs,
    const Ti *flat_zas,
    const Ti *flat_zbs,
    const Tv *flat_wa,
    const Tv *flat_wb)
{
    Network_OTF<Ti, Tv> *net = new Network_OTF<Ti, Tv>();
    net->num_groups = ngs;

    net->excit_types = new int[ngs];

    int32 map_size = 1 << norb;
    int32 *a_map = new int32[map_size];
    int32 *b_map = new int32[map_size];
    std::fill(a_map, a_map + map_size, -1);
    std::fill(b_map, b_map + map_size, -1);

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        for (int32 a = 0; a < basis->blocks[i].num_a; ++a)
        {
            a_map[basis->blocks[i].astrs[a]] = a;
        }
        for (int32 b = 0; b < basis->blocks[i].num_b; ++b)
        {
            b_map[basis->blocks[i].bstrs[b]] = b;
        }
    }

    net->map.a_idx_map = a_map;
    net->map.b_idx_map = b_map;

    net->groups = new SVDGroup_OTF<Ti, Tv>[ngs];
    uint64 z_offset_a = 0, z_offset_b = 0;
    uint64 w_offset_a = 0, w_offset_b = 0;

    for (int64 g = 0; g < ngs; ++g)
    {
        Ti ax = axs[g];
        Ti bx = bxs[g];

        int type = 0;
        if (ax != 0 && bx == 0)
            type = 1;
        else if (ax == 0 && bx != 0)
            type = 2;
        else if (ax != 0 && bx != 0)
            type = 3;

        net->excit_types[g] = static_cast<uint8>(type);
        net->groups[g].ax = ax;
        net->groups[g].bx = bx;
        net->groups[g].rank = ranks[g];

        uint32 nza = num_zas[g];
        uint32 nzb = num_zbs[g];
        net->groups[g].num_za = nza;
        net->groups[g].num_zb = nzb;

        net->groups[g].unique_zas = new Ti[nza];
        std::copy(flat_zas + z_offset_a, flat_zas + z_offset_a + nza, net->groups[g].unique_zas);
        z_offset_a += nza;

        net->groups[g].unique_zbs = new Ti[nzb];
        std::copy(flat_zbs + z_offset_b, flat_zbs + z_offset_b + nzb, net->groups[g].unique_zbs);
        z_offset_b += nzb;

        uint64 wa_size = nza * ranks[g];
        net->groups[g].wa = new Tv[wa_size];
        std::copy(flat_wa + w_offset_a, flat_wa + w_offset_a + wa_size, net->groups[g].wa);
        w_offset_a += wa_size;

        uint64 wb_size = nzb * ranks[g];
        net->groups[g].wb = new Tv[wb_size];
        std::copy(flat_wb + w_offset_b, flat_wb + w_offset_b + wb_size, net->groups[g].wb);
        w_offset_b += wb_size;
    }

    return static_cast<void *>(net);
}

template <int Rank, typename Ti, typename Tv>
inline void gather_contract_diag_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    constexpr int64 BATCH_SIZE = 64;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;

    int max_b_count = 0;
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    int max_rank = 0;
    if constexpr (Rank == 1 || Rank == 2)
    {
        max_rank = Rank;
    }
    else
    {
        for (int64 g = 0; g < num_groups; ++g)
        {
            if (groups[g].rank > max_rank)
                max_rank = groups[g].rank;
        }
    }

    std::vector<Tv> batch_b_phase(BATCH_SIZE * max_b_count * max_rank);

#pragma omp parallel
    {
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const int a_count = block.num_a;
            const int b_count = block.num_b;

            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const uint16 rank = group.rank;
                    const Ti *zbs = group.unique_zbs;
                    Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                    Tv *pb1 = pb0 + max_b_count;

                    for (int b = 0; b < b_count; ++b)
                    {
                        const Ti str_b = block.bstrs[b];

                        if constexpr (Rank == 1)
                        {
                            Tv pt0 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                            }
                            pb0[b] = pt0;
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pt0 = {}, pt1 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            const Tv *w1 = group.ptr_wb(1);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                                pt1 += parity ? -w1[k] : w1[k];
                            }
                            pb0[b] = pt0;
                            pb1[b] = pt1;
                        }
                        else
                        {
                            Tv *pbn = pb0 + b * rank;
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wb(r);
                                for (int k = 0; k < group.num_zb; ++k)
                                {
                                    const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pbn[r] = ptn;
                            }
                        }
                    }
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < a_count; ++a)
                {
                    const Ti str_a = block.astrs[a];
                    const int64 ptr = block.offset + (int64)a * b_count;
                    const Tv *__restrict__ src = src_vec + ptr;
                    Tv *__restrict__ dst = dst_vec + ptr;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const uint16 rank = group.rank;
                        const Ti *zas = group.unique_zas;
                        const Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                        const Tv *pb1 = pb0 + max_b_count;

                        if constexpr (Rank == 1)
                        {
                            Tv pa0 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                            }
#pragma omp simd
                            for (int b = 0; b < b_count; ++b)
                            {
                                dst[b] += src[b] * pa0 * pb0[b];
                            }
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pa0 = {}, pa1 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            const Tv *w1 = group.ptr_wa(1);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                                pa1 += parity ? -w1[k] : w1[k];
                            }
#pragma omp simd
                            for (int b = 0; b < b_count; ++b)
                            {
                                dst[b] += src[b] * (pa0 * pb0[b] + pa1 * pb1[b]);
                            }
                        }
                        else
                        {
                            Tv pan[64] = {};
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wa(r);
                                for (int k = 0; k < group.num_za; ++k)
                                {
                                    const bool parity = std::popcount(str_a & zas[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pan[r] = ptn;
                            }
#pragma omp simd
                            for (int b = 0; b < b_count; ++b)
                            {
                                Tv vt = {};
                                const Tv *pbn = pb0 + b * rank;
                                for (uint16 r = 0; r < rank; ++r)
                                {
                                    vt += pan[r] * pbn[r];
                                }
                                dst[b] += src[b] * vt;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
inline void gather_contract_pure_a_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    constexpr int64 BATCH_SIZE = 64;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_b_count = 0;
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    int max_rank = 0;
    if constexpr (Rank == 1 || Rank == 2)
    {
        max_rank = Rank;
    }
    else
    {
        for (int64 g = 0; g < num_groups; ++g)
        {
            if (groups[g].rank > max_rank)
                max_rank = groups[g].rank;
        }
    }

    std::vector<Tv> batch_b_phase(BATCH_SIZE * max_b_count * max_rank);
    std::vector<int> batch_src_block_idxs(BATCH_SIZE);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 axsym = get_string_sym(group.ax, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + dst_block.bsym;
                    const int64 src_block_idx = block_map[bid];

                    batch_src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                        continue;

                    const uint16 rank = group.rank;
                    const Ti *zbs = group.unique_zbs;
                    Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                    Tv *pb1 = pb0 + max_b_count;

                    for (int b = 0; b < dst_b_count; ++b)
                    {
                        const Ti str_b = dst_block.bstrs[b];

                        if constexpr (Rank == 1)
                        {
                            Tv pt0 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                            }
                            pb0[b] = pt0;
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pt0 = {}, pt1 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            const Tv *w1 = group.ptr_wb(1);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                                pt1 += parity ? -w1[k] : w1[k];
                            }
                            pb0[b] = pt0;
                            pb1[b] = pt1;
                        }
                        else
                        {
                            Tv *pbn = pb0 + b * rank;
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wb(r);
                                for (int k = 0; k < group.num_zb; ++k)
                                {
                                    const bool parity = std::popcount(str_b & zbs[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pbn[r] = ptn;
                            }
                        }
                    }
                }
#pragma omp for schedule(dynamic)
                for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
                {
                    const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                    Tv *__restrict__ dst = dst_vec + dst_block.offset + (int64)dst_a_idx * dst_b_count;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        int src_block_idx = batch_src_block_idxs[batch_idx];

                        if (src_block_idx == -1)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1)
                            continue;

                        const uint16 rank = group.rank;
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const Ti *__restrict__ zas = group.unique_zas;
                        const Tv *__restrict__ src = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;
                        const Tv *pb0 = batch_b_phase.data() + batch_idx * max_b_count * max_rank;
                        const Tv *pb1 = pb0 + max_b_count;

                        if constexpr (Rank == 1)
                        {
                            Tv pa0 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                            }
#pragma omp simd
                            for (int b = 0; b < dst_b_count; ++b)
                            {
                                dst[b] += src[b] * pa0 * pb0[b];
                            }
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pa0 = {}, pa1 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            const Tv *w1 = group.ptr_wa(1);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                                pa1 += parity ? -w1[k] : w1[k];
                            }
#pragma omp simd
                            for (int b = 0; b < dst_b_count; ++b)
                            {
                                dst[b] += src[b] * (pa0 * pb0[b] + pa1 * pb1[b]);
                            }
                        }
                        else
                        {
                            Tv pan[64] = {};
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wa(r);
                                for (int k = 0; k < group.num_za; ++k)
                                {
                                    const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pan[r] = ptn;
                            }
#pragma omp simd
                            for (int b = 0; b < dst_b_count; ++b)
                            {
                                Tv vt = {};
                                const Tv *pbn = pb0 + b * rank;
                                for (uint16 r = 0; r < rank; ++r)
                                {
                                    vt += pan[r] * pbn[r];
                                }
                                dst[b] += src[b] * vt;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
inline void gather_contract_pure_b_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    constexpr int64 BATCH_SIZE = 64;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_b_count = 0;
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    int max_rank = 0;
    if constexpr (Rank == 1 || Rank == 2)
    {
        max_rank = Rank;
    }
    else
    {
        for (int64 g = 0; g < num_groups; ++g)
        {
            if (groups[g].rank > max_rank)
                max_rank = groups[g].rank;
        }
    }

    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 bxsym = get_string_sym(group.bx, orbsym);
                    const int64 bid = dst_block.asym * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];

                    batch_buf.src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                    {
                        batch_buf.valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    const uint16 rank = group.rank;
                    const Ti *zbs = group.unique_zbs;
                    int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                    int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                    Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                    Tv *pb1 = pb0 + max_b_count;

                    int count = 0;
                    for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                    {
                        const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                        const Ti src_str_b = dst_str_b ^ group.bx;
                        const int src_b_idx = idx_map.b_idx_map[src_str_b];

                        if (src_b_idx == -1)
                            continue;

                        src_b_ptr[count] = src_b_idx;
                        dst_b_ptr[count] = dst_b_idx;

                        if constexpr (Rank == 1)
                        {
                            Tv pt0 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                            }
                            pb0[count] = pt0;
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pt0 = {}, pt1 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            const Tv *w1 = group.ptr_wb(1);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                                pt1 += parity ? -w1[k] : w1[k];
                            }
                            pb0[count] = pt0;
                            pb1[count] = pt1;
                        }
                        else
                        {
                            Tv *pbn = pb0 + count * rank;
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wb(r);
                                for (int k = 0; k < group.num_zb; ++k)
                                {
                                    const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pbn[r] = ptn;
                            }
                        }
                        count++;
                    }
                    batch_buf.valid_b_counts[batch_idx] = count;
                }
#pragma omp for schedule(dynamic)
                for (int a = 0; a < dst_a_count; ++a)
                {
                    const Ti str_a = dst_block.astrs[a];
                    Tv *__restrict__ dst = dst_vec + dst_block.offset + (int64)a * dst_b_count;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const uint16 rank = group.rank;
                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const Ti *__restrict__ zas = group.unique_zas;
                        const Tv *__restrict__ src = src_vec + src_block.offset + (int64)a * src_block.num_b;

                        const int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const Tv *pb1 = pb0 + max_b_count;

                        if constexpr (Rank == 1)
                        {
                            Tv pa0 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * pa0 * pb0[vb];
                            }
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pa0 = {}, pa1 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            const Tv *w1 = group.ptr_wa(1);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                                pa1 += parity ? -w1[k] : w1[k];
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * (pa0 * pb0[vb] + pa1 * pb1[vb]);
                            }
                        }
                        else
                        {
                            Tv pan[64] = {};
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wa(r);
                                for (int k = 0; k < group.num_za; ++k)
                                {
                                    const bool parity = std::popcount(str_a & zas[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pan[r] = ptn;
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                Tv vt = {};
                                const Tv *pbn = pb0 + vb * rank;
                                for (uint16 r = 0; r < rank; ++r)
                                {
                                    vt += pan[r] * pbn[r];
                                }
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * vt;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
inline void gather_contract_mixed_batched_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const IndexMap &idx_map,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    const int64 num_groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    constexpr int64 BATCH_SIZE = 64;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int64 *block_map = basis->block_map;
    const int64 *orbsym = basis->orbsym;
    const int64 num_irreps = basis->num_irreps;

    int max_b_count = 0;
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
    }

    int max_rank = 0;
    if constexpr (Rank == 1 || Rank == 2)
    {
        max_rank = Rank;
    }
    else
    {
        for (int64 g = 0; g < num_groups; ++g)
        {
            if (groups[g].rank > max_rank)
                max_rank = groups[g].rank;
        }
    }

    SharedBatchBuffer<Tv> batch_buf(BATCH_SIZE, max_b_count, max_rank);

#pragma omp parallel
    {
        for (int dst_block_idx = 0; dst_block_idx < num_blocks; ++dst_block_idx)
        {
            const BlockDesc<Ti> &dst_block = blocks[dst_block_idx];
            const int dst_a_count = dst_block.num_a;
            const int dst_b_count = dst_block.num_b;

            for (int64 batch_start = 0; batch_start < num_groups; batch_start += BATCH_SIZE)
            {
                const int64 cur_batch_size = std::min(BATCH_SIZE, num_groups - batch_start);

#pragma omp for schedule(dynamic)
                for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                {
                    const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                    const int64 axsym = get_string_sym(group.ax, orbsym);
                    const int64 bxsym = get_string_sym(group.bx, orbsym);
                    const int64 bid = (dst_block.asym ^ axsym) * num_irreps + (dst_block.bsym ^ bxsym);
                    const int64 src_block_idx = block_map[bid];

                    batch_buf.src_block_idxs[batch_idx] = src_block_idx;

                    if (src_block_idx == -1)
                    {
                        batch_buf.valid_b_counts[batch_idx] = 0;
                        continue;
                    }

                    const uint16 rank = group.rank;
                    const Ti *zbs = group.unique_zbs;
                    int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                    int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                    Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                    Tv *pb1 = pb0 + max_b_count;

                    int count = 0;
                    for (int dst_b_idx = 0; dst_b_idx < dst_b_count; ++dst_b_idx)
                    {
                        const Ti dst_str_b = dst_block.bstrs[dst_b_idx];
                        const Ti src_str_b = dst_str_b ^ group.bx;
                        const int src_b_idx = idx_map.b_idx_map[src_str_b];

                        if (src_b_idx == -1)
                            continue;

                        src_b_ptr[count] = src_b_idx;
                        dst_b_ptr[count] = dst_b_idx;

                        if constexpr (Rank == 1)
                        {
                            Tv pt0 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                            }
                            pb0[count] = pt0;
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pt0 = {}, pt1 = {};
                            const Tv *w0 = group.ptr_wb(0);
                            const Tv *w1 = group.ptr_wb(1);
                            for (int k = 0; k < group.num_zb; ++k)
                            {
                                const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                pt0 += parity ? -w0[k] : w0[k];
                                pt1 += parity ? -w1[k] : w1[k];
                            }
                            pb0[count] = pt0;
                            pb1[count] = pt1;
                        }
                        else
                        {
                            Tv *pbn = pb0 + count * rank;
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wb(r);
                                for (int k = 0; k < group.num_zb; ++k)
                                {
                                    const bool parity = std::popcount(src_str_b & zbs[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pbn[r] = ptn;
                            }
                        }
                        count++;
                    }
                    batch_buf.valid_b_counts[batch_idx] = count;
                }
#pragma omp for schedule(dynamic)
                for (int dst_a_idx = 0; dst_a_idx < dst_a_count; ++dst_a_idx)
                {
                    const Ti dst_str_a = dst_block.astrs[dst_a_idx];
                    Tv *__restrict__ dst = dst_vec + dst_block.offset + (int64)dst_a_idx * dst_b_count;

                    for (int64 batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
                    {
                        const int valid_b_count = batch_buf.valid_b_counts[batch_idx];
                        if (valid_b_count == 0)
                            continue;

                        const SVDGroup_OTF<Ti, Tv> &group = groups[batch_start + batch_idx];
                        const Ti src_str_a = dst_str_a ^ group.ax;
                        const int src_a_idx = idx_map.a_idx_map[src_str_a];

                        if (src_a_idx == -1)
                            continue;

                        const uint16 rank = group.rank;
                        const int src_block_idx = batch_buf.src_block_idxs[batch_idx];
                        const BlockDesc<Ti> &src_block = blocks[src_block_idx];
                        const Ti *__restrict__ zas = group.unique_zas;
                        const Tv *__restrict__ src = src_vec + src_block.offset + (int64)src_a_idx * src_block.num_b;

                        const int *src_b_ptr = batch_buf.ptr_src_b(batch_idx);
                        const int *dst_b_ptr = batch_buf.ptr_dst_b(batch_idx);
                        const Tv *pb0 = batch_buf.ptr_phase(batch_idx);
                        const Tv *pb1 = pb0 + max_b_count;

                        if constexpr (Rank == 1)
                        {
                            Tv pa0 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * pa0 * pb0[vb];
                            }
                        }
                        else if constexpr (Rank == 2)
                        {
                            Tv pa0 = {}, pa1 = {};
                            const Tv *w0 = group.ptr_wa(0);
                            const Tv *w1 = group.ptr_wa(1);
                            for (int k = 0; k < group.num_za; ++k)
                            {
                                const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                pa0 += parity ? -w0[k] : w0[k];
                                pa1 += parity ? -w1[k] : w1[k];
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * (pa0 * pb0[vb] + pa1 * pb1[vb]);
                            }
                        }
                        else
                        {
                            Tv pan[64] = {};
                            for (uint16 r = 0; r < rank; ++r)
                            {
                                Tv ptn = {};
                                const Tv *wr = group.ptr_wa(r);
                                for (int k = 0; k < group.num_za; ++k)
                                {
                                    const bool parity = std::popcount(src_str_a & zas[k]) & 1;
                                    ptn += parity ? -wr[k] : wr[k];
                                }
                                pan[r] = ptn;
                            }
#pragma omp simd
                            for (int vb = 0; vb < valid_b_count; ++vb)
                            {
                                Tv vt = {};
                                const Tv *pbn = pb0 + vb * rank;
                                for (uint16 r = 0; r < rank; ++r)
                                {
                                    vt += pan[r] * pbn[r];
                                }
                                dst[dst_b_ptr[vb]] += src[src_b_ptr[vb]] * vt;
                            }
                        }
                    }
                }
            }
        }
    }
}
