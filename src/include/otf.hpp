#pragma once
#include "common.hpp"

inline constexpr int64 BATCH_SIZE = 256;

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

    inline Tv *ptr_wa(int r) const { return wa + r * num_za; }
    inline Tv *ptr_wb(int r) const { return wb + r * num_zb; }
};

struct IndexMap
{
    const int *a_idx_map;
    const int *b_idx_map;
};

template <typename Ti, typename Tv>
struct Network_OTF
{
    IndexMap map;
    int64 num_groups;

    std::vector<SVDGroup_OTF<Ti, Tv>> diag_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_a_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> pure_b_groups;
    std::vector<SVDGroup_OTF<Ti, Tv>> mixed_groups;

    uint8 *excit_types = nullptr;
    SVDGroup_OTF<Ti, Tv> *flat_groups = nullptr;
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
void destroy_network_otf(void *net_ptr)
{
    if (!net_ptr)
        return;

    Network_OTF<Ti, Tv> *net = static_cast<Network_OTF<Ti, Tv> *>(net_ptr);

    auto free_group_inner = [](SVDGroup_OTF<Ti, Tv> &group)
    {
        if (group.unique_zas)
        {
            delete[] group.unique_zas;
            group.unique_zas = nullptr;
        }
        if (group.unique_zbs)
        {
            delete[] group.unique_zbs;
            group.unique_zbs = nullptr;
        }
        if (group.wa)
        {
            delete[] group.wa;
            group.wa = nullptr;
        }
        if (group.wb)
        {
            delete[] group.wb;
            group.wb = nullptr;
        }
    };

    auto clear_bucket = [&](std::vector<SVDGroup_OTF<Ti, Tv>> &group_vec)
    {
        for (auto &group : group_vec)
        {
            free_group_inner(group);
        }
        group_vec.clear();
    };

    clear_bucket(net->diag_groups);
    clear_bucket(net->pure_a_groups);
    clear_bucket(net->pure_b_groups);
    clear_bucket(net->mixed_groups);

    if (net->flat_groups)
    {
        for (int64 i = 0; i < net->num_groups; ++i)
        {
            free_group_inner(net->flat_groups[i]);
        }
        delete[] net->flat_groups;
        net->flat_groups = nullptr;
    }

    if (net->excit_types)
    {
        delete[] net->excit_types;
        net->excit_types = nullptr;
    }

    if (net->map.a_idx_map)
    {
        delete[] net->map.a_idx_map;
        net->map.a_idx_map = nullptr;
    }
    if (net->map.b_idx_map)
    {
        delete[] net->map.b_idx_map;
        net->map.b_idx_map = nullptr;
    }

    delete net;
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE void get_upper(
    const BlockDesc<Ti> *blocks,
    const SVDGroup_OTF<Ti, Tv> *__restrict__ groups,
    int64 num_blocks,
    int64 num_groups,
    int &max_count,
    int &max_rank)
{
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_b > max_count)
            max_count = blocks[i].num_b;
    }

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
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE void compute_phases_direct_aos(
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank)
{
    if constexpr (Rank == 1)
    {
        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[i] = pt0;
        }
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv *p1 = p0 + max_count;

        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            Tv pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }
            p0[i] = pt0;
            p1[i] = pt1;
        }
    }
    else
    {
        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv *pbn = p0 + i * rank;
            for (uint16 r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;
                Tv ptn = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(str & zs[k]) & 1;
                    ptn += parity ? -wr[k] : wr[k];
                }
                pbn[r] = ptn;
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE int compute_phases_indirect_aos(
    Ti x,
    const int *idx_map,
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    int *__restrict__ src_idxs,
    int *__restrict__ dst_idxs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank,
    bool is_same_block, bool enforce_upper_triangle)
{
    int count = 0;
    for (int i = 0; i < num_strs; ++i)
    {
        Ti dst_str = strs[i];
        Ti src_str = dst_str ^ x;
        int src_idx = idx_map[src_str];

        if (src_idx == -1)
            continue;

        if (is_same_block && enforce_upper_triangle && src_idx < i)
            continue;

        src_idxs[count] = src_idx;
        dst_idxs[count] = i;

        if constexpr (Rank == 1)
        {
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[count] = pt0;
        }
        else if constexpr (Rank == 2)
        {
            const Tv *w1 = w0 + num_zs;

            Tv pt0 = {}, pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }

            p0[count] = pt0;
            p0[count + max_count] = pt1;
        }
        else
        {
            Tv *pbn = p0 + i * rank;
            for (uint16 r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;
                Tv ptn = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(src_str & zs[k]) & 1;
                    ptn += parity ? -wr[k] : wr[k];
                }
                pbn[r] = ptn;
            }
        }
        count++;
    }

    return count;
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE void compute_a_phase_impl(
    uint16 rank, Ti str_a, int num_za, const Ti *zas, const Tv *wa0,
    Tv &pa0, Tv &pa1, Tv *pan)
{
    if constexpr (Rank == 1)
    {
        for (int k = 0; k < num_za; ++k)
        {
            const bool parity = std::popcount(str_a & zas[k]) & 1;
            pa0 += parity ? -wa0[k] : wa0[k];
        }
    }
    else if constexpr (Rank == 2)
    {
        const Tv *wa1 = wa0 + num_za;
        for (int k = 0; k < num_za; ++k)
        {
            const bool parity = std::popcount(str_a & zas[k]) & 1;
            pa0 += parity ? -wa0[k] : wa0[k];
            pa1 += parity ? -wa1[k] : wa1[k];
        }
    }
    else
    {
        for (uint16 r = 0; r < rank; ++r)
        {
            const Tv *war = wa0 + r * num_za;
            Tv ptn = {};
            for (int k = 0; k < num_za; ++k)
            {
                const bool parity = std::popcount(str_a & zas[k]) & 1;
                ptn += parity ? -war[k] : war[k];
            }
            pan[r] = ptn;
        }
    }
}

template <int Rank, typename Tv>
FORCE_INLINE void update_vec_direct_aos(
    int b_count, int max_b_count, uint16 rank,
    Tv pa0, Tv pa1, const Tv *pan,
    const Tv *pb0, const Tv *pb1,
    const Tv *src,
    Tv *dst)
{
#pragma omp simd
    for (int b = 0; b < b_count; ++b)
    {
        Tv vt = {};
        if constexpr (Rank == 1)
        {
            vt = pa0 * pb0[b];
        }
        else if constexpr (Rank == 2)
        {
            vt = pa0 * pb0[b] + pa1 * pb1[b];
        }
        else
        {
            const Tv *pbn = pb0 + b * rank;
            for (uint16 r = 0; r < rank; ++r)
            {
                vt += pan[r] * pbn[r];
            }
        }
        dst[b] += src[b] * vt;
    }
}

template <int Rank, typename Tv>
FORCE_INLINE void update_vec_indirect_aos(
    int b_count, int max_b_count, uint16 rank,
    Tv pa0, Tv pa1, const Tv *pan,
    const Tv *pb0, const Tv *pb1,
    const int *src_b, const int *dst_b,
    const Tv *src,
    Tv *dst)
{
#pragma omp simd
    for (int b = 0; b < b_count; ++b)
    {
        Tv vt = {};
        if constexpr (Rank == 1)
        {
            vt = pa0 * pb0[b];
        }
        else if constexpr (Rank == 2)
        {
            vt = pa0 * pb0[b] + pa1 * pb1[b];
        }
        else
        {
            const Tv *pbn = pb0 + b * rank;
            for (uint16 r = 0; r < rank; ++r)
            {
                vt += pan[r] * pbn[r];
            }
        }
        dst[dst_b[b]] += src[src_b[b]] * vt;
    }
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE void compute_phases_direct_soa(
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank)
{
    if constexpr (Rank == 1)
    {
        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[i] = pt0;
        }
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv *p1 = p0 + max_count;

        for (int i = 0; i < num_strs; ++i)
        {
            const Ti str = strs[i];
            Tv pt0 = {};
            Tv pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }
            p0[i] = pt0;
            p1[i] = pt1;
        }
    }
    else
    {
        for (uint16 r = 0; r < rank; ++r)
        {
            const Tv *wr = w0 + r * num_zs;
            Tv *pr = p0 + r * max_count;

            for (int i = 0; i < num_strs; ++i)
            {
                const Ti str = strs[i];
                Tv ptn = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(str & zs[k]) & 1;
                    ptn += parity ? -wr[k] : wr[k];
                }
                pr[i] = ptn;
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
FORCE_INLINE int compute_phases_indirect_soa(
    Ti x,
    const int *idx_map,
    const Ti *__restrict__ strs, int num_strs,
    const Tv *__restrict__ w0,
    const Ti *__restrict__ zs, int num_zs,
    int *__restrict__ src_idxs,
    int *__restrict__ dst_idxs,
    Tv *__restrict__ p0, int max_count,
    uint16 rank,
    bool is_same_block, bool enforce_upper_triangle)
{
    int count = 0;
    for (int i = 0; i < num_strs; ++i)
    {
        Ti dst_str = strs[i];
        Ti src_str = dst_str ^ x;
        int src_idx = idx_map[src_str];

        if (src_idx == -1)
            continue;

        if (is_same_block && enforce_upper_triangle && src_idx < i)
            continue;

        src_idxs[count] = src_idx;
        dst_idxs[count] = i;

        if constexpr (Rank == 1)
        {
            Tv pt0 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            p0[count] = pt0;
        }
        else if constexpr (Rank == 2)
        {
            const Tv *w1 = w0 + num_zs;

            Tv pt0 = {}, pt1 = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(src_str & zs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }

            p0[count] = pt0;
            p0[count + max_count] = pt1;
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;

                Tv ptr = {};
                for (int k = 0; k < num_zs; ++k)
                {
                    const bool parity = std::popcount(src_str & zs[k]) & 1;
                    ptr += parity ? -wr[k] : wr[k];
                }

                p0[count + r * max_count] = ptr;
            }
        }
        count++;
    }

    return count;
}

template <int Rank, typename Tv>
FORCE_INLINE Tv compute_coeff_soa(
    int a, int b,
    const Tv *__restrict__ pa,
    const Tv *__restrict__ pb,
    int max_a_count, int max_b_count, uint16 rank)
{
    Tv vt = {};
    if constexpr (Rank == 1)
    {
        vt = pa[a] * pb[b];
    }
    else if constexpr (Rank == 2)
    {
        vt = pa[a] * pb[b] + pa[a + max_a_count] * pb[b + max_b_count];
    }
    else
    {
        for (uint16 r = 0; r < rank; ++r)
        {
            vt += pa[a + r * max_a_count] * pb[b + r * max_b_count];
        }
    }

    return vt;
}
