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

template <int Rank,
          typename Ti,
          typename Tv>
FORCE_INLINE void get_upper(
    const BlockDesc<Ti> *blocks,
    const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_blocks, int64 num_groups,
    int &max_a_count, int &max_b_count, int &max_rank)
{
    for (int64 i = 0; i < num_blocks; ++i)
    {
        if (blocks[i].num_a > max_a_count)
            max_a_count = blocks[i].num_a;

        if (blocks[i].num_b > max_b_count)
            max_b_count = blocks[i].num_b;
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

template <int Rank,
          int MemLay,
          typename Ti,
          typename Tv>
FORCE_INLINE void compute_phases(
    const Ti *strs, int num_strs,
    const Ti *zs, int num_zs,
    const Tv *w0, Tv *p0, int max_count, int rank)
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
            Tv pt0 = {}, pt1 = {};
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
        if constexpr (MemLay == 1)
        {
            for (int i = 0; i < num_strs; ++i)
            {
                const Ti str = strs[i];
                Tv *pi = p0 + i * rank;
                for (int r = 0; r < rank; ++r)
                {
                    const Tv *wr = w0 + r * num_zs;
                    Tv pt = {};
                    for (int k = 0; k < num_zs; ++k)
                    {
                        const bool parity = std::popcount(str & zs[k]) & 1;
                        pt += parity ? -wr[k] : wr[k];
                    }
                    pi[r] = pt;
                }
            }
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;
                Tv *pr = p0 + r * max_count;
                for (int i = 0; i < num_strs; ++i)
                {
                    const Ti str = strs[i];
                    Tv pt = {};
                    for (int k = 0; k < num_zs; ++k)
                    {
                        const bool parity = std::popcount(str & zs[k]) & 1;
                        pt += parity ? -wr[k] : wr[k];
                    }
                    pr[i] = pt;
                }
            }
        }
    }
}

template <int Rank,
          int MemLay,
          typename Ti,
          typename Tv>
FORCE_INLINE int compute_phases_symm(
    Ti x,
    const int *idx_map, const Ti *strs, int num_strs,
    const Ti *zs, int num_zs,
    const Tv *w0, Tv *p0, int max_count, int rank,
    int *src_idxs, int *dst_idxs,
    bool is_same_block, bool enforce_upper_triangle)
{
    int count = {};
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

        count++;
    }

    if constexpr (Rank == 1)
    {
        for (int i = 0; i < count; ++i)
        {
            const Ti str = strs[dst_idxs[i]] ^ x;
            Tv pt = {};
            for (int k = 0; k < num_zs; ++k)
            {
                const bool parity = std::popcount(str & zs[k]) & 1;
                pt += parity ? -w0[k] : w0[k];
            }
            p0[i] = pt;
        }
    }
    else if constexpr (Rank == 2)
    {
        const Tv *w1 = w0 + num_zs;
        Tv *p1 = p0 + max_count;
        for (int i = 0; i < count; ++i)
        {
            const Ti str = strs[dst_idxs[i]] ^ x;
            Tv pt0 = {}, pt1 = {};
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
        if constexpr (MemLay == 1)
        {
            for (int i = 0; i < count; ++i)
            {
                const Ti str = strs[dst_idxs[i]] ^ x;
                Tv *pi = p0 + i * rank;
                for (int r = 0; r < rank; ++r)
                {
                    const Tv *wr = w0 + r * num_zs;
                    Tv pt = {};
                    for (int k = 0; k < num_zs; ++k)
                    {
                        const bool parity = std::popcount(str & zs[k]) & 1;
                        pt += parity ? -wr[k] : wr[k];
                    }
                    pi[r] = pt;
                }
            }
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zs;
                Tv *pr = p0 + r * max_count;
                for (int i = 0; i < count; ++i)
                {
                    const Ti str = strs[dst_idxs[i]] ^ x;
                    Tv pt = {};
                    for (int k = 0; k < num_zs; ++k)
                    {
                        const bool parity = std::popcount(str & zs[k]) & 1;
                        pt += parity ? -wr[k] : wr[k];
                    }
                    pr[i] = pt;
                }
            }
        }
    }

    return count;
}

template <int Rank,
          int MemLay,
          typename Tv>
FORCE_INLINE Tv compute_coeff(
    int a, int b, const Tv *pa, const Tv *pb,
    int max_a_count, int max_b_count, int rank)
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
        if constexpr (MemLay == 1)
        {
            const Tv *pan = pa + a * rank;
            const Tv *pbn = pb + b * rank;
            for (int r = 0; r < rank; ++r)
            {
                vt += pan[r] * pbn[r];
            }
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                vt += pa[a + r * max_a_count] * pb[b + r * max_b_count];
            }
        }
    }

    return vt;
}

template <int Rank,
          typename Ti,
          typename Tv>
FORCE_INLINE void compute_a_phase(
    Ti str_a, const Ti *zas, int num_za, const Tv *wa0, int rank,
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
        for (int r = 0; r < rank; ++r)
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

template <int Rank,
          int MemLay,
          int Symm,
          typename Tv>
FORCE_INLINE void update_dst(
    int b_count, int rank,
    Tv pa0, Tv pa1, const Tv *pan,
    const Tv *pb, int max_b_count,
    const int *src_b_idx, const int *dst_b_idx,
    const Tv *src, Tv *dst)
{
#pragma omp simd
    for (int b = 0; b < b_count; ++b)
    {
        Tv vt = {};
        if constexpr (Rank == 1)
        {
            vt = pa0 * pb[b];
        }
        else if constexpr (Rank == 2)
        {
            vt = pa0 * pb[b] + pa1 * pb[b + max_b_count];
        }
        else
        {
            if constexpr (MemLay == 1)
            {
                const Tv *pbn = pb + b * rank;
                for (int r = 0; r < rank; ++r)
                {
                    vt += pan[r] * pbn[r];
                }
            }
            else
            {
                for (int r = 0; r < rank; ++r)
                {
                    vt += pan[r] * pb[b + r * max_b_count];
                }
            }
        }
        if constexpr (Symm == 1)
        {
            dst[dst_b_idx[b]] += src[src_b_idx[b]] * vt;
        }
        else
        {
            dst[b] += src[b] * vt;
        }
    }
}
