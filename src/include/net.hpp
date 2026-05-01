#pragma once
#include "common.hpp"
#include <vector>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

template <typename Tv>
FORCE_INLINE Tv fast_diag_exp(const Tv& vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        // 实数域（分子体系）：对角线严格为 0，exp(0) == 1.0
        // 为了极致性能，编译器遇到 type=double 会直接把整个内层循环优化为 1.0！
        return static_cast<Tv>(1.0); 
    }
    else
    {
        // 复数域（周期性体系）：对角线必定是纯虚数，直接提取虚部
        double val = vt.imag() * theta;
        
        // 使用实数的 cos 和 sin，彻底避开昂贵的 __cexp 库函数调用
        return Tv(std::cos(val), std::sin(val));
    }
}

// 同样的，给梯度也准备一个极速导数版本
template <typename Tv>
FORCE_INLINE Tv fast_diag_grad(const Tv& vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        // 实数域下导数也必定为 0
        return static_cast<Tv>(0.0);
    }
    else
    {
        // 导数公式: dU = v_t * exp(v_t * theta)
        double val = vt.imag() * theta;
        Tv u(std::cos(val), std::sin(val));
        return vt * u;
    }
}

template <typename Ti,
          typename Tv>
struct TransR1
{
    Ti src_idx;
    Ti dst_idx;
    Tv w0;
};

template <typename Ti,
          typename Tv>
struct TransR2
{
    Ti src_idx;
    Ti dst_idx;
    Tv w0, w1;
};

template <typename Ti>
struct TransRN
{
    Ti src_idx;
    Ti dst_idx;
    uint64 w_offset;
};

struct PureRoute
{
    uint64 jump_offset;
    uint64 phase_offset;
    uint32 n;
    uint16 block_src_idx;
    uint16 block_dst_idx;
};

struct MixedRoute
{
    uint64 a_jump_offset;
    uint64 b_jump_offset;
    uint32 na;
    uint32 nb;
    uint16 block_src_idx;
    uint16 block_dst_idx;
};

template <typename Ti,
          typename Tv>
struct GroupArena
{
    TransR1<Ti, Tv> *r1_jumps;
    uint64 num_r1_jumps;
    Tv *r1_phases;
    uint64 num_r1_phases;

    TransR2<Ti, Tv> *r2_jumps;
    uint64 num_r2_jumps;
    Tv *r2_phases;
    uint64 num_r2_phases;

    TransRN<Ti> *rn_jumps;
    uint64 num_rn_jumps;
    Tv *rn_weights;
    uint64 num_rn_weights;
    Tv *rn_phases;
    uint64 num_rn_phases;
};

template <typename Ti,
          typename Tv>
struct SVDNetwork
{
    Ti *azs;
    Ti *bzs;
    Tv *cs;
    uint64 *gs;
    uint64 ngs;

    uint8 *excit_types;
    uint16 *group_ranks;

    GroupArena<Ti, Tv> *arenas;

    PureRoute **pure_a_routes;
    uint64 *num_pure_a_routes;
    PureRoute **pure_b_routes;
    uint64 *num_pure_b_routes;
    MixedRoute **mixed_routes;
    uint64 *num_mixed_routes;
};

template <typename Ti,
          typename Tv>
struct TempArena
{
    std::vector<TransR1<Ti, Tv>> r1_jumps;
    std::vector<Tv> r1_phases;

    std::vector<TransR2<Ti, Tv>> r2_jumps;
    std::vector<Tv> r2_phases;

    std::vector<TransRN<Ti>> rn_jumps;
    std::vector<Tv> rn_weights;
    std::vector<Tv> rn_phases;

    std::vector<PureRoute> pure_routes;
    std::vector<MixedRoute> mixed_routes;

    void rollback_jumps(int rank, uint64 j_size, uint64 w_size)
    {
        if (rank == 1)
            r1_jumps.resize(j_size);
        else if (rank == 2)
            r2_jumps.resize(j_size);
        else
        {
            rn_jumps.resize(j_size);
            rn_weights.resize(w_size);
        }
    }
};

#define PURE_A_IDX(blk, idx, b) ((blk).offset + (int64)(idx) * (blk).num_b + (b))
#define PURE_B_IDX(blk, a, idx) ((blk).offset + (int64)(a) * (blk).num_b + (idx))
#define MIXED_IDX(blk, a_idx, b_idx) ((blk).offset + (int64)(a_idx) * (blk).num_b + (b_idx))

template <typename Ti, typename Tv>
FORCE_INLINE Tv calc_w_r1(
    Ti str, int64 len, const Tv *w, const Ti *z)
{
    Tv res = {};
    for (int64 k = 0; k < len; ++k)
    {
        const bool parity = std::popcount(z[k] & str) & 1;
        res += parity ? -w[k] : w[k];
    }
    return res;
}

template <typename Ti, typename Tv>
FORCE_INLINE void calc_w_r2(
    Ti str, int64 len, const Tv *w, const Ti *z, Tv &w0, Tv &w1)
{
    w0 = {};
    w1 = {};
    for (int64 k = 0; k < len; ++k)
    {
        const bool parity = std::popcount(z[k] & str) & 1;
        w0 += parity ? -w[k] : w[k];
        w1 += parity ? -w[k + len] : w[k + len];
    }
}

template <typename Ti, typename Tv>
FORCE_INLINE void push_w_rn(
    Ti str, int64 len, int rank, const Tv *w, const Ti *z, std::vector<Tv> &out_vec)
{
    for (int r = 0; r < rank; ++r)
    {
        Tv res = {};
        const Tv *wr = w + r * len;
        for (int64 k = 0; k < len; ++k)
        {
            const bool parity = std::popcount(z[k] & str) & 1;
            res += parity ? -wr[k] : wr[k];
        }
        out_vec.push_back(res);
    }
}

template <typename Ti, typename Tv>
FORCE_INLINE void append_jump_temp(
    TempArena<Ti, Tv> &temp, int rank, Ti src_idx, Ti dst_idx, Ti str, int64 len,
    const Tv *w, const Ti *z)
{
    if (rank == 1)
    {
        temp.r1_jumps.push_back({src_idx, dst_idx, calc_w_r1<Ti, Tv>(str, len, w, z)});
    }
    else if (rank == 2)
    {
        Tv w0, w1;
        calc_w_r2<Ti, Tv>(str, len, w, z, w0, w1);
        temp.r2_jumps.push_back({src_idx, dst_idx, w0, w1});
    }
    else
    {
        const uint64 w_offset = temp.rn_weights.size();
        push_w_rn<Ti, Tv>(str, len, rank, w, z, temp.rn_weights);
        temp.rn_jumps.push_back({src_idx, dst_idx, w_offset});
    }
}

template <typename Ti, typename Tv>
FORCE_INLINE void append_phase_temp(
    TempArena<Ti, Tv> &temp, int rank, Ti str, int64 len, const Tv *w, const Ti *z)
{
    if (rank == 1)
    {
        temp.r1_phases.push_back(calc_w_r1<Ti, Tv>(str, len, w, z));
    }
    else if (rank == 2)
    {
        Tv p0, p1;
        calc_w_r2<Ti, Tv>(str, len, w, z, p0, p1);
        temp.r2_phases.push_back(p0);
        temp.r2_phases.push_back(p1);
    }
    else
    {
        push_w_rn<Ti, Tv>(str, len, rank, w, z, temp.rn_phases);
    }
}

template <typename Ti, typename Tv>
void get_diagonal_elements_svd_network(
    const BasisManager<Ti> *__restrict__ basis,
    const SVDNetwork<Ti, Tv> *__restrict__ net,
    Tv *__restrict__ diags)
{
    const Ti *azs = net->azs;
    const Ti *bzs = net->bzs;
    const Tv *cs = net->cs;
    const uint64 *gs = net->gs;
    const uint8 *types = net->excit_types;

    for (uint64 g = 0; g < net->ngs; ++g)
    {
        if (types[g] == 0)
        {
            const uint64 lb = gs[g];
            const uint64 rb = gs[g + 1];
#pragma omp parallel
            for (int64 i = 0; i < basis->num_blocks; ++i)
            {
                const BlockDesc<Ti> &block = basis->blocks[i];
#pragma omp for schedule(guided)
                for (int64 a = 0; a < block.num_a; ++a)
                {
                    const Ti astr = block.astrs[a];
                    const int64 row_ptr = block.offset + a * block.num_b;
                    for (int64 b = 0; b < block.num_b; ++b)
                    {
                        const Ti bstr = block.bstrs[b];

                        Tv vt = {};
                        for (uint64 k = lb; k < rb; ++k)
                        {
                            const bool parity = (std::popcount(azs[k] & astr) ^ std::popcount(bzs[k] & bstr)) & 1;
                            vt += parity ? -cs[k] : cs[k];
                        }

                        diags[row_ptr + b] += vt;
                    }
                }
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void apply_diag_terms(
    const BasisManager<Ti> *__restrict__ basis,
    const Ti *__restrict__ azs,
    const Ti *__restrict__ bzs,
    const Tv *__restrict__ cs,
    const uint64 n_terms,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
#pragma omp parallel
    {
        bool *parity_a = new bool[n_terms];

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            const BlockDesc<Ti> &block = basis->blocks[i];
            const int64 num_b = block.num_b;
#pragma omp for schedule(guided) nowait
            for (int64 a = 0; a < block.num_a; ++a)
            {
                const Ti astr = block.astrs[a];
                const int64 row_ptr = block.offset + a * num_b;

                for (uint64 k = 0; k < n_terms; ++k)
                {
                    parity_a[k] = std::popcount(azs[k] & astr) & 1;
                }

                for (int64 b = 0; b < num_b; ++b)
                {
                    const Ti bstr = block.bstrs[b];
                    const int64 gid = row_ptr + b;

                    Tv vt = {};
                    for (uint64 k = 0; k < n_terms; ++k)
                    {
                        const bool parity_b = std::popcount(bzs[k] & bstr) & 1;
                        const bool parity = parity_a[k] ^ parity_b;
                        vt += parity ? -cs[k] : cs[k];
                    }

                    dst[gid] += src[gid] * vt;
                }
            }
        }

        delete[] parity_a;
    }
}
