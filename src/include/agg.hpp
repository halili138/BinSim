#pragma once
#include "net.hpp"
#include <iomanip>
#include <utility>
#include <likwid-marker.h>

template <typename Tv>
struct PureA_Edge_R1
{
    uint64 src_ptr;
    Tv pa0;
    const Tv *b_phases;
};

template <typename Tv>
struct PureA_Edge_R2
{
    uint64 src_ptr;
    Tv pa0, pa1;
    const Tv *b_phases;
};

template <typename Tv>
struct PureA_Edge_RN
{
    uint64 src_ptr;
    const Tv *pa_weights;
    const Tv *b_phases;
    uint32 rank;
    uint32 _padding;
};

template <typename Ti,
          typename Tv>
struct PureB_Edge_R1
{
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *a_phases;
    Tv pb0;
    Ti src_b;
    Ti dst_b;
};

template <typename Ti,
          typename Tv>
struct PureB_Edge_R2
{
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *a_phases;
    Tv pb0, pb1;
    Ti src_b;
    Ti dst_b;
};

template <typename Ti,
          typename Tv>
struct PureB_Edge_RN
{
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *pb_weights;
    const Tv *a_phases;
    Ti src_b;
    Ti dst_b;
    uint32 rank;
    uint32 _padding;
};

struct Mixed_Ax_Node
{
    int64 src_ptr;
    uint32 leaf_offset;
    uint32 num_leaves;
};

template <typename Ti,
          typename Tv>
struct Mixed_G_Leaf_R1
{
    Tv pa0;
    const TransR1<Ti, Tv> *b_jumps;
    uint32 num_b_jumps;
    uint32 _padding;
};

template <typename Ti,
          typename Tv>
struct Mixed_G_Leaf_R2
{
    Tv pa0, pa1;
    const TransR2<Ti, Tv> *b_jumps;
    uint32 num_b_jumps;
    uint32 _padding;
};

template <typename Ti,
          typename Tv>
struct Mixed_G_Leaf_RN
{
    const Tv *pa_weights;
    const TransRN<Ti> *b_jumps;
    const Tv *b_weights_base;
    uint32 num_b_jumps;
    uint32 rank;
};

template <typename Ti,
          typename Tv>
struct AggBlock
{
    uint32 num_a, num_b;
    uint64 offset_dst;

    uint32 *pure_a_r1_offsets;
    PureA_Edge_R1<Tv> *pure_a_r1_edges;
    uint32 *pure_a_r2_offsets;
    PureA_Edge_R2<Tv> *pure_a_r2_edges;
    uint32 *pure_a_rn_offsets;
    PureA_Edge_RN<Tv> *pure_a_rn_edges;

    uint32 num_pure_b_r1;
    PureB_Edge_R1<Ti, Tv> *pure_b_r1_edges;
    uint32 num_pure_b_r2;
    PureB_Edge_R2<Ti, Tv> *pure_b_r2_edges;
    uint32 num_pure_b_rn;
    PureB_Edge_RN<Ti, Tv> *pure_b_rn_edges;

    uint32 *mixed_r1_ax_offsets;
    Mixed_Ax_Node *mixed_r1_ax_nodes;
    Mixed_G_Leaf_R1<Ti, Tv> *mixed_r1_g_leaves;
    uint32 *mixed_r2_ax_offsets;
    Mixed_Ax_Node *mixed_r2_ax_nodes;
    Mixed_G_Leaf_R2<Ti, Tv> *mixed_r2_g_leaves;
    uint32 *mixed_rn_ax_offsets;
    Mixed_Ax_Node *mixed_rn_ax_nodes;
    Mixed_G_Leaf_RN<Ti, Tv> *mixed_rn_g_leaves;
};

template <typename Ti,
          typename Tv>
struct AggSVDNetwork
{
    Ti *azs;
    Ti *bzs;
    Tv *cs;
    uint64 *gs;
    uint64 ngs;
    uint8 *excit_types;

    AggBlock<Ti, Tv> *blocks;
    uint64 num_blocks;

    GroupArena<Ti, Tv> *arenas;

    TransR1<Ti, Tv> **mixed_b_rev_r1;
    TransR2<Ti, Tv> **mixed_b_rev_r2;
    TransRN<Ti> **mixed_b_rev_rn;
};

template <typename Ti,
          typename Tv>
void get_diagonal_elements_agg(
    const BasisManager<Ti> *basis,
    const AggSVDNetwork<Ti, Tv> *agg,
    Tv *__restrict__ diags)
{
    const Ti *azs = agg->azs;
    const Ti *bzs = agg->bzs;
    const Tv *cs = agg->cs;
    const uint64 *gs = agg->gs;
    const uint8 *types = agg->excit_types;

    for (uint64 g = 0; g < agg->ngs; ++g)
    {
        if (types[g] == 0)
        {
            const int64 lb = gs[g];
            const int64 rb = gs[g + 1];
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
                        for (int64 k = lb; k < rb; ++k)
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
