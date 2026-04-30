#pragma once
#include "net.hpp"
#include "agg.hpp"

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_a_r1(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const uint32 num_b,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_edge = block.pure_a_r1_offsets[a];
    const uint32 end_edge = block.pure_a_r1_offsets[a + 1];
    const PureA_Edge_R1<Tv> *__restrict__ edges_base = block.pure_a_r1_edges;

    for (uint32 e = start_edge; e < end_edge; ++e)
    {
        const PureA_Edge_R1<Tv> &edge = edges_base[e];
        const Tv pa0 = edge.pa0;
        const Tv *__restrict__ local_b_phases = edge.b_phases;
        const Tv *__restrict__ local_src = src + edge.src_ptr;
#pragma omp simd
        for (uint32 b = 0; b < num_b; ++b)
        {
            local_dst[b] += pa0 * local_b_phases[b] * local_src[b];
        }
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_a_r2(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const uint32 num_b,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_edge = block.pure_a_r2_offsets[a];
    const uint32 end_edge = block.pure_a_r2_offsets[a + 1];
    const PureA_Edge_R2<Tv> *__restrict__ edges_base = block.pure_a_r2_edges;

    for (uint32 e = start_edge; e < end_edge; ++e)
    {
        const PureA_Edge_R2<Tv> &edge = edges_base[e];
        const Tv pa0 = edge.pa0;
        const Tv pa1 = edge.pa1;
        const Tv *__restrict__ local_b_phases = edge.b_phases;
        const Tv *__restrict__ local_src = src + edge.src_ptr;
#pragma omp simd
        for (uint32 b = 0; b < num_b; ++b)
        {
            local_dst[b] += (pa0 * local_b_phases[b * 2] +
                             pa1 * local_b_phases[b * 2 + 1]) *
                            local_src[b];
        }
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_a_rn(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const uint32 num_b,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_edge = block.pure_a_rn_offsets[a];
    const uint32 end_edge = block.pure_a_rn_offsets[a + 1];
    const PureA_Edge_RN<Tv> *__restrict__ edges_base = block.pure_a_rn_edges;

    for (uint32 e = start_edge; e < end_edge; ++e)
    {
        const PureA_Edge_RN<Tv> &edge = edges_base[e];
        const uint32 rank = edge.rank;
        const Tv *__restrict__ pa_w = edge.pa_weights;
        const Tv *__restrict__ local_b_phases = edge.b_phases;
        const Tv *__restrict__ local_src = src + edge.src_ptr;
#pragma omp simd
        for (uint32 b = 0; b < num_b; ++b)
        {
            Tv p = {};
            for (uint32 r = 0; r < rank; ++r)
            {
                p += pa_w[r] * local_b_phases[b * rank + r];
            }
            local_dst[b] += p * local_src[b];
        }
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_b_r1(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 num_edges = block.num_pure_b_r1;
    const PureB_Edge_R1<Ti, Tv> *__restrict__ edges = block.pure_b_r1_edges;

    for (uint32 e = 0; e < num_edges; ++e)
    {
        const PureB_Edge_R1<Ti, Tv> &edge = edges[e];
        const Tv vt = edge.a_phases[a] * edge.pb0;
        const uint64 src_ptr = edge.src_offset + a * edge.src_num_b;

        local_dst[edge.dst_b] += vt * src[src_ptr + edge.src_b];
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_b_r2(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 a2 = a * 2;
    const uint32 num_edges = block.num_pure_b_r2;
    const PureB_Edge_R2<Ti, Tv> *__restrict__ edges = block.pure_b_r2_edges;

    for (uint32 e = 0; e < num_edges; ++e)
    {
        const PureB_Edge_R2<Ti, Tv> &edge = edges[e];
        const Tv vt = edge.a_phases[a2] * edge.pb0 + edge.a_phases[a2 + 1] * edge.pb1;
        const uint64 src_ptr = edge.src_offset + a * edge.src_num_b;

        local_dst[edge.dst_b] += vt * src[src_ptr + edge.src_b];
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_pure_b_rn(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 num_edges = block.num_pure_b_rn;
    const PureB_Edge_RN<Ti, Tv> *__restrict__ edges = block.pure_b_rn_edges;

    for (uint32 e = 0; e < num_edges; ++e)
    {
        const PureB_Edge_RN<Ti, Tv> &edge = edges[e];
        const uint32 rank = edge.rank;
        const uint32 ar = a * rank;
        const uint64 src_ptr = edge.src_offset + a * edge.src_num_b;

        Tv vt = {};
        for (uint32 r = 0; r < rank; ++r)
        {
            vt += edge.a_phases[ar + r] * edge.pb_weights[r];
        }

        local_dst[edge.dst_b] += vt * src[src_ptr + edge.src_b];
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_mixed_r1(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_node = block.mixed_r1_ax_offsets[a];
    const uint32 end_node = block.mixed_r1_ax_offsets[a + 1];
    const Mixed_Ax_Node *__restrict__ nodes_base = block.mixed_r1_ax_nodes;
    const Mixed_G_Leaf_R1<Ti, Tv> *__restrict__ edge_base = block.mixed_r1_g_leaves;

    for (uint32 n = start_node; n < end_node; ++n)
    {
        const Mixed_Ax_Node &node = nodes_base[n];
        const Tv *__restrict__ local_src = src + node.src_ptr;
        const uint32 start_leaf = node.leaf_offset;
        const uint32 end_leaf = start_leaf + node.num_leaves;

        for (uint32 l = start_leaf; l < end_leaf; ++l)
        {
            const Mixed_G_Leaf_R1<Ti, Tv> &g_edge = edge_base[l];
            const Tv pa0 = g_edge.pa0;
            const uint32 num_b = g_edge.num_b_jumps;
            const TransR1<Ti, Tv> *__restrict__ b_jumps = g_edge.b_jumps;
#pragma GCC unroll 4
            for (uint32 ib = 0; ib < num_b; ++ib)
            {
                const TransR1<Ti, Tv> &jb = b_jumps[ib];
                local_dst[jb.dst_idx] += pa0 * jb.w0 * local_src[jb.src_idx];
            }
        }
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_mixed_r2(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_node = block.mixed_r2_ax_offsets[a];
    const uint32 end_node = block.mixed_r2_ax_offsets[a + 1];
    const Mixed_Ax_Node *__restrict__ nodes_base = block.mixed_r2_ax_nodes;
    const Mixed_G_Leaf_R2<Ti, Tv> *__restrict__ edge_base = block.mixed_r2_g_leaves;

    for (uint32 n = start_node; n < end_node; ++n)
    {
        const Mixed_Ax_Node &node = nodes_base[n];
        const Tv *__restrict__ local_src = src + node.src_ptr;
        const uint32 start_leaf = node.leaf_offset;
        const uint32 end_leaf = start_leaf + node.num_leaves;

        for (uint32 l = start_leaf; l < end_leaf; ++l)
        {
            const Mixed_G_Leaf_R2<Ti, Tv> &g_edge = edge_base[l];
            const Tv pa0 = g_edge.pa0;
            const Tv pa1 = g_edge.pa1;
            const uint32 num_b = g_edge.num_b_jumps;
            const TransR2<Ti, Tv> *__restrict__ b_jumps = g_edge.b_jumps;
#pragma GCC unroll 4
            for (uint32 ib = 0; ib < num_b; ++ib)
            {
                const TransR2<Ti, Tv> &jb = b_jumps[ib];
                local_dst[jb.dst_idx] += (pa0 * jb.w0 + pa1 * jb.w1) * local_src[jb.src_idx];
            }
        }
    }
}

template <typename Ti,
          typename Tv>
FORCE_INLINE static void hvec_mixed_rn(
    const AggBlock<Ti, Tv> &block,
    const uint32 a,
    const Tv *__restrict__ src,
    Tv *__restrict__ local_dst)
{
    const uint32 start_node = block.mixed_rn_ax_offsets[a];
    const uint32 end_node = block.mixed_rn_ax_offsets[a + 1];
    const Mixed_Ax_Node *__restrict__ nodes_base = block.mixed_rn_ax_nodes;
    const Mixed_G_Leaf_RN<Ti, Tv> *__restrict__ edge_base = block.mixed_rn_g_leaves;

    for (uint32 n = start_node; n < end_node; ++n)
    {
        const Mixed_Ax_Node &node = nodes_base[n];
        const Tv *__restrict__ local_src = src + node.src_ptr;
        const uint32 start_leaf = node.leaf_offset;
        const uint32 end_leaf = start_leaf + node.num_leaves;

        for (uint32 l = start_leaf; l < end_leaf; ++l)
        {
            const Mixed_G_Leaf_RN<Ti, Tv> &g_edge = edge_base[l];
            const uint32 rank = g_edge.rank;
            const Tv *__restrict__ pa_w = g_edge.pa_weights;
            const Tv *__restrict__ pb_w_base = g_edge.b_weights_base;
            const uint32 num_b = g_edge.num_b_jumps;
            const TransRN<Ti> *__restrict__ b_jumps = g_edge.b_jumps;
#pragma GCC unroll 4
            for (uint32 ib = 0; ib < num_b; ++ib)
            {
                const TransRN<Ti> &jb = b_jumps[ib];
                const Tv *pb_w = pb_w_base + jb.w_offset;

                Tv p = {};
                for (uint32 r = 0; r < rank; ++r)
                {
                    p += pa_w[r] * pb_w[r];
                }

                local_dst[jb.dst_idx] += p * local_src[jb.src_idx];
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_aggregated_svd(
    const AggSVDNetwork<Ti, Tv> *__restrict__ agg_net,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
#pragma omp parallel
    {
        for (uint64 i = 0; i < agg_net->num_blocks; ++i)
        {
            const AggBlock<Ti, Tv> &block = agg_net->blocks[i];
            const uint32 num_a = block.num_a;
            const uint32 num_b = block.num_b;
            const uint64 offset_dst = block.offset_dst;

            if (num_a == 0 || num_b == 0)
                continue;

#pragma omp for schedule(dynamic) nowait
            for (uint32 a = 0; a < num_a; ++a)
            {
                Tv *__restrict__ local_dst = dst + offset_dst + a * num_b;

                hvec_pure_a_r1<Ti, Tv>(block, a, num_b, src, local_dst);
                hvec_pure_a_r2<Ti, Tv>(block, a, num_b, src, local_dst);
                hvec_pure_a_rn<Ti, Tv>(block, a, num_b, src, local_dst);

                hvec_pure_b_r1<Ti, Tv>(block, a, src, local_dst);
                hvec_pure_b_r2<Ti, Tv>(block, a, src, local_dst);
                hvec_pure_b_rn<Ti, Tv>(block, a, src, local_dst);

                hvec_mixed_r1<Ti, Tv>(block, a, src, local_dst);
                hvec_mixed_r2<Ti, Tv>(block, a, src, local_dst);
                hvec_mixed_rn<Ti, Tv>(block, a, src, local_dst);
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void hvec_direct_agg_network(
    const BasisManager<Ti> *basis,
    const AggSVDNetwork<Ti, Tv> *agg,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const Ti *azs = agg->azs;
    const Ti *bzs = agg->bzs;
    const Tv *cs = agg->cs;
    const uint64 *gs = agg->gs;
    const uint8 *types = agg->excit_types;

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst[i] = {};
    }

    for (uint64 g = 0; g < agg->ngs; ++g)
    {
        if (types[g] == 0)
        {
            const uint64 lb = gs[g];
            const uint64 rb = gs[g + 1];
            const uint64 n_terms = rb - lb;

            if (n_terms != 0)
            {
                apply_diag_terms<Ti, Tv>(
                    basis,
                    azs + lb, bzs + lb, cs + lb, n_terms,
                    src, dst);
            }
        }
    }

    hvec_aggregated_svd<Ti, Tv>(agg, src, dst);
}

template <typename Ti,
          typename Tv>
void hvec_direct_agg_network_benchmark(
    const BasisManager<Ti> *basis,
    const AggSVDNetwork<Ti, Tv> *agg,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst,
    int enable_likwid)
{
    if (enable_likwid)
    {
#pragma omp parallel
        {
            LIKWID_MARKER_START("hvec_direct_agg");
        }
    }

    const Ti *azs = agg->azs;
    const Ti *bzs = agg->bzs;
    const Tv *cs = agg->cs;
    const uint64 *gs = agg->gs;
    const uint8 *types = agg->excit_types;

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst[i] = Tv(0);
    }

    for (uint64 g = 0; g < agg->ngs; ++g)
    {
        if (types[g] == 0)
        {
            const uint64 lb = gs[g];
            const uint64 rb = gs[g + 1];
            const uint64 n_terms = rb - lb;

            if (n_terms != 0)
            {
                apply_diag_terms<Ti, Tv>(
                    basis,
                    azs + lb, bzs + lb, cs + lb, n_terms,
                    src, dst);
            }
        }
    }

    hvec_aggregated_svd<Ti, Tv>(agg, src, dst);

    if (enable_likwid)
    {
#pragma omp parallel
        {
            LIKWID_MARKER_STOP("hvec_direct_agg");
        }
    }
}
