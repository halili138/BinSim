#include "agg.hpp"
#include <omp.h>
#include <vector>
#include <algorithm>

struct FlatPureA_R1
{
    uint32 dst_a;
    uint32 g;
    uint64 src_ptr;
    double pa0;
    const double *b_phases;
    bool operator<(const FlatPureA_R1 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

struct FlatPureA_R2
{
    uint32 dst_a;
    uint32 g;
    uint64 src_ptr;
    double pa0, pa1;
    const double *b_phases;
    bool operator<(const FlatPureA_R2 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

struct FlatPureA_RN
{
    uint32 dst_a;
    uint32 g;
    uint64 src_ptr;
    const double *pa_weights;
    const double *b_phases;
    uint32 rank;
    bool operator<(const FlatPureA_RN &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

struct FlatPureB_R1
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const double *a_phases;
    double pb0;
    uint32 src_b;
    uint32 dst_b;
    bool operator<(const FlatPureB_R1 &o) const { return g < o.g; } // 按 g 排序
};

struct FlatPureB_R2
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const double *a_phases;
    double pb0, pb1;
    uint32 src_b;
    uint32 dst_b;
    bool operator<(const FlatPureB_R2 &o) const { return g < o.g; }
};

struct FlatPureB_RN
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const double *pb_weights;
    const double *a_phases;
    uint32 src_b;
    uint32 dst_b;
    uint32 rank;
    bool operator<(const FlatPureB_RN &o) const { return g < o.g; }
};

struct FlatMixed_R1
{
    uint32 dst_a;
    uint32 g;
    int64 src_ptr;
    double pa0;
    const TransR1 *b_jumps;
    uint32 num_b_jumps;
    bool operator<(const FlatMixed_R1 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        if (src_ptr != o.src_ptr)
            return src_ptr < o.src_ptr;
        return g < o.g;
    }
};

struct FlatMixed_R2
{
    uint32 dst_a;
    uint32 g;
    int64 src_ptr;
    double pa0, pa1;
    const TransR2 *b_jumps;
    uint32 num_b_jumps;
    bool operator<(const FlatMixed_R2 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        if (src_ptr != o.src_ptr)
            return src_ptr < o.src_ptr;
        return g < o.g;
    }
};

struct FlatMixed_RN
{
    uint32 dst_a;
    uint32 g;
    int64 src_ptr;
    const double *pa_weights;
    const TransRN *b_jumps;
    const double *b_weights_base;
    uint32 num_b_jumps;
    uint32 rank;
    bool operator<(const FlatMixed_RN &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        if (src_ptr != o.src_ptr)
            return src_ptr < o.src_ptr;
        return g < o.g;
    }
};

struct TempBlockEdges
{
    std::vector<FlatPureA_R1> pa_r1;
    std::vector<FlatPureA_R2> pa_r2;
    std::vector<FlatPureA_RN> pa_rn;
    std::vector<FlatPureB_R1> pb_r1;
    std::vector<FlatPureB_R2> pb_r2;
    std::vector<FlatPureB_RN> pb_rn;
    std::vector<FlatMixed_R1> mx_r1;
    std::vector<FlatMixed_R2> mx_r2;
    std::vector<FlatMixed_RN> mx_rn;
};

FORCE_INLINE const TransR1 *emit_rev_b(int64 g, const TransR1 *fwd, uint32 nb, AggSVDNetwork *agg, uint32 &rev_idx)
{
    TransR1 *start = agg->mixed_b_rev_r1[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, fwd[i].w0};
    rev_idx += nb;
    return start;
}

FORCE_INLINE const TransR2 *emit_rev_b(int64 g, const TransR2 *fwd, uint32 nb, AggSVDNetwork *agg, uint32 &rev_idx)
{
    TransR2 *start = agg->mixed_b_rev_r2[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, fwd[i].w0, fwd[i].w1};
    rev_idx += nb;
    return start;
}

FORCE_INLINE const TransRN *emit_rev_b(int64 g, const TransRN *fwd, uint32 nb, AggSVDNetwork *agg, uint32 &rev_idx)
{
    TransRN *start = agg->mixed_b_rev_rn[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, fwd[i].w_offset};
    rev_idx += nb;
    return start;
}

void build_flat_pure_a(
    int64 g,
    int64 rank,
    int64 axsym,
    const BasisManager *basis,
    const TempArena &temp,
    const GroupArena &arena,
    TempBlockEdges *temp_blocks)
{
    for (const PureRoute &route : temp.pure_routes)
    {
        const BlockDesc &block_src = basis->blocks[route.block_src_idx];

        int64 bid = (block_src.asym ^ axsym) * basis->num_irreps + block_src.bsym;
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[block_dst_idx];
        uint32 g_id = static_cast<uint32>(g);

        if (rank == 1)
        {
            const double *b_phases = arena.r1_phases + route.phase_offset;
            const TransR1 *jumps = arena.r1_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].pa_r1.push_back(
                    {jumps[k].dst_idx, g_id, src_row, jumps[k].w0, b_phases});

                temp_blocks[route.block_src_idx].pa_r1.push_back(
                    {jumps[k].src_idx, g_id, dst_row, jumps[k].w0, b_phases});
            }
        }
        else if (rank == 2)
        {
            const double *b_phases = arena.r2_phases + route.phase_offset;
            const TransR2 *jumps = arena.r2_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].pa_r2.push_back(
                    {jumps[k].dst_idx, g_id, src_row, jumps[k].w0, jumps[k].w1, b_phases});

                temp_blocks[route.block_src_idx].pa_r2.push_back(
                    {jumps[k].src_idx, g_id, dst_row, jumps[k].w0, jumps[k].w1, b_phases});
            }
        }
        else
        {
            const double *b_phases = arena.rn_phases + route.phase_offset;
            const TransRN *jumps = arena.rn_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;
                const double *pa_w = arena.rn_weights + jumps[k].w_offset;

                temp_blocks[block_dst_idx].pa_rn.push_back(
                    {jumps[k].dst_idx, g_id, src_row, pa_w, b_phases, static_cast<uint32>(rank)});

                temp_blocks[route.block_src_idx].pa_rn.push_back(
                    {jumps[k].src_idx, g_id, dst_row, pa_w, b_phases, static_cast<uint32>(rank)});
            }
        }
    }
}

void build_flat_pure_b(
    int64 g,
    int64 rank,
    int64 bxsym,
    const BasisManager *basis,
    const TempArena &temp,
    const GroupArena &arena,
    TempBlockEdges *temp_blocks)
{
    for (const PureRoute &route : temp.pure_routes)
    {
        const BlockDesc &block_src = basis->blocks[route.block_src_idx];

        int64 bid = block_src.asym * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;
        const BlockDesc &block_dst = basis->blocks[block_dst_idx];
        uint32 g_id = static_cast<uint32>(g);

        if (rank == 1)
        {
            const double *a_phases = arena.r1_phases + route.phase_offset;
            const TransR1 *jumps = arena.r1_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                temp_blocks[block_dst_idx].pb_r1.push_back(
                    {g_id,
                     static_cast<uint64>(block_src.offset),
                     static_cast<uint64>(block_src.num_b),
                     a_phases,
                     jumps[k].w0,
                     jumps[k].src_idx,
                     jumps[k].dst_idx});

                temp_blocks[route.block_src_idx].pb_r1.push_back(
                    {g_id,
                     static_cast<uint64>(block_dst.offset),
                     static_cast<uint64>(block_dst.num_b),
                     a_phases, jumps[k].w0,
                     jumps[k].dst_idx,
                     jumps[k].src_idx});
            }
        }
        else if (rank == 2)
        {
            const double *a_phases = arena.r2_phases + route.phase_offset;
            const TransR2 *jumps = arena.r2_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                temp_blocks[block_dst_idx].pb_r2.push_back(
                    {g_id,
                     static_cast<uint64>(block_src.offset),
                     static_cast<uint64>(block_src.num_b),
                     a_phases,
                     jumps[k].w0,
                     jumps[k].w1,
                     jumps[k].src_idx,
                     jumps[k].dst_idx});

                temp_blocks[route.block_src_idx].pb_r2.push_back(
                    {g_id,
                     static_cast<uint64>(block_dst.offset),
                     static_cast<uint64>(block_dst.num_b),
                     a_phases,
                     jumps[k].w0,
                     jumps[k].w1,
                     jumps[k].dst_idx,
                     jumps[k].src_idx});
            }
        }
        else
        {
            const double *a_phases = arena.rn_phases + route.phase_offset;
            const TransRN *jumps = arena.rn_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                const double *pb_w = arena.rn_weights + jumps[k].w_offset;

                temp_blocks[block_dst_idx].pb_rn.push_back(
                    {g_id,
                     static_cast<uint64>(block_src.offset),
                     static_cast<uint64>(block_src.num_b),
                     pb_w,
                     a_phases,
                     jumps[k].src_idx,
                     jumps[k].dst_idx,
                     static_cast<uint32>(rank)});

                temp_blocks[route.block_src_idx].pb_rn.push_back(
                    {g_id,
                     static_cast<uint64>(block_dst.offset),
                     static_cast<uint64>(block_dst.num_b),
                     pb_w,
                     a_phases,
                     jumps[k].dst_idx,
                     jumps[k].src_idx,
                     static_cast<uint32>(rank)});
            }
        }
    }
}

void build_flat_mixed(
    int64 g,
    int64 rank,
    int64 axsym,
    int64 bxsym,
    const BasisManager *basis,
    const TempArena &temp,
    const GroupArena &arena,
    TempBlockEdges *temp_blocks,
    AggSVDNetwork *agg)
{
    uint32 rev_r1_idx = 0, rev_r2_idx = 0, rev_rn_idx = 0;
    uint32 g_id = static_cast<uint32>(g);

    for (const MixedRoute &route : temp.mixed_routes)
    {
        const BlockDesc &block_src = basis->blocks[route.block_src_idx];

        int64 bid = (block_src.asym ^ axsym) * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[block_dst_idx];

        if (rank == 1)
        {
            const TransR1 *a_jumps = arena.r1_jumps + route.a_jump_offset;
            const TransR1 *b_fwd = arena.r1_jumps + route.b_jump_offset;
            const TransR1 *b_rev = emit_rev_b(g, b_fwd, route.nb, agg, rev_r1_idx);

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].mx_r1.push_back(
                    {a_jumps[k].dst_idx, g_id, static_cast<int64>(src_row), a_jumps[k].w0, b_fwd, route.nb});

                temp_blocks[route.block_src_idx].mx_r1.push_back(
                    {a_jumps[k].src_idx, g_id, static_cast<int64>(dst_row), a_jumps[k].w0, b_rev, route.nb});
            }
        }
        else if (rank == 2)
        {
            const TransR2 *a_jumps = arena.r2_jumps + route.a_jump_offset;
            const TransR2 *b_fwd = arena.r2_jumps + route.b_jump_offset;
            const TransR2 *b_rev = emit_rev_b(g, b_fwd, route.nb, agg, rev_r2_idx);

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].mx_r2.push_back(
                    {a_jumps[k].dst_idx, g_id, static_cast<int64>(src_row), a_jumps[k].w0, a_jumps[k].w1, b_fwd, route.nb});

                temp_blocks[route.block_src_idx].mx_r2.push_back(
                    {a_jumps[k].src_idx, g_id, static_cast<int64>(dst_row), a_jumps[k].w0, a_jumps[k].w1, b_rev, route.nb});
            }
        }
        else
        {
            const TransRN *a_jumps = arena.rn_jumps + route.a_jump_offset;
            const TransRN *b_fwd = arena.rn_jumps + route.b_jump_offset;
            const TransRN *b_rev = emit_rev_b(g, b_fwd, route.nb, agg, rev_rn_idx);

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;
                const double *pa_w = arena.rn_weights + a_jumps[k].w_offset;

                temp_blocks[block_dst_idx].mx_rn.push_back(
                    {a_jumps[k].dst_idx,
                     g_id,
                     static_cast<int64>(src_row),
                     pa_w,
                     b_fwd,
                     arena.rn_weights,
                     route.nb,
                     static_cast<uint32>(rank)});

                temp_blocks[route.block_src_idx].mx_rn.push_back(
                    {a_jumps[k].src_idx,
                     g_id,
                     static_cast<int64>(dst_row),
                     pa_w,
                     b_rev,
                     arena.rn_weights,
                     route.nb,
                     static_cast<uint32>(rank)});
            }
        }
    }
}

template <typename T_Flat, typename T_Edge>
void compress_pure_a(std::vector<T_Flat> &flat_edges, uint32 num_a, uint32 *&offsets, T_Edge *&edges_out)
{
    std::sort(flat_edges.begin(), flat_edges.end());

    size_t M = flat_edges.size();
    offsets = new uint32[num_a + 1]();
    edges_out = M ? new T_Edge[M] : nullptr;

    for (size_t i = 0; i < M; ++i)
    {
        offsets[flat_edges[i].dst_a + 1]++;
        if constexpr (std::is_same_v<T_Edge, PureA_Edge_R1>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa0,
                flat_edges[i].b_phases};
        else if constexpr (std::is_same_v<T_Edge, PureA_Edge_R2>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa0,
                flat_edges[i].pa1,
                flat_edges[i].b_phases};
        else if constexpr (std::is_same_v<T_Edge, PureA_Edge_RN>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa_weights,
                flat_edges[i].b_phases,
                flat_edges[i].rank, 0};
    }
    for (uint32 a = 0; a < num_a; ++a)
        offsets[a + 1] += offsets[a];
}

template <typename T_Flat, typename T_Edge>
void compress_pure_b(std::vector<T_Flat> &flat_edges, uint32 &num_edges, T_Edge *&edges_out)
{
    std::sort(flat_edges.begin(), flat_edges.end());

    size_t M = flat_edges.size();
    num_edges = static_cast<uint32>(M);
    edges_out = M ? new T_Edge[M] : nullptr;
    for (size_t i = 0; i < M; ++i)
    {
        if constexpr (std::is_same_v<T_Edge, PureB_Edge_R1>)
            edges_out[i] = {
                flat_edges[i].src_offset,
                flat_edges[i].src_num_b,
                flat_edges[i].a_phases,
                flat_edges[i].pb0,
                flat_edges[i].src_b,
                flat_edges[i].dst_b};
        else if constexpr (std::is_same_v<T_Edge, PureB_Edge_R2>)
            edges_out[i] = {
                flat_edges[i].src_offset,
                flat_edges[i].src_num_b,
                flat_edges[i].a_phases,
                flat_edges[i].pb0,
                flat_edges[i].pb1,
                flat_edges[i].src_b,
                flat_edges[i].dst_b};
        else if constexpr (std::is_same_v<T_Edge, PureB_Edge_RN>)
            edges_out[i] = {
                flat_edges[i].src_offset,
                flat_edges[i].src_num_b,
                flat_edges[i].pb_weights,
                flat_edges[i].a_phases,
                flat_edges[i].src_b,
                flat_edges[i].dst_b,
                flat_edges[i].rank, 0};
    }
}

template <typename T_Flat, typename T_Node, typename T_Leaf>
void compress_mixed_csr(std::vector<T_Flat> &flat_edges, uint32 num_a, uint32 *&offsets, T_Node *&nodes, T_Leaf *&leaves)
{
    std::sort(flat_edges.begin(), flat_edges.end());
    size_t M = flat_edges.size();

    uint32 unique_nodes = 0;
    if (M > 0)
    {
        unique_nodes = 1;
        for (size_t i = 1; i < M; ++i)
        {
            if (flat_edges[i].dst_a != flat_edges[i - 1].dst_a || flat_edges[i].src_ptr != flat_edges[i - 1].src_ptr)
                unique_nodes++;
        }
    }

    offsets = new uint32[num_a + 1]();
    nodes = unique_nodes ? new T_Node[unique_nodes] : nullptr;
    leaves = M ? new T_Leaf[M] : nullptr;

    if (M == 0)
        return;

    uint32 node_idx = 0;
    for (size_t i = 0; i < M; ++i)
    {
        bool new_node = (i == 0 || flat_edges[i].dst_a != flat_edges[i - 1].dst_a || flat_edges[i].src_ptr != flat_edges[i - 1].src_ptr);

        if (new_node)
        {
            nodes[node_idx].src_ptr = flat_edges[i].src_ptr;
            nodes[node_idx].leaf_offset = static_cast<uint32>(i);
            nodes[node_idx].num_leaves = 1;
            node_idx++;
            offsets[flat_edges[i].dst_a + 1]++;
        }
        else
        {
            nodes[node_idx - 1].num_leaves++;
        }

        if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_R1>)
            leaves[i] = {
                flat_edges[i].pa0,
                flat_edges[i].b_jumps,
                flat_edges[i].num_b_jumps,
                0u};
        else if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_R2>)
            leaves[i] = {
                flat_edges[i].pa0,
                flat_edges[i].pa1,
                flat_edges[i].b_jumps,
                flat_edges[i].num_b_jumps,
                0u};
        else if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_RN>)
            leaves[i] = {
                flat_edges[i].pa_weights,
                flat_edges[i].b_jumps,
                flat_edges[i].b_weights_base,
                flat_edges[i].num_b_jumps,
                flat_edges[i].rank};
    }

    for (uint32 a = 0; a < num_a; ++a)
        offsets[a + 1] += offsets[a];
}

void build_all_agg_edges(
    int64 ngs,
    const uint32 *axs,
    const uint32 *bxs,
    const int64 *ranks,
    const int64 *num_as,
    const int64 *num_bs,
    const uint32 *flat_azs,
    const uint32 *flat_bzs,
    const double *flat_wa,
    const double *flat_wb,
    const int64 *orbsym,
    const BasisManager *basis,
    AggSVDNetwork *agg)
{
    std::vector<int64> off_az(ngs, 0), off_bz(ngs, 0), off_wa(ngs, 0), off_wb(ngs, 0);
    int64 caz = 0, cbz = 0, cwa = 0, cwb = 0;
    for (int64 g = 0; g < ngs; ++g)
    {
        off_az[g] = caz;
        off_bz[g] = cbz;
        off_wa[g] = cwa;
        off_wb[g] = cwb;
        caz += num_as[g];
        cbz += num_bs[g];
        cwa += num_as[g] * ranks[g];
        cwb += num_bs[g] * ranks[g];
    }

    int num_threads = omp_get_max_threads();

    std::vector<std::vector<TempBlockEdges>> thread_tb(num_threads, std::vector<TempBlockEdges>(agg->num_blocks));

#pragma omp parallel for schedule(dynamic)
    for (int64 g = 0; g < ngs; ++g)
    {
        int tid = omp_get_thread_num();
        uint32 ax = axs[g], bx = bxs[g];
        int64 rank = ranks[g];

        int type = 0;
        if (ax != 0 && bx == 0)
            type = 1;
        else if (ax == 0 && bx != 0)
            type = 2;
        else if (ax != 0 && bx != 0)
            type = 3;

        agg->excit_types[g] = type;

        int64 axsym = get_string_sym(ax, orbsym);
        int64 bxsym = get_string_sym(bx, orbsym);

        TempArena temp;

        switch (type)
        {
        case 1:
            build_pure_a(
                g, ax, rank, num_as[g], num_bs[g],
                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                flat_azs, flat_bzs, flat_wa, flat_wb,
                orbsym, basis, temp);
            break;
        case 2:
            build_pure_b(g, bx, rank, num_as[g], num_bs[g],
                         off_az[g], off_bz[g], off_wa[g], off_wb[g],
                         flat_azs, flat_bzs, flat_wa, flat_wb,
                         orbsym, basis, temp);
            break;
        case 3:
            build_mixed(g, ax, bx, rank, num_as[g], num_bs[g],
                        off_az[g], off_bz[g], off_wa[g], off_wb[g],
                        flat_azs, flat_bzs, flat_wa, flat_wb,
                        orbsym, basis, temp);
            break;
        default:
            break;
        }

        GroupArena &arena = agg->arenas[g];

        arena.num_r1_jumps = temp.r1_jumps.size();
        arena.r1_jumps = arena.num_r1_jumps ? new TransR1[arena.num_r1_jumps] : nullptr;
        if (arena.num_r1_jumps)
            std::copy(temp.r1_jumps.begin(), temp.r1_jumps.end(), arena.r1_jumps);

        arena.num_r1_phases = temp.r1_phases.size();
        arena.r1_phases = arena.num_r1_phases ? new double[arena.num_r1_phases] : nullptr;
        if (arena.num_r1_phases)
            std::copy(temp.r1_phases.begin(), temp.r1_phases.end(), arena.r1_phases);

        arena.num_r2_jumps = temp.r2_jumps.size();
        arena.r2_jumps = arena.num_r2_jumps ? new TransR2[arena.num_r2_jumps] : nullptr;
        if (arena.num_r2_jumps)
            std::copy(temp.r2_jumps.begin(), temp.r2_jumps.end(), arena.r2_jumps);

        arena.num_r2_phases = temp.r2_phases.size();
        arena.r2_phases = arena.num_r2_phases ? new double[arena.num_r2_phases] : nullptr;
        if (arena.num_r2_phases)
            std::copy(temp.r2_phases.begin(), temp.r2_phases.end(), arena.r2_phases);

        arena.num_rn_jumps = temp.rn_jumps.size();
        arena.rn_jumps = arena.num_rn_jumps ? new TransRN[arena.num_rn_jumps] : nullptr;
        if (arena.num_rn_jumps)
            std::copy(temp.rn_jumps.begin(), temp.rn_jumps.end(), arena.rn_jumps);

        arena.num_rn_weights = temp.rn_weights.size();
        arena.rn_weights = arena.num_rn_weights ? new double[arena.num_rn_weights] : nullptr;
        if (arena.num_rn_weights)
            std::copy(temp.rn_weights.begin(), temp.rn_weights.end(), arena.rn_weights);

        arena.num_rn_phases = temp.rn_phases.size();
        arena.rn_phases = arena.num_rn_phases ? new double[arena.num_rn_phases] : nullptr;
        if (arena.num_rn_phases)
            std::copy(temp.rn_phases.begin(), temp.rn_phases.end(), arena.rn_phases);

        if (type == 3)
        {
            uint32 n_r1 = 0, n_r2 = 0, n_rn = 0;
            for (const MixedRoute &r : temp.mixed_routes)
            {
                if (rank == 1)
                    n_r1 += r.nb;
                else if (rank == 2)
                    n_r2 += r.nb;
                else
                    n_rn += r.nb;
            }

            if (n_r1)
                agg->mixed_b_rev_r1[g] = new TransR1[n_r1];
            if (n_r2)
                agg->mixed_b_rev_r2[g] = new TransR2[n_r2];
            if (n_rn)
                agg->mixed_b_rev_rn[g] = new TransRN[n_rn];
        }

        TempBlockEdges *local_tb = thread_tb[tid].data();

        switch (type)
        {
        case 1:
            build_flat_pure_a(g, rank, axsym, basis, temp, arena, local_tb);
            break;
        case 2:
            build_flat_pure_b(g, rank, bxsym, basis, temp, arena, local_tb);
            break;
        case 3:
            build_flat_mixed(g, rank, axsym, bxsym, basis, temp, arena, local_tb, agg);
            break;
        default:
            break;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < agg->num_blocks; ++i)
    {
        AggBlock &ab = agg->blocks[i];
        TempBlockEdges combined_tb;

        size_t total_pa_r1 = 0, total_pa_r2 = 0, total_pa_rn = 0;
        size_t total_pb_r1 = 0, total_pb_r2 = 0, total_pb_rn = 0;
        size_t total_mx_r1 = 0, total_mx_r2 = 0, total_mx_rn = 0;

        for (int t = 0; t < num_threads; ++t)
        {
            TempBlockEdges &ttb = thread_tb[t][i];

            total_pa_r1 += ttb.pa_r1.size();
            total_pa_r2 += ttb.pa_r2.size();
            total_pa_rn += ttb.pa_rn.size();
            total_pb_r1 += ttb.pb_r1.size();
            total_pb_r2 += ttb.pb_r2.size();
            total_pb_rn += ttb.pb_rn.size();
            total_mx_r1 += ttb.mx_r1.size();
            total_mx_r2 += ttb.mx_r2.size();
            total_mx_rn += ttb.mx_rn.size();
        }

        combined_tb.pa_r1.reserve(total_pa_r1);
        combined_tb.pa_r2.reserve(total_pa_r2);
        combined_tb.pa_rn.reserve(total_pa_rn);
        combined_tb.pb_r1.reserve(total_pb_r1);
        combined_tb.pb_r2.reserve(total_pb_r2);
        combined_tb.pb_rn.reserve(total_pb_rn);
        combined_tb.mx_r1.reserve(total_mx_r1);
        combined_tb.mx_r2.reserve(total_mx_r2);
        combined_tb.mx_rn.reserve(total_mx_rn);

        for (int t = 0; t < num_threads; ++t)
        {
            TempBlockEdges &ttb = thread_tb[t][i];

            if (!ttb.pa_r1.empty())
                combined_tb.pa_r1.insert(combined_tb.pa_r1.end(), ttb.pa_r1.begin(), ttb.pa_r1.end());
            if (!ttb.pa_r2.empty())
                combined_tb.pa_r2.insert(combined_tb.pa_r2.end(), ttb.pa_r2.begin(), ttb.pa_r2.end());
            if (!ttb.pa_rn.empty())
                combined_tb.pa_rn.insert(combined_tb.pa_rn.end(), ttb.pa_rn.begin(), ttb.pa_rn.end());

            if (!ttb.pb_r1.empty())
                combined_tb.pb_r1.insert(combined_tb.pb_r1.end(), ttb.pb_r1.begin(), ttb.pb_r1.end());
            if (!ttb.pb_r2.empty())
                combined_tb.pb_r2.insert(combined_tb.pb_r2.end(), ttb.pb_r2.begin(), ttb.pb_r2.end());
            if (!ttb.pb_rn.empty())
                combined_tb.pb_rn.insert(combined_tb.pb_rn.end(), ttb.pb_rn.begin(), ttb.pb_rn.end());

            if (!ttb.mx_r1.empty())
                combined_tb.mx_r1.insert(combined_tb.mx_r1.end(), ttb.mx_r1.begin(), ttb.mx_r1.end());
            if (!ttb.mx_r2.empty())
                combined_tb.mx_r2.insert(combined_tb.mx_r2.end(), ttb.mx_r2.begin(), ttb.mx_r2.end());
            if (!ttb.mx_rn.empty())
                combined_tb.mx_rn.insert(combined_tb.mx_rn.end(), ttb.mx_rn.begin(), ttb.mx_rn.end());
        }

        compress_pure_a(combined_tb.pa_r1, ab.num_a, ab.pure_a_r1_offsets, ab.pure_a_r1_edges);
        compress_pure_a(combined_tb.pa_r2, ab.num_a, ab.pure_a_r2_offsets, ab.pure_a_r2_edges);
        compress_pure_a(combined_tb.pa_rn, ab.num_a, ab.pure_a_rn_offsets, ab.pure_a_rn_edges);

        compress_pure_b(combined_tb.pb_r1, ab.num_pure_b_r1, ab.pure_b_r1_edges);
        compress_pure_b(combined_tb.pb_r2, ab.num_pure_b_r2, ab.pure_b_r2_edges);
        compress_pure_b(combined_tb.pb_rn, ab.num_pure_b_rn, ab.pure_b_rn_edges);

        compress_mixed_csr(combined_tb.mx_r1, ab.num_a, ab.mixed_r1_ax_offsets, ab.mixed_r1_ax_nodes, ab.mixed_r1_g_leaves);
        compress_mixed_csr(combined_tb.mx_r2, ab.num_a, ab.mixed_r2_ax_offsets, ab.mixed_r2_ax_nodes, ab.mixed_r2_g_leaves);
        compress_mixed_csr(combined_tb.mx_rn, ab.num_a, ab.mixed_rn_ax_offsets, ab.mixed_rn_ax_nodes, ab.mixed_rn_g_leaves);
    }
}
