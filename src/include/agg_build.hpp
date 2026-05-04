#pragma once
#include "agg.hpp"
#include "net_build.hpp"

template <typename Ti,
          typename Tv>
struct FlatPureA_R1
{
    Ti dst_a;
    uint32 g;
    uint64 src_ptr;
    Tv pa0;
    const Tv *b_phases;
    bool operator<(const FlatPureA_R1 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

template <typename Ti,
          typename Tv>
struct FlatPureA_R2
{
    Ti dst_a;
    uint32 g;
    uint64 src_ptr;
    Tv pa0, pa1;
    const Tv *b_phases;
    bool operator<(const FlatPureA_R2 &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

template <typename Ti,
          typename Tv>
struct FlatPureA_RN
{
    Ti dst_a;
    uint32 g;
    uint64 src_ptr;
    const Tv *pa_weights;
    const Tv *b_phases;
    uint32 rank;
    bool operator<(const FlatPureA_RN &o) const
    {
        if (dst_a != o.dst_a)
            return dst_a < o.dst_a;
        return g < o.g;
    }
};

template <typename Ti,
          typename Tv>
struct FlatPureB_R1
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *a_phases;
    Tv pb0;
    Ti src_b;
    Ti dst_b;
    bool operator<(const FlatPureB_R1 &o) const { return g < o.g; }
};

template <typename Ti,
          typename Tv>
struct FlatPureB_R2
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *a_phases;
    Tv pb0, pb1;
    Ti src_b;
    Ti dst_b;
    bool operator<(const FlatPureB_R2 &o) const { return g < o.g; }
};

template <typename Ti,
          typename Tv>
struct FlatPureB_RN
{
    uint32 g;
    uint64 src_offset;
    uint64 src_num_b;
    const Tv *pb_weights;
    const Tv *a_phases;
    Ti src_b;
    Ti dst_b;
    uint32 rank;
    bool operator<(const FlatPureB_RN &o) const { return g < o.g; }
};

template <typename Ti,
          typename Tv>
struct FlatMixed_R1
{
    Ti dst_a;
    uint32 g;
    int64 src_ptr;
    Tv pa0;
    const TransR1<Ti, Tv> *b_jumps;
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

template <typename Ti,
          typename Tv>
struct FlatMixed_R2
{
    Ti dst_a;
    uint32 g;
    int64 src_ptr;
    Tv pa0, pa1;
    const TransR2<Ti, Tv> *b_jumps;
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

template <typename Ti,
          typename Tv>
struct FlatMixed_RN
{
    Ti dst_a;
    uint32 g;
    int64 src_ptr;
    const Tv *pa_weights;
    const TransRN<Ti> *b_jumps;
    const Tv *b_weights_base;
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

template <typename Ti,
          typename Tv>
struct TempBlockEdges
{
    std::vector<FlatPureA_R1<Ti, Tv>> pa_r1;
    std::vector<FlatPureA_R2<Ti, Tv>> pa_r2;
    std::vector<FlatPureA_RN<Ti, Tv>> pa_rn;
    std::vector<FlatPureB_R1<Ti, Tv>> pb_r1;
    std::vector<FlatPureB_R2<Ti, Tv>> pb_r2;
    std::vector<FlatPureB_RN<Ti, Tv>> pb_rn;
    std::vector<FlatMixed_R1<Ti, Tv>> mx_r1;
    std::vector<FlatMixed_R2<Ti, Tv>> mx_r2;
    std::vector<FlatMixed_RN<Ti, Tv>> mx_rn;
};

template <typename Ti,
          typename Tv>
FORCE_INLINE const TransR1<Ti, Tv> *emit_rev_b(
    int64 g, const TransR1<Ti, Tv> *fwd, uint32 nb,
    AggSVDNetwork<Ti, Tv> *agg, uint32 &rev_idx)
{
    TransR1<Ti, Tv> *start = agg->mixed_b_rev_r1[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, math_conj(fwd[i].w0)};
    rev_idx += nb;
    return start;
}

template <typename Ti,
          typename Tv>
FORCE_INLINE const TransR2<Ti, Tv> *emit_rev_b(
    int64 g, const TransR2<Ti, Tv> *fwd, uint32 nb,
    AggSVDNetwork<Ti, Tv> *agg, uint32 &rev_idx)
{
    TransR2<Ti, Tv> *start = agg->mixed_b_rev_r2[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, math_conj(fwd[i].w0), math_conj(fwd[i].w1)};
    rev_idx += nb;
    return start;
}

template <typename Ti,
          typename Tv>
FORCE_INLINE const TransRN<Ti> *emit_rev_b(
    int64 g, const TransRN<Ti> *fwd, uint32 nb,
    AggSVDNetwork<Ti, Tv> *agg, uint32 &rev_idx, uint64 base_w_size)
{
    constexpr bool is_cplx = !std::is_arithmetic_v<Tv>;
    uint64 offset = is_cplx ? base_w_size : 0;

    TransRN<Ti> *start = agg->mixed_b_rev_rn[g] + rev_idx;
    for (uint32 i = 0; i < nb; ++i)
        start[i] = {fwd[i].dst_idx, fwd[i].src_idx, fwd[i].w_offset + offset};
    rev_idx += nb;
    return start;
}

template <typename Ti,
          typename Tv>
void build_flat_pure_a(
    int64 g, int64 rank, int64 axsym,
    const BasisManager<Ti> *basis,
    const TempArena<Ti, Tv> &temp,
    const GroupArena<Ti, Tv> &arena,
    TempBlockEdges<Ti, Tv> *temp_blocks)
{
    constexpr bool is_cplx = !std::is_arithmetic_v<Tv>;

    for (const PureRoute &route : temp.pure_routes)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[route.block_src_idx];
        int64 bid = (block_src.asym ^ axsym) * basis->num_irreps + block_src.bsym;
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[block_dst_idx];
        uint32 g_id = static_cast<uint32>(g);

        if (rank == 1)
        {
            const Tv *b_phases = arena.r1_phases + route.phase_offset;
            const Tv *b_phases_rev = b_phases + (is_cplx ? temp.r1_phases.size() : 0);
            const TransR1<Ti, Tv> *jumps = arena.r1_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].pa_r1.push_back(
                    {jumps[k].dst_idx, g_id, src_row, jumps[k].w0, b_phases});
                temp_blocks[route.block_src_idx].pa_r1.push_back(
                    {jumps[k].src_idx, g_id, dst_row, math_conj(jumps[k].w0), b_phases_rev});
            }
        }
        else if (rank == 2)
        {
            const Tv *b_phases = arena.r2_phases + route.phase_offset;
            const Tv *b_phases_rev = b_phases + (is_cplx ? temp.r2_phases.size() : 0);
            const TransR2<Ti, Tv> *jumps = arena.r2_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].pa_r2.push_back(
                    {jumps[k].dst_idx, g_id, src_row, jumps[k].w0, jumps[k].w1, b_phases});
                temp_blocks[route.block_src_idx].pa_r2.push_back(
                    {jumps[k].src_idx, g_id, dst_row, math_conj(jumps[k].w0), math_conj(jumps[k].w1), b_phases_rev});
            }
        }
        else
        {
            const Tv *b_phases = arena.rn_phases + route.phase_offset;
            const Tv *b_phases_rev = b_phases + (is_cplx ? temp.rn_phases.size() : 0);
            const TransRN<Ti> *jumps = arena.rn_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                uint64 src_row = block_src.offset + jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + jumps[k].dst_idx * block_dst.num_b;
                const Tv *pa_w = arena.rn_weights + jumps[k].w_offset;
                const Tv *pa_w_rev = pa_w + (is_cplx ? temp.rn_weights.size() : 0);

                temp_blocks[block_dst_idx].pa_rn.push_back(
                    {jumps[k].dst_idx, g_id, src_row, pa_w, b_phases, static_cast<uint32>(rank)});
                temp_blocks[route.block_src_idx].pa_rn.push_back(
                    {jumps[k].src_idx, g_id, dst_row, pa_w_rev, b_phases_rev, static_cast<uint32>(rank)});
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void build_flat_pure_b(
    int64 g, int64 rank, int64 bxsym,
    const BasisManager<Ti> *basis,
    const TempArena<Ti, Tv> &temp,
    const GroupArena<Ti, Tv> &arena,
    TempBlockEdges<Ti, Tv> *temp_blocks)
{
    constexpr bool is_cplx = !std::is_arithmetic_v<Tv>;

    for (const PureRoute &route : temp.pure_routes)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[route.block_src_idx];
        int64 bid = block_src.asym * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[block_dst_idx];
        uint32 g_id = static_cast<uint32>(g);

        if (rank == 1)
        {
            const Tv *a_phases = arena.r1_phases + route.phase_offset;
            const Tv *a_phases_rev = a_phases + (is_cplx ? temp.r1_phases.size() : 0);
            const TransR1<Ti, Tv> *jumps = arena.r1_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                temp_blocks[block_dst_idx].pb_r1.push_back(
                    {g_id, static_cast<uint64>(block_src.offset), static_cast<uint64>(block_src.num_b),
                     a_phases, jumps[k].w0, jumps[k].src_idx, jumps[k].dst_idx});
                temp_blocks[route.block_src_idx].pb_r1.push_back(
                    {g_id, static_cast<uint64>(block_dst.offset), static_cast<uint64>(block_dst.num_b),
                     a_phases_rev, math_conj(jumps[k].w0), jumps[k].dst_idx, jumps[k].src_idx});
            }
        }
        else if (rank == 2)
        {
            const Tv *a_phases = arena.r2_phases + route.phase_offset;
            const Tv *a_phases_rev = a_phases + (is_cplx ? temp.r2_phases.size() : 0);
            const TransR2<Ti, Tv> *jumps = arena.r2_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                temp_blocks[block_dst_idx].pb_r2.push_back(
                    {g_id, static_cast<uint64>(block_src.offset), static_cast<uint64>(block_src.num_b),
                     a_phases, jumps[k].w0, jumps[k].w1, jumps[k].src_idx, jumps[k].dst_idx});
                temp_blocks[route.block_src_idx].pb_r2.push_back(
                    {g_id, static_cast<uint64>(block_dst.offset), static_cast<uint64>(block_dst.num_b),
                     a_phases_rev, math_conj(jumps[k].w0), math_conj(jumps[k].w1), jumps[k].dst_idx, jumps[k].src_idx});
            }
        }
        else
        {
            const Tv *a_phases = arena.rn_phases + route.phase_offset;
            const Tv *a_phases_rev = a_phases + (is_cplx ? temp.rn_phases.size() : 0);
            const TransRN<Ti> *jumps = arena.rn_jumps + route.jump_offset;

            for (uint32 k = 0; k < route.n; ++k)
            {
                const Tv *pb_w = arena.rn_weights + jumps[k].w_offset;
                const Tv *pb_w_rev = pb_w + (is_cplx ? temp.rn_weights.size() : 0);

                temp_blocks[block_dst_idx].pb_rn.push_back(
                    {g_id, static_cast<uint64>(block_src.offset), static_cast<uint64>(block_src.num_b),
                     pb_w, a_phases, jumps[k].src_idx, jumps[k].dst_idx, static_cast<uint32>(rank)});
                temp_blocks[route.block_src_idx].pb_rn.push_back(
                    {g_id, static_cast<uint64>(block_dst.offset), static_cast<uint64>(block_dst.num_b),
                     pb_w_rev, a_phases_rev, jumps[k].dst_idx, jumps[k].src_idx, static_cast<uint32>(rank)});
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void build_flat_mixed(
    int64 g, int64 rank, int type, int64 axsym, int64 bxsym,
    const BasisManager<Ti> *basis,
    const TempArena<Ti, Tv> &temp,
    const GroupArena<Ti, Tv> &arena,
    TempBlockEdges<Ti, Tv> *temp_blocks,
    AggSVDNetwork<Ti, Tv> *agg)
{
    constexpr bool is_cplx = !std::is_arithmetic_v<Tv>;
    uint32 rev_r1_idx = 0, rev_r2_idx = 0, rev_rn_idx = 0;
    uint32 g_id = static_cast<uint32>(g);

    for (const MixedRoute &route : temp.mixed_routes)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[route.block_src_idx];
        int64 bid = (block_src.asym ^ axsym) * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 block_dst_idx = basis->block_map[bid];
        if (block_dst_idx == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[block_dst_idx];

        if (rank == 1)
        {
            const TransR1<Ti, Tv> *a_jumps = arena.r1_jumps + route.a_jump_offset;
            const TransR1<Ti, Tv> *b_fwd = arena.r1_jumps + route.b_jump_offset;
            // 【核心修改】：type == 0 时不需要反向边，直接赋 nullptr
            const TransR1<Ti, Tv> *b_rev = (type != 0) ? emit_rev_b<Ti, Tv>(
                                                             g, b_fwd, route.nb, agg, rev_r1_idx)
                                                       : nullptr;

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].mx_r1.push_back(
                    {a_jumps[k].dst_idx,
                     g_id,
                     static_cast<int64>(src_row),
                     a_jumps[k].w0,
                     b_fwd,
                     route.nb});

                // 【核心修改】：如果不是对角项，才添加反向跳跃边
                if (type != 0)
                {
                    temp_blocks[route.block_src_idx].mx_r1.push_back(
                        {a_jumps[k].src_idx,
                         g_id,
                         static_cast<int64>(dst_row),
                         math_conj(a_jumps[k].w0),
                         b_rev,
                         route.nb});
                }
            }
        }
        else if (rank == 2)
        {
            const TransR2<Ti, Tv> *a_jumps = arena.r2_jumps + route.a_jump_offset;
            const TransR2<Ti, Tv> *b_fwd = arena.r2_jumps + route.b_jump_offset;
            const TransR2<Ti, Tv> *b_rev = (type != 0) ? emit_rev_b<Ti, Tv>(
                                                             g, b_fwd, route.nb, agg, rev_r2_idx)
                                                       : nullptr;

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;

                temp_blocks[block_dst_idx].mx_r2.push_back(
                    {a_jumps[k].dst_idx,
                     g_id,
                     static_cast<int64>(src_row),
                     a_jumps[k].w0,
                     a_jumps[k].w1,
                     b_fwd,
                     route.nb});

                if (type != 0)
                {
                    temp_blocks[route.block_src_idx].mx_r2.push_back(
                        {a_jumps[k].src_idx,
                         g_id,
                         static_cast<int64>(dst_row),
                         math_conj(a_jumps[k].w0),
                         math_conj(a_jumps[k].w1),
                         b_rev,
                         route.nb});
                }
            }
        }
        else
        {
            const TransRN<Ti> *a_jumps = arena.rn_jumps + route.a_jump_offset;
            const TransRN<Ti> *b_fwd = arena.rn_jumps + route.b_jump_offset;
            const TransRN<Ti> *b_rev = (type != 0) ? emit_rev_b<Ti, Tv>(
                                                         g,
                                                         b_fwd,
                                                         route.nb,
                                                         agg,
                                                         rev_rn_idx,
                                                         temp.rn_weights.size())
                                                   : nullptr;

            for (uint32 k = 0; k < route.na; ++k)
            {
                uint64 src_row = block_src.offset + a_jumps[k].src_idx * block_src.num_b;
                uint64 dst_row = block_dst.offset + a_jumps[k].dst_idx * block_dst.num_b;

                const Tv *pa_w = arena.rn_weights + a_jumps[k].w_offset;
                const Tv *pa_w_rev = pa_w + (is_cplx ? temp.rn_weights.size() : 0);

                temp_blocks[block_dst_idx].mx_rn.push_back(
                    {a_jumps[k].dst_idx,
                     g_id,
                     static_cast<int64>(src_row),
                     pa_w,
                     b_fwd,
                     arena.rn_weights,
                     route.nb,
                     static_cast<uint32>(rank)});

                if (type != 0)
                {
                    temp_blocks[route.block_src_idx].mx_rn.push_back(
                        {a_jumps[k].src_idx,
                         g_id,
                         static_cast<int64>(dst_row),
                         pa_w_rev,
                         b_rev,
                         arena.rn_weights,
                         route.nb,
                         static_cast<uint32>(rank)});
                }
            }
        }
    }
}

template <typename Tv,
          typename T_Flat,
          typename T_Edge>
void compress_pure_a(
    std::vector<T_Flat> &flat_edges,
    uint32 num_a,
    uint32 *&offsets,
    T_Edge *&edges_out)
{
    std::sort(flat_edges.begin(), flat_edges.end());

    size_t M = flat_edges.size();
    offsets = new uint32[num_a + 1]();
    edges_out = M ? new T_Edge[M] : nullptr;

    for (size_t i = 0; i < M; ++i)
    {
        offsets[flat_edges[i].dst_a + 1]++;
        if constexpr (std::is_same_v<T_Edge, PureA_Edge_R1<Tv>>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa0,
                flat_edges[i].b_phases};
        else if constexpr (std::is_same_v<T_Edge, PureA_Edge_R2<Tv>>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa0,
                flat_edges[i].pa1,
                flat_edges[i].b_phases};
        else if constexpr (std::is_same_v<T_Edge, PureA_Edge_RN<Tv>>)
            edges_out[i] = {
                flat_edges[i].src_ptr,
                flat_edges[i].pa_weights,
                flat_edges[i].b_phases,
                flat_edges[i].rank, 0};
    }
    for (uint32 a = 0; a < num_a; ++a)
        offsets[a + 1] += offsets[a];
}

template <typename Ti,
          typename Tv,
          typename T_Flat,
          typename T_Edge>
void compress_pure_b(
    std::vector<T_Flat> &flat_edges,
    uint32 &num_edges,
    T_Edge *&edges_out)
{
    std::sort(flat_edges.begin(), flat_edges.end());

    size_t M = flat_edges.size();
    num_edges = static_cast<uint32>(M);
    edges_out = M ? new T_Edge[M] : nullptr;
    for (size_t i = 0; i < M; ++i)
    {
        if constexpr (std::is_same_v<T_Edge, PureB_Edge_R1<Ti, Tv>>)
            edges_out[i] = {
                flat_edges[i].src_offset,
                flat_edges[i].src_num_b,
                flat_edges[i].a_phases,
                flat_edges[i].pb0,
                flat_edges[i].src_b,
                flat_edges[i].dst_b};
        else if constexpr (std::is_same_v<T_Edge, PureB_Edge_R2<Ti, Tv>>)
            edges_out[i] = {
                flat_edges[i].src_offset,
                flat_edges[i].src_num_b,
                flat_edges[i].a_phases,
                flat_edges[i].pb0,
                flat_edges[i].pb1,
                flat_edges[i].src_b,
                flat_edges[i].dst_b};
        else if constexpr (std::is_same_v<T_Edge, PureB_Edge_RN<Ti, Tv>>)
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

template <typename Ti,
          typename Tv,
          typename T_Flat,
          typename T_Node,
          typename T_Leaf>
void compress_mixed_csr(
    std::vector<T_Flat> &flat_edges,
    uint32 num_a,
    uint32 *&offsets,
    T_Node *&nodes,
    T_Leaf *&leaves)
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

        if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_R1<Ti, Tv>>)
            leaves[i] = {
                flat_edges[i].pa0,
                flat_edges[i].b_jumps,
                flat_edges[i].num_b_jumps,
                0u};
        else if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_R2<Ti, Tv>>)
            leaves[i] = {
                flat_edges[i].pa0,
                flat_edges[i].pa1,
                flat_edges[i].b_jumps,
                flat_edges[i].num_b_jumps,
                0u};
        else if constexpr (std::is_same_v<T_Leaf, Mixed_G_Leaf_RN<Ti, Tv>>)
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

template <typename Ti,
          typename Tv>
void build_all_agg_edges(
    int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_as,
    const int64 *num_bs,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb,
    const BasisManager<Ti> *basis,
    AggSVDNetwork<Ti, Tv> *agg)
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
    std::vector<std::vector<TempBlockEdges<Ti, Tv>>> thread_tb(num_threads, std::vector<TempBlockEdges<Ti, Tv>>(agg->num_blocks));

#pragma omp parallel for schedule(dynamic)
    for (int64 g = 0; g < ngs; ++g)
    {
        int tid = omp_get_thread_num();
        Ti ax = axs[g], bx = bxs[g];
        int64 rank = ranks[g];

        int type = 0;
        if (ax != 0 && bx == 0)
            type = 1;
        else if (ax == 0 && bx != 0)
            type = 2;
        else if (ax != 0 && bx != 0)
            type = 3;

        agg->excit_types[g] = type;

        int64 axsym = get_string_sym(ax, basis->orbsym);
        int64 bxsym = get_string_sym(bx, basis->orbsym);

        TempArena<Ti, Tv> temp;

        switch (type)
        {
        case 1:
            build_pure_a<Ti, Tv>(
                g, ax, rank, num_as[g], num_bs[g],
                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                flat_azs, flat_bzs, flat_wa, flat_wb, 
                basis, temp);
            break;
        case 2:
            build_pure_b<Ti, Tv>(g, bx, rank, num_as[g], num_bs[g],
                                 off_az[g], off_bz[g], off_wa[g], off_wb[g],
                                 flat_azs, flat_bzs, flat_wa, flat_wb,
                                 basis, temp);
            break;
        case 0:
        case 3:
            build_mixed<Ti, Tv>(g, ax, bx, rank, num_as[g], num_bs[g],
                                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                                flat_azs, flat_bzs, flat_wa, flat_wb,
                                basis, temp);
            break;
        default:
            break;
        }

        GroupArena<Ti, Tv> &arena = agg->arenas[g];
        constexpr bool is_cplx = !std::is_arithmetic_v<Tv>;

        // R1
        arena.num_r1_jumps = temp.r1_jumps.size();
        arena.r1_jumps = arena.num_r1_jumps ? new TransR1<Ti, Tv>[arena.num_r1_jumps] : nullptr;
        if (arena.num_r1_jumps)
            std::copy(temp.r1_jumps.begin(), temp.r1_jumps.end(), arena.r1_jumps);

        arena.num_r1_phases = temp.r1_phases.size() * (is_cplx ? 2 : 1);
        arena.r1_phases = arena.num_r1_phases ? new Tv[arena.num_r1_phases] : nullptr;
        if (!temp.r1_phases.empty())
        {
            std::copy(temp.r1_phases.begin(), temp.r1_phases.end(), arena.r1_phases);
            if constexpr (is_cplx)
            {
                for (size_t k = 0; k < temp.r1_phases.size(); ++k)
                    arena.r1_phases[temp.r1_phases.size() + k] = math_conj(temp.r1_phases[k]);
            }
        }

        // R2
        arena.num_r2_jumps = temp.r2_jumps.size();
        arena.r2_jumps = arena.num_r2_jumps ? new TransR2<Ti, Tv>[arena.num_r2_jumps] : nullptr;
        if (arena.num_r2_jumps)
            std::copy(temp.r2_jumps.begin(), temp.r2_jumps.end(), arena.r2_jumps);

        arena.num_r2_phases = temp.r2_phases.size() * (is_cplx ? 2 : 1);
        arena.r2_phases = arena.num_r2_phases ? new Tv[arena.num_r2_phases] : nullptr;
        if (!temp.r2_phases.empty())
        {
            std::copy(temp.r2_phases.begin(), temp.r2_phases.end(), arena.r2_phases);
            if constexpr (is_cplx)
            {
                for (size_t k = 0; k < temp.r2_phases.size(); ++k)
                    arena.r2_phases[temp.r2_phases.size() + k] = math_conj(temp.r2_phases[k]);
            }
        }

        // RN
        arena.num_rn_jumps = temp.rn_jumps.size();
        arena.rn_jumps = arena.num_rn_jumps ? new TransRN<Ti>[arena.num_rn_jumps] : nullptr;
        if (arena.num_rn_jumps)
            std::copy(temp.rn_jumps.begin(), temp.rn_jumps.end(), arena.rn_jumps);

        arena.num_rn_weights = temp.rn_weights.size() * (is_cplx ? 2 : 1);
        arena.rn_weights = arena.num_rn_weights ? new Tv[arena.num_rn_weights] : nullptr;
        if (!temp.rn_weights.empty())
        {
            std::copy(temp.rn_weights.begin(), temp.rn_weights.end(), arena.rn_weights);
            if constexpr (is_cplx)
            {
                for (size_t k = 0; k < temp.rn_weights.size(); ++k)
                    arena.rn_weights[temp.rn_weights.size() + k] = math_conj(temp.rn_weights[k]);
            }
        }

        arena.num_rn_phases = temp.rn_phases.size() * (is_cplx ? 2 : 1);
        arena.rn_phases = arena.num_rn_phases ? new Tv[arena.num_rn_phases] : nullptr;
        if (!temp.rn_phases.empty())
        {
            std::copy(temp.rn_phases.begin(), temp.rn_phases.end(), arena.rn_phases);
            if constexpr (is_cplx)
            {
                for (size_t k = 0; k < temp.rn_phases.size(); ++k)
                    arena.rn_phases[temp.rn_phases.size() + k] = math_conj(temp.rn_phases[k]);
            }
        }

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
                agg->mixed_b_rev_r1[g] = new TransR1<Ti, Tv>[n_r1];
            if (n_r2)
                agg->mixed_b_rev_r2[g] = new TransR2<Ti, Tv>[n_r2];
            if (n_rn)
                agg->mixed_b_rev_rn[g] = new TransRN<Ti>[n_rn];
        }

        TempBlockEdges<Ti, Tv> *local_tb = thread_tb[tid].data();

        switch (type)
        {
        case 1:
            build_flat_pure_a<Ti, Tv>(g, rank, axsym, basis, temp, arena, local_tb);
            break;
        case 2:
            build_flat_pure_b<Ti, Tv>(g, rank, bxsym, basis, temp, arena, local_tb);
            break;
        case 0:
        case 3:
            build_flat_mixed<Ti, Tv>(g, rank, type, axsym, bxsym, basis, temp, arena, local_tb, agg);
            break;
        default:
            break;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < agg->num_blocks; ++i)
    {
        AggBlock<Ti, Tv> &ab = agg->blocks[i];
        TempBlockEdges<Ti, Tv> combined_tb;

        size_t total_pa_r1 = 0, total_pa_r2 = 0, total_pa_rn = 0;
        size_t total_pb_r1 = 0, total_pb_r2 = 0, total_pb_rn = 0;
        size_t total_mx_r1 = 0, total_mx_r2 = 0, total_mx_rn = 0;

        for (int t = 0; t < num_threads; ++t)
        {
            TempBlockEdges<Ti, Tv> &ttb = thread_tb[t][i];

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
            TempBlockEdges<Ti, Tv> &ttb = thread_tb[t][i];

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

        compress_pure_a<Tv>(combined_tb.pa_r1, ab.num_a, ab.pure_a_r1_offsets, ab.pure_a_r1_edges);
        compress_pure_a<Tv>(combined_tb.pa_r2, ab.num_a, ab.pure_a_r2_offsets, ab.pure_a_r2_edges);
        compress_pure_a<Tv>(combined_tb.pa_rn, ab.num_a, ab.pure_a_rn_offsets, ab.pure_a_rn_edges);

        compress_pure_b<Ti, Tv>(combined_tb.pb_r1, ab.num_pure_b_r1, ab.pure_b_r1_edges);
        compress_pure_b<Ti, Tv>(combined_tb.pb_r2, ab.num_pure_b_r2, ab.pure_b_r2_edges);
        compress_pure_b<Ti, Tv>(combined_tb.pb_rn, ab.num_pure_b_rn, ab.pure_b_rn_edges);

        compress_mixed_csr<Ti, Tv>(combined_tb.mx_r1, ab.num_a, ab.mixed_r1_ax_offsets, ab.mixed_r1_ax_nodes, ab.mixed_r1_g_leaves);
        compress_mixed_csr<Ti, Tv>(combined_tb.mx_r2, ab.num_a, ab.mixed_r2_ax_offsets, ab.mixed_r2_ax_nodes, ab.mixed_r2_g_leaves);
        compress_mixed_csr<Ti, Tv>(combined_tb.mx_rn, ab.num_a, ab.mixed_rn_ax_offsets, ab.mixed_rn_ax_nodes, ab.mixed_rn_g_leaves);
    }
}

template <typename Ti,
          typename Tv>
void *build_direct_agg_network(
    const BasisManager<Ti> *basis,
    int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_as,
    const int64 *num_bs,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb)
{
    AggSVDNetwork<Ti, Tv> *agg = new AggSVDNetwork<Ti, Tv>();

    agg->ngs = ngs;
    agg->excit_types = new uint8[ngs]();
    agg->arenas = new GroupArena<Ti, Tv>[ngs]();
    agg->mixed_b_rev_r1 = new TransR1<Ti, Tv> *[ngs]();
    agg->mixed_b_rev_r2 = new TransR2<Ti, Tv> *[ngs]();
    agg->mixed_b_rev_rn = new TransRN<Ti> *[ngs]();

    agg->num_blocks = basis->num_blocks;
    agg->blocks = new AggBlock<Ti, Tv>[agg->num_blocks]();

    for (int64 i = 0; i < agg->num_blocks; ++i)
    {
        agg->blocks[i].num_a = basis->blocks[i].num_a;
        agg->blocks[i].num_b = basis->blocks[i].num_b;
        agg->blocks[i].offset_dst = basis->blocks[i].offset;
    }

    build_all_agg_edges<Ti, Tv>(ngs, axs, bxs, ranks, num_as, num_bs,
                                flat_azs, flat_bzs, flat_wa, flat_wb,
                                basis, agg);

    return static_cast<void *>(agg);
}

template <typename Ti,
          typename Tv>
void destroy_direct_agg_network(AggSVDNetwork<Ti, Tv> *agg)
{
    if (agg->blocks)
    {
        for (uint64 i = 0; i < agg->num_blocks; ++i)
        {
            AggBlock<Ti, Tv> &b = agg->blocks[i];
            delete[] b.pure_a_r1_offsets;
            delete[] b.pure_a_r1_edges;
            delete[] b.pure_a_r2_offsets;
            delete[] b.pure_a_r2_edges;
            delete[] b.pure_a_rn_offsets;
            delete[] b.pure_a_rn_edges;

            delete[] b.pure_b_r1_edges;
            delete[] b.pure_b_r2_edges;
            delete[] b.pure_b_rn_edges;

            delete[] b.mixed_r1_ax_offsets;
            delete[] b.mixed_r1_ax_nodes;
            delete[] b.mixed_r1_g_leaves;
            delete[] b.mixed_r2_ax_offsets;
            delete[] b.mixed_r2_ax_nodes;
            delete[] b.mixed_r2_g_leaves;
            delete[] b.mixed_rn_ax_offsets;
            delete[] b.mixed_rn_ax_nodes;
            delete[] b.mixed_rn_g_leaves;
        }
        delete[] agg->blocks;
    }

    if (agg->arenas)
    {
        for (uint64 g = 0; g < agg->ngs; ++g)
        {
            GroupArena<Ti, Tv> &a = agg->arenas[g];
            delete[] a.r1_jumps;
            delete[] a.r1_phases;
            delete[] a.r2_jumps;
            delete[] a.r2_phases;
            delete[] a.rn_jumps;
            delete[] a.rn_weights;
            delete[] a.rn_phases;
        }
        delete[] agg->arenas;
    }

    if (agg->mixed_b_rev_r1)
    {
        for (uint64 g = 0; g < agg->ngs; ++g)
            delete[] agg->mixed_b_rev_r1[g];
        delete[] agg->mixed_b_rev_r1;
    }
    if (agg->mixed_b_rev_r2)
    {
        for (uint64 g = 0; g < agg->ngs; ++g)
            delete[] agg->mixed_b_rev_r2[g];
        delete[] agg->mixed_b_rev_r2;
    }
    if (agg->mixed_b_rev_rn)
    {
        for (uint64 g = 0; g < agg->ngs; ++g)
            delete[] agg->mixed_b_rev_rn[g];
        delete[] agg->mixed_b_rev_rn;
    }

    delete[] agg->excit_types;
    delete agg;
}

template <typename Ti,
          typename Tv>
void print_agg_network_info(const AggSVDNetwork<Ti, Tv> *agg)
{
    uint64_t c_pa_r1 = 0, c_pa_r2 = 0, c_pa_rn = 0;
    uint64_t c_pb_r1 = 0, c_pb_r2 = 0, c_pb_rn = 0;
    uint64_t c_mx_r1 = 0, c_mx_r2 = 0, c_mx_rn = 0;

    size_t mem_blocks = agg->num_blocks * sizeof(AggBlock<Ti, Tv>);

    for (uint64_t i = 0; i < agg->num_blocks; ++i)
    {
        const AggBlock<Ti, Tv> &b = agg->blocks[i];

        c_pb_r1 += b.num_pure_b_r1;
        mem_blocks += b.num_pure_b_r1 * sizeof(PureB_Edge_R1<Ti, Tv>);
        c_pb_r2 += b.num_pure_b_r2;
        mem_blocks += b.num_pure_b_r2 * sizeof(PureB_Edge_R2<Ti, Tv>);
        c_pb_rn += b.num_pure_b_rn;
        mem_blocks += b.num_pure_b_rn * sizeof(PureB_Edge_RN<Ti, Tv>);

        mem_blocks += (b.num_a + 1) * sizeof(uint32) * 9;

        if (b.pure_a_r1_offsets)
        {
            uint32 n = b.pure_a_r1_offsets[b.num_a];
            c_pa_r1 += n;
            mem_blocks += n * sizeof(PureA_Edge_R1<Tv>);
        }
        if (b.pure_a_r2_offsets)
        {
            uint32 n = b.pure_a_r2_offsets[b.num_a];
            c_pa_r2 += n;
            mem_blocks += n * sizeof(PureA_Edge_R2<Tv>);
        }
        if (b.pure_a_rn_offsets)
        {
            uint32 n = b.pure_a_rn_offsets[b.num_a];
            c_pa_rn += n;
            mem_blocks += n * sizeof(PureA_Edge_RN<Tv>);
        }

        if (b.mixed_r1_ax_offsets)
        {
            uint32 n_nodes = b.mixed_r1_ax_offsets[b.num_a];
            uint32 n_leaves = n_nodes ? (b.mixed_r1_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_r1_ax_nodes[n_nodes - 1].num_leaves) : 0;
            c_mx_r1 += n_leaves;
            mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_R1<Ti, Tv>);
        }
        if (b.mixed_r2_ax_offsets)
        {
            uint32 n_nodes = b.mixed_r2_ax_offsets[b.num_a];
            uint32 n_leaves = n_nodes ? (b.mixed_r2_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_r2_ax_nodes[n_nodes - 1].num_leaves) : 0;
            c_mx_r2 += n_leaves;
            mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_R2<Ti, Tv>);
        }
        if (b.mixed_rn_ax_offsets)
        {
            uint32 n_nodes = b.mixed_rn_ax_offsets[b.num_a];
            uint32 n_leaves = n_nodes ? (b.mixed_rn_ax_nodes[n_nodes - 1].leaf_offset + b.mixed_rn_ax_nodes[n_nodes - 1].num_leaves) : 0;
            c_mx_rn += n_leaves;
            mem_blocks += n_nodes * sizeof(Mixed_Ax_Node) + n_leaves * sizeof(Mixed_G_Leaf_RN<Ti, Tv>);
        }
    }

    size_t mem_arenas = agg->ngs * sizeof(GroupArena<Ti, Tv>);
    size_t mem_rev = agg->ngs * sizeof(TransR1<Ti, Tv> *) * 3;

    for (uint64 g = 0; g < agg->ngs; ++g)
    {
        const GroupArena<Ti, Tv> &a = agg->arenas[g];
        mem_arenas += a.num_r1_jumps * sizeof(TransR1<Ti, Tv>) + a.num_r1_phases * sizeof(Tv) +
                      a.num_r2_jumps * sizeof(TransR2<Ti, Tv>) + a.num_r2_phases * sizeof(Tv) +
                      a.num_rn_jumps * sizeof(TransRN<Ti>) + a.num_rn_weights * sizeof(Tv) + a.num_rn_phases * sizeof(Tv);
    }

    size_t mem_base = sizeof(AggSVDNetwork<Ti, Tv>) + agg->ngs * sizeof(uint8);
    size_t mem_total = mem_arenas + mem_blocks + mem_rev + mem_base;

    double gb = 1024.0 * 1024.0 * 1024.0;
    double mb = 1024.0 * 1024.0;

    std::cout << "\n===========================================================\n";
    std::cout << "     [ AGG Network Memory & Topology Report(CSR version)]  \n";
    std::cout << "===========================================================\n";
    std::cout << "[1] General Information\n";
    std::cout << "  - Number of Groups (ngs) : " << agg->ngs << "\n";
    std::cout << "  - Number of Symm Blocks  : " << agg->num_blocks << "\n\n";

    std::cout << "[2] Edge Topology (Valid Edges Count)\n";
    std::cout << "  - Pure A Edges : R1(" << c_pa_r1 << "), R2(" << c_pa_r2 << "), RN(" << c_pa_rn << ")\n";
    std::cout << "  - Pure B Edges : R1(" << c_pb_r1 << "), R2(" << c_pb_r2 << "), RN(" << c_pb_rn << ")\n";
    std::cout << "  - Mixed Edges  : R1(" << c_mx_r1 << "), R2(" << c_mx_r2 << "), RN(" << c_mx_rn << ")\n\n";

    std::cout << "[3] Memory Footprint (Absolute Raw Pointer Cost)\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  - SVD Arenas (Weights/Phases) : " << (mem_arenas / gb) << " GB\n";
    std::cout << "  - Agg Blocks (CSR Edges)      : " << (mem_blocks / gb) << " GB\n";
    std::cout << "  - Mixed Rev Jumps Pointers    : " << (mem_rev / mb) << " MB\n";
    std::cout << "  - Base Struct Overhead        : " << (mem_base / mb) << " MB\n";
    std::cout << "  ---------------------------------------------------------\n";
    std::cout << "  - TOTAL PHYSICAL MEMORY       : " << (mem_total / gb) << " GB\n";
    std::cout << "===========================================================\n\n";
}
