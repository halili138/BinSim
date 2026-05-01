#pragma once
#include "net.hpp"

template <typename Ti,
          typename Tv>
static void hvec_diag_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR1<Ti, Tv> *aj = arena.r1_jumps + R.a_jump_offset;
        const TransR1<Ti, Tv> *bj = arena.r1_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR1<Ti, Tv> &ja = aj[ia];
                const TransR1<Ti, Tv> &jb = bj[ib];
                const Tv vt = ja.w0 * jb.w0;
                const int64 idx = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                dst[idx] += src[idx] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_diag_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR2<Ti, Tv> *aj = arena.r2_jumps + R.a_jump_offset;
        const TransR2<Ti, Tv> *bj = arena.r2_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR2<Ti, Tv> &ja = aj[ia];
                const TransR2<Ti, Tv> &jb = bj[ib];
                const Tv vt = ja.w0 * jb.w0 + ja.w1 * jb.w1;
                const int64 idx = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                dst[idx] += src[idx] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_diag_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const Tv *weights = arena.rn_weights;
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransRN<Ti> *aj = arena.rn_jumps + R.a_jump_offset;
        const TransRN<Ti> *bj = arena.rn_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransRN<Ti> &ja = aj[ia];
                const TransRN<Ti> &jb = bj[ib];
                const Tv *wa = weights + ja.w_offset;
                const Tv *wb = weights + jb.w_offset;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const int64 idx = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                dst[idx] += src[idx] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_a_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR1<Ti, Tv> *__restrict__ jumps = arena.r1_jumps + R.jump_offset;
        const Tv *__restrict__ phases = arena.r1_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransR1<Ti, Tv> &j = jumps[ia];
                const Tv vt = j.w0 * phases[b];
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_a_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR2<Ti, Tv> *__restrict__ jumps = arena.r2_jumps + R.jump_offset;
        const Tv *__restrict__ phases = arena.r2_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransR2<Ti, Tv> &j = jumps[ia];
                const Tv *pb = phases + b * 2;
                const Tv vt = j.w0 * pb[0] + j.w1 * pb[1];
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_a_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransRN<Ti> *__restrict__ jumps = arena.rn_jumps + R.jump_offset;
        const Tv *__restrict__ weights = arena.rn_weights;
        const Tv *__restrict__ phases = arena.rn_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransRN<Ti> &j = jumps[ia];
                const Tv *__restrict__ w = weights + j.w_offset;
                const Tv *__restrict__ pb = phases + b * rank;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += w[r] * pb[r];
                }
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_b_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR1<Ti, Tv> *__restrict__ jumps = arena.r1_jumps + R.jump_offset;
        const Tv *__restrict__ phases = arena.r1_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR1<Ti, Tv> &j = jumps[ib];
                const Tv vt = phases[a] * j.w0;
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_b_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR2<Ti, Tv> *__restrict__ jumps = arena.r2_jumps + R.jump_offset;
        const Tv *__restrict__ phases = arena.r2_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR2<Ti, Tv> &j = jumps[ib];
                const Tv *pa = phases + a * 2;
                const Tv vt = pa[0] * j.w0 + pa[1] * j.w1;
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_b_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransRN<Ti> *__restrict__ jumps = arena.rn_jumps + R.jump_offset;
        const Tv *__restrict__ weights = arena.rn_weights;
        const Tv *__restrict__ phases = arena.rn_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransRN<Ti> &j = jumps[ib];
                const Tv *__restrict__ w = weights + j.w_offset;
                const Tv *__restrict__ pa = phases + a * rank;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += pa[r] * w[r];
                }
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_mixed_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR1<Ti, Tv> *aj = arena.r1_jumps + R.a_jump_offset;
        const TransR1<Ti, Tv> *bj = arena.r1_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR1<Ti, Tv> &ja = aj[ia];
                const TransR1<Ti, Tv> &jb = bj[ib];
                const Tv vt = ja.w0 * jb.w0;
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_mixed_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR2<Ti, Tv> *aj = arena.r2_jumps + R.a_jump_offset;
        const TransR2<Ti, Tv> *bj = arena.r2_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR2<Ti, Tv> &ja = aj[ia];
                const TransR2<Ti, Tv> &jb = bj[ib];
                const Tv vt = ja.w0 * jb.w0 + ja.w1 * jb.w1;
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_mixed_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const Tv *weights = arena.rn_weights;
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransRN<Ti> *aj = arena.rn_jumps + R.a_jump_offset;
        const TransRN<Ti> *bj = arena.rn_jumps + R.b_jump_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransRN<Ti> &ja = aj[ia];
                const TransRN<Ti> &jb = bj[ib];
                const Tv *wa = weights + ja.w_offset;
                const Tv *wb = weights + jb.w_offset;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
static void hvec_diag(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    if (!num_routes)
        return;

    switch (rank)
    {
    case 1:
        hvec_diag_r1<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_diag_r2<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_diag_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_a(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    if (num_routes == 0)
        return;

    switch (rank)
    {
    case 1:
        hvec_pure_a_r1<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_pure_a_r2<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_pure_a_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    }
}

template <typename Ti,
          typename Tv>
static void hvec_pure_b(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    if (num_routes == 0)
        return;

    switch (rank)
    {
    case 1:
        hvec_pure_b_r1<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_pure_b_r2<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_pure_b_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    }
}

template <typename Ti,
          typename Tv>
static void hvec_mixed(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    if (!num_routes)
        return;

    switch (rank)
    {
    case 1:
        hvec_mixed_r1<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_mixed_r2<Ti, Tv>(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_mixed_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    }
}

template <typename Ti,
          typename Tv>
void hvec_svd_network(
    const BasisManager<Ti> *__restrict__ basis,
    const SVDNetwork<Ti, Tv> *__restrict__ net,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const Ti *azs = net->azs;
    const Ti *bzs = net->bzs;
    const Tv *cs = net->cs;
    const uint64 *gs = net->gs;
    const uint8 *types = net->excit_types;

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst[i] = {};
    }

    for (uint64 g = 0; g < net->ngs; ++g)
    {
        const uint8 type = types[g];
        const uint64 lb = gs[g];
        const uint64 rb = gs[g + 1];
        const uint64 n_terms = rb - lb;

        if (type != 0 && n_terms == 0)
            continue;

        switch (type)
        {
        case 0:
            hvec_diag<Ti, Tv>(
                basis,
                net->mixed_routes[g],
                net->num_mixed_routes[g],
                net->group_ranks[g],
                net->arenas[g],
                src, dst);
            break;
        case 1:
            hvec_pure_a<Ti, Tv>(
                basis,
                net->pure_a_routes[g],
                net->num_pure_a_routes[g],
                net->group_ranks[g],
                net->arenas[g],
                src, dst);
            break;
        case 2:
            hvec_pure_b<Ti, Tv>(
                basis,
                net->pure_b_routes[g],
                net->num_pure_b_routes[g],
                net->group_ranks[g],
                net->arenas[g],
                src, dst);
            break;
        case 3:
            hvec_mixed<Ti, Tv>(
                basis,
                net->mixed_routes[g],
                net->num_mixed_routes[g],
                net->group_ranks[g],
                net->arenas[g],
                src, dst);
            break;
        default:
            std::cerr << "Error: Unexpected type = " << static_cast<int>(type)
                      << " at group g = " << g << " when hvec_svd"
                      << std::endl;
            break;
        }
    }
}
