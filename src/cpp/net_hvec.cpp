#include "net.hpp"
#include <omp.h>

static void hvec_pure_a_r1(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR1 *__restrict__ jumps = arena.r1_jumps + R.jump_offset;
        const double *__restrict__ phases = arena.r1_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransR1 &j = jumps[ia];
                const double vt = j.w0 * phases[b];
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_pure_a_r2(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR2 *__restrict__ jumps = arena.r2_jumps + R.jump_offset;
        const double *__restrict__ phases = arena.r2_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransR2 &j = jumps[ia];
                const double *pb = phases + b * 2;
                const double vt = j.w0 * pb[0] + j.w1 * pb[1];
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_pure_a_rn(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const uint16 rank = arena.rank;
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransRN *__restrict__ jumps = arena.rn_jumps + R.jump_offset;
        const double *__restrict__ weights = arena.rn_weights;
        const double *__restrict__ phases = arena.rn_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                const TransRN &j = jumps[ia];
                const double *__restrict__ w = weights + j.w_offset;
                const double *__restrict__ pb = phases + b * rank;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += w[r] * pb[r];
                }
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, b);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_pure_b_r1(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR1 *__restrict__ jumps = arena.r1_jumps + R.jump_offset;
        const double *__restrict__ phases = arena.r1_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR1 &j = jumps[ib];
                const double vt = phases[a] * j.w0;
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_pure_b_r2(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransR2 *__restrict__ jumps = arena.r2_jumps + R.jump_offset;
        const double *__restrict__ phases = arena.r2_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR2 &j = jumps[ib];
                const double *pa = phases + a * 2;
                const double vt = pa[0] * j.w0 + pa[1] * j.w1;
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_pure_b_rn(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const uint16 rank = arena.rank;
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 ir = 0; ir < num_routes; ++ir)
    {
        const PureRoute &R = routes[ir];
        const TransRN *__restrict__ jumps = arena.rn_jumps + R.jump_offset;
        const double *__restrict__ weights = arena.rn_weights;
        const double *__restrict__ phases = arena.rn_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < blk_src.num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransRN &j = jumps[ib];
                const double *__restrict__ w = weights + j.w_offset;
                const double *__restrict__ pa = phases + a * rank;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += pa[r] * w[r];
                }
                const int64 idx_src = PURE_B_IDX(blk_src, a, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, j.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_mixed_r1(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR1 *aj = arena.r1_jumps + R.a_jump_offset;
        const TransR1 *bj = arena.r1_jumps + R.b_jump_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR1 &ja = aj[ia];
                const TransR1 &jb = bj[ib];
                const double vt = ja.w0 * jb.w0;
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_mixed_r2(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransR2 *aj = arena.r2_jumps + R.a_jump_offset;
        const TransR2 *bj = arena.r2_jumps + R.b_jump_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransR2 &ja = aj[ia];
                const TransR2 &jb = bj[ib];
                const double vt = ja.w0 * jb.w0 + ja.w1 * jb.w1;
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

static void hvec_mixed_rn(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    const uint16 rank = arena.rank;
    const double *weights = arena.rn_weights;
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const TransRN *aj = arena.rn_jumps + R.a_jump_offset;
        const TransRN *bj = arena.rn_jumps + R.b_jump_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                const TransRN &ja = aj[ia];
                const TransRN &jb = bj[ib];
                const double *wa = weights + ja.w_offset;
                const double *wb = weights + jb.w_offset;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                dst[idx_src] += src[idx_dst] * vt;
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

void apply_diag_terms(
    const BasisManager *__restrict__ basis,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
#pragma omp parallel
    {
        int *phase_a = new int[n_terms];

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            const BlockDesc &block = basis->blocks[i];
            const int64 num_b = block.num_b;

#pragma omp for schedule(guided) nowait
            for (int64 a = 0; a < block.num_a; ++a)
            {
                const uint32 astr = block.astrs[a];
                const int64 row_ptr = block.offset + a * num_b;

                for (int64 k = 0; k < n_terms; ++k)
                {
                    phase_a[k] = phase(azs[k] & astr);
                }

                for (int64 b = 0; b < num_b; ++b)
                {
                    const uint32 bstr = block.bstrs[b];
                    const int64 gid = row_ptr + b;

                    double vt = 0.0;
                    for (int64 k = 0; k < n_terms; ++k)
                    {
                        vt += cs[k] * phase_a[k] * phase(bzs[k] & bstr);
                    }

                    dst[gid] += src[gid] * vt;
                }
            }
        }

        delete[] phase_a;
    }
}

void hvec_pure_a(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    if (num_routes == 0)
        return;

    switch (arena.rank)
    {
    case 1:
        hvec_pure_a_r1(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_pure_a_r2(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_pure_a_rn(basis, routes, num_routes, arena, src, dst);
        break;
    }
}

void hvec_pure_b(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    if (num_routes == 0)
        return;

    switch (arena.rank)
    {
    case 1:
        hvec_pure_b_r1(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_pure_b_r2(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_pure_b_rn(basis, routes, num_routes, arena, src, dst);
        break;
    }
}

void hvec_mixed(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
    if (!num_routes)
        return;

    switch (arena.rank)
    {
    case 1:
        hvec_mixed_r1(basis, routes, num_routes, arena, src, dst);
        break;
    case 2:
        hvec_mixed_r2(basis, routes, num_routes, arena, src, dst);
        break;
    default:
        hvec_mixed_rn(basis, routes, num_routes, arena, src, dst);
        break;
    }
}
