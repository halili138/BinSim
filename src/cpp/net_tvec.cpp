#include "net.hpp"
#include <omp.h>

static void tvec_pure_a_r1(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR1 *jumps = arena.r1_jumps + R.jump_offset;
        const double *phases = arena.r1_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransR1 &j = jumps[ia];
                const double vt = j.w0 * phases[ib];
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_pure_a_r2(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR2 *jumps = arena.r2_jumps + R.jump_offset;
        const double *phases = arena.r2_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransR2 &j = jumps[ia];
                const double *pb = phases + ib * 2;
                const double vt = j.w0 * pb[0] + j.w1 * pb[1];
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_pure_a_rn(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const uint16 rank = arena.rank;
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
    const double *weights = arena.rn_weights;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransRN *jumps = arena.rn_jumps + R.jump_offset;
        const double *phases = arena.rn_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransRN &j = jumps[ia];
                const double *w = weights + j.w_offset;
                const double *pb = phases + ib * rank;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += w[r] * pb[r];
                }
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_pure_b_r1(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR1 *jumps = arena.r1_jumps + R.jump_offset;
        const double *phases = arena.r1_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR1 &j = jumps[ib];
                const double vt = phases[ia] * j.w0;
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_pure_b_r2(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR2 *jumps = arena.r2_jumps + R.jump_offset;
        const double *phases = arena.r2_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR2 &j = jumps[ib];
                const double *pa = phases + ia * 2;
                const double vt = pa[0] * j.w0 + pa[1] * j.w1;
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_pure_b_rn(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const uint16 rank = arena.rank;
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
    const double *weights = arena.rn_weights;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransRN *jumps = arena.rn_jumps + R.jump_offset;
        const double *phases = arena.rn_phases + R.phase_offset;
        const BlockDesc &blk_src = blocks[R.block_src_idx];
        const BlockDesc &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransRN &j = jumps[ib];
                const double *w = weights + j.w_offset;
                const double *pa = phases + ia * rank;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += pa[r] * w[r];
                }
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_mixed_r1(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
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
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_mixed_r2(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
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
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

static void tvec_mixed_rn(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    const uint16 rank = arena.rank;
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc *__restrict__ blocks = basis->blocks;
    const double *weights = arena.rn_weights;
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
                const double *wa = weights + ja.w_offset;
                const TransRN &jb = bj[ib];
                const double *wb = weights + jb.w_offset;
                double vt = 0.0;
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const double vd = 1.0 + cd * (vt * vt);
                const double vo = co * vt;
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const double vi = vec[si];
                const double vj = vec[di];
                vec[si] = vi * vd - vj * vo;
                vec[di] = vj * vd + vi * vo;
            }
        }
    }
}

void tvec_pure_a(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (arena.rank)
    {
    case 1:
        tvec_pure_a_r1(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_pure_a_r2(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_pure_a_rn(basis, routes, num_routes, arena, theta, vec);
        break;
    }
}

void tvec_pure_b(
    const BasisManager *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (arena.rank)
    {
    case 1:
        tvec_pure_b_r1(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_pure_b_r2(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_pure_b_rn(basis, routes, num_routes, arena, theta, vec);
        break;
    }
}

void tvec_mixed(
    const BasisManager *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena &arena,
    const double theta,
    double *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (arena.rank)
    {
    case 1:
        tvec_mixed_r1(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_mixed_r2(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_mixed_rn(basis, routes, num_routes, arena, theta, vec);
        break;
    }
}
