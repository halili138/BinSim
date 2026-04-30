#pragma once
#include "net.hpp"

template <typename Ti,
          typename Tv>
void tvec_pure_a_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR1<Ti, Tv> *jumps = arena.r1_jumps + R.jump_offset;
        const Tv *phases = arena.r1_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransR1<Ti, Tv> &j = jumps[ia];
                const Tv vt = j.w0 * phases[ib];
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_a_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR2<Ti, Tv> *jumps = arena.r2_jumps + R.jump_offset;
        const Tv *phases = arena.r2_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransR2<Ti, Tv> &j = jumps[ia];
                const Tv *pb = phases + ib * 2;
                const Tv vt = j.w0 * pb[0] + j.w1 * pb[1];
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_a_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransRN<Ti> *jumps = arena.rn_jumps + R.jump_offset;
        const Tv *phases = arena.rn_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 nb = blk_src.num_b;
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 ib = 0; ib < nb; ++ib)
            {
                const TransRN<Ti> &j = jumps[ia];
                const Tv *w = weights + j.w_offset;
                const Tv *pb = phases + ib * rank;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += w[r] * pb[r];
                }
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 di = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_b_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR1<Ti, Tv> *jumps = arena.r1_jumps + R.jump_offset;
        const Tv *phases = arena.r1_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR1<Ti, Tv> &j = jumps[ib];
                const Tv vt = phases[ia] * j.w0;
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_b_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransR2<Ti, Tv> *jumps = arena.r2_jumps + R.jump_offset;
        const Tv *phases = arena.r2_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransR2<Ti, Tv> &j = jumps[ib];
                const Tv *pa = phases + ia * 2;
                const Tv vt = pa[0] * j.w0 + pa[1] * j.w1;
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_b_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const PureRoute &R = routes[i];
        const TransRN<Ti> *jumps = arena.rn_jumps + R.jump_offset;
        const Tv *phases = arena.rn_phases + R.phase_offset;
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 ia = 0; ia < blk_src.num_a; ++ia)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                const TransRN<Ti> &j = jumps[ib];
                const Tv *w = weights + j.w_offset;
                const Tv *pa = phases + ia * rank;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += pa[r] * w[r];
                }
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 di = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_mixed_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
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
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_mixed_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
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
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_mixed_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    const double cd = cos(theta) - 1.0;
    const double co = sin(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
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
                const Tv *wa = weights + ja.w_offset;
                const TransRN<Ti> &jb = bj[ib];
                const Tv *wb = weights + jb.w_offset;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const Tv vd = 1.0 + cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 di = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv vi = vec[si];
                const Tv vj = vec[di];
                vec[si] = vi * vd - vj * vo_rev;
                vec[di] = vj * vd + vi * vo_fwd;
            }
        }
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_a(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (rank)
    {
    case 1:
        tvec_pure_a_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_pure_a_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_pure_a_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, vec);
        break;
    }
}

template <typename Ti,
          typename Tv>
void tvec_pure_b(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (rank)
    {
    case 1:
        tvec_pure_b_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_pure_b_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_pure_b_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, vec);
        break;
    }
}

template <typename Ti,
          typename Tv>
void tvec_mixed(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    Tv *__restrict__ vec)
{
    if (!num_routes)
        return;

    switch (rank)
    {
    case 1:
        tvec_mixed_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    case 2:
        tvec_mixed_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, vec);
        break;
    default:
        tvec_mixed_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, vec);
        break;
    }
}

template <typename Ti,
          typename Tv>
void tvec_svd_network(
    const BasisManager<Ti> *__restrict__ basis,
    const SVDNetwork<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    Tv *__restrict__ vec)
{
    int type = net->excit_types[idx];

    switch (type)
    {
    case 1:
        tvec_pure_a<Ti, Tv>(
            basis,
            net->pure_a_routes[idx],
            net->num_pure_a_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, vec);
        break;
    case 2:
        tvec_pure_b<Ti, Tv>(
            basis,
            net->pure_b_routes[idx],
            net->num_pure_b_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, vec);
        break;
    case 3:
        tvec_mixed<Ti, Tv>(
            basis,
            net->mixed_routes[idx],
            net->num_mixed_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, vec);
        break;
    default:
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type)
                  << " when tvec_svd"
                  << std::endl;
        break;
    }
}
