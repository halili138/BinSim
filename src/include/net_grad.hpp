#include "net.hpp"

template <typename Ti,
          typename Tv>
Tv grad_diag_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv du = fast_diag_grad<Tv>(vt, theta);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                res += math_conj(lp[si] * du) * rp[si];
            }
        }
    }
    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_diag_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv du = fast_diag_grad<Tv>(vt, theta);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                res += math_conj(lp[si] * du) * rp[si];
            }
        }
    }
    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_diag_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv *wa = weights + ja.w_offset;
                const TransRN<Ti> &jb = bj[ib];
                const Tv *wb = weights + jb.w_offset;
                Tv vt = {};
                for (uint16 r = 0; r < rank; ++r)
                {
                    vt += wa[r] * wb[r];
                }
                const Tv du = fast_diag_grad<Tv>(vt, theta);
                const int64 si = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                res += math_conj(lp[si] * du) * rp[si];
            }
        }
    }
    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_a_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_a_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_a_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_A_IDX(blk_src, j.src_idx, ib);
                const int64 idx_dst = PURE_A_IDX(blk_dst, j.dst_idx, ib);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }
    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_b_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_b_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_pure_b_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = PURE_B_IDX(blk_src, ia, j.src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, ia, j.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_mixed_r1(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_mixed_r2(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }

    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_mixed_rn(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    const double cd = -sin(theta);
    const double co = cos(theta);
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
    const Tv *weights = arena.rn_weights;
    Tv res = {};
#pragma omp parallel reduction(+ : res)
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
                const Tv vd = cd * (vt * math_conj(vt));
                const Tv vo_fwd = co * vt;
                const Tv vo_rev = co * math_conj(vt);
                const int64 idx_src = MIXED_IDX(blk_src, ja.src_idx, jb.src_idx);
                const int64 idx_dst = MIXED_IDX(blk_dst, ja.dst_idx, jb.dst_idx);
                const Tv r0 = rp[idx_src];
                const Tv r1 = rp[idx_dst];
                const Tv out_src = r0 * vd + r1 * vo_rev;
                const Tv out_dst = r1 * vd - r0 * vo_fwd;
                res += math_conj(lp[idx_src]) * out_src +
                       math_conj(lp[idx_dst]) * out_dst;
            }
        }
    }
    return res;
}

template <typename Ti,
          typename Tv>
Tv grad_diag(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    if (!num_routes)
        return {};

    switch (rank)
    {
    case 1:
        return grad_diag_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    case 2:
        return grad_diag_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    default:
        return grad_diag_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, lp, rp);
    }
}

template <typename Ti,
          typename Tv>
Tv grad_pure_a(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    if (!num_routes)
        return {};

    switch (rank)
    {
    case 1:
        return grad_pure_a_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    case 2:
        return grad_pure_a_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    default:
        return grad_pure_a_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, lp, rp);
    }
}

template <typename Ti,
          typename Tv>
Tv grad_pure_b(
    const BasisManager<Ti> *__restrict__ basis,
    const PureRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    if (!num_routes)
        return {};

    switch (rank)
    {
    case 1:
        return grad_pure_b_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    case 2:
        return grad_pure_b_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    default:
        return grad_pure_b_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, lp, rp);
    }
}

template <typename Ti,
          typename Tv>
Tv grad_mixed(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    if (!num_routes)
        return {};

    switch (rank)
    {
    case 1:
        return grad_mixed_r1<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    case 2:
        return grad_mixed_r2<Ti, Tv>(basis, routes, num_routes, arena, theta, lp, rp);
    default:
        return grad_mixed_rn<Ti, Tv>(basis, routes, num_routes, rank, arena, theta, lp, rp);
    }
}

template <typename Ti,
          typename Tv>
Tv grad_svd_network(
    const BasisManager<Ti> *__restrict__ basis,
    const SVDNetwork<Ti, Tv> *__restrict__ net,
    const int64 idx,
    const double theta,
    const Tv *__restrict__ lp,
    const Tv *__restrict__ rp)
{
    int type = net->excit_types[idx];
    Tv res = {};

    switch (type)
    {
    case 0:
        res = grad_diag<Ti, Tv>(
            basis,
            net->mixed_routes[idx],
            net->num_mixed_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, lp, rp);
        break;
    case 1:
        res = grad_pure_a<Ti, Tv>(
            basis,
            net->pure_a_routes[idx],
            net->num_pure_a_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, lp, rp);
        break;
    case 2:
        res = grad_pure_b<Ti, Tv>(
            basis,
            net->pure_b_routes[idx],
            net->num_pure_b_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, lp, rp);
        break;
    case 3:
        res = grad_mixed<Ti, Tv>(
            basis,
            net->mixed_routes[idx],
            net->num_mixed_routes[idx],
            net->group_ranks[idx],
            net->arenas[idx],
            theta, lp, rp);
        break;
    default:
        std::cerr << "Error: Unexpected type = " << type
                  << " when grad_svd"
                  << std::endl;
        break;
    }

    return res;
}
