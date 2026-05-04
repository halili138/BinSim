#pragma once
#include "net.hpp"

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void hvec_diag_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];

        const TransR1<Ti, Tv> *__restrict__ aj1 = nullptr;
        const TransR1<Ti, Tv> *__restrict__ bj1 = nullptr;
        const TransR2<Ti, Tv> *__restrict__ aj2 = nullptr;
        const TransR2<Ti, Tv> *__restrict__ bj2 = nullptr;
        const TransRN<Ti> *__restrict__ ajn = nullptr;
        const TransRN<Ti> *__restrict__ bjn = nullptr;
        const Tv *__restrict__ weights = nullptr;

        if constexpr (Rank == 1)
        {
            aj1 = arena.r1_jumps + R.a_jump_offset;
            bj1 = arena.r1_jumps + R.b_jump_offset;
        }
        else if constexpr (Rank == 2)
        {
            aj2 = arena.r2_jumps + R.a_jump_offset;
            bj2 = arena.r2_jumps + R.b_jump_offset;
        }
        else
        {
            ajn = arena.rn_jumps + R.a_jump_offset;
            bjn = arena.rn_jumps + R.b_jump_offset;
            weights = arena.rn_weights;
        }
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                Tv vt{};
                Ti src_a = 0, src_b = 0;

                if constexpr (Rank == 1)
                {
                    const auto &ja = aj1[ia];
                    const auto &jb = bj1[ib];
                    vt = ja.w0 * jb.w0;
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                }
                else if constexpr (Rank == 2)
                {
                    const auto &ja = aj2[ia];
                    const auto &jb = bj2[ib];
                    vt = ja.w0 * jb.w0 + ja.w1 * jb.w1;
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                }
                else
                {
                    const auto &ja = ajn[ia];
                    const auto &jb = bjn[ib];
                    const Tv *wa = weights + ja.w_offset;
                    const Tv *wb = weights + jb.w_offset;
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        vt += wa[r] * wb[r];
                    }
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                }

                const int64 idx = MIXED_IDX(blk_src, src_a, src_b);
                dst[idx] += src[idx] * vt;
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void hvec_pure_a_impl(
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
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_b = blk_src.num_b;

        const TransR1<Ti, Tv> *__restrict__ r1_jumps = nullptr;
        const Tv *__restrict__ r1_phases = nullptr;
        const TransR2<Ti, Tv> *__restrict__ r2_jumps = nullptr;
        const Tv *__restrict__ r2_phases = nullptr;
        const TransRN<Ti> *__restrict__ rn_jumps = nullptr;
        const Tv *__restrict__ rn_weights = nullptr;
        const Tv *__restrict__ rn_phases = nullptr;

        if constexpr (Rank == 1)
        {
            r1_jumps = arena.r1_jumps + R.jump_offset;
            r1_phases = arena.r1_phases + R.phase_offset;
        }
        else if constexpr (Rank == 2)
        {
            r2_jumps = arena.r2_jumps + R.jump_offset;
            r2_phases = arena.r2_phases + R.phase_offset;
        }
        else
        {
            rn_jumps = arena.rn_jumps + R.jump_offset;
            rn_weights = arena.rn_weights;
            rn_phases = arena.rn_phases + R.phase_offset;
        }
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.n; ++ia)
        {
            for (int64 b = 0; b < num_b; ++b)
            {
                Tv vt{};
                Ti src_idx = 0, dst_idx = 0;

                if constexpr (Rank == 1)
                {
                    const auto &j = r1_jumps[ia];
                    vt = j.w0 * r1_phases[b];
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }
                else if constexpr (Rank == 2)
                {
                    const auto &j = r2_jumps[ia];
                    const Tv *pb = r2_phases + b * 2;
                    vt = j.w0 * pb[0] + j.w1 * pb[1];
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }
                else
                {
                    const auto &j = rn_jumps[ia];
                    const Tv *w = rn_weights + j.w_offset;
                    const Tv *pb = rn_phases + b * rank;
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        vt += w[r] * pb[r];
                    }
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }

                const int64 idx_src = PURE_A_IDX(blk_src, src_idx, b);
                const int64 idx_dst = PURE_A_IDX(blk_dst, dst_idx, b);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void hvec_pure_b_impl(
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
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];
        const int64 num_a = blk_src.num_a;

        const TransR1<Ti, Tv> *__restrict__ r1_jumps = nullptr;
        const Tv *__restrict__ r1_phases = nullptr;
        const TransR2<Ti, Tv> *__restrict__ r2_jumps = nullptr;
        const Tv *__restrict__ r2_phases = nullptr;
        const TransRN<Ti> *__restrict__ rn_jumps = nullptr;
        const Tv *__restrict__ rn_weights = nullptr;
        const Tv *__restrict__ rn_phases = nullptr;

        if constexpr (Rank == 1)
        {
            r1_jumps = arena.r1_jumps + R.jump_offset;
            r1_phases = arena.r1_phases + R.phase_offset;
        }
        else if constexpr (Rank == 2)
        {
            r2_jumps = arena.r2_jumps + R.jump_offset;
            r2_phases = arena.r2_phases + R.phase_offset;
        }
        else
        {
            rn_jumps = arena.rn_jumps + R.jump_offset;
            rn_weights = arena.rn_weights;
            rn_phases = arena.rn_phases + R.phase_offset;
        }
#pragma omp for collapse(2) schedule(static) nowait
        for (int64 a = 0; a < num_a; ++a)
        {
            for (uint32 ib = 0; ib < R.n; ++ib)
            {
                Tv vt{};
                Ti src_idx = 0, dst_idx = 0;

                if constexpr (Rank == 1)
                {
                    const auto &j = r1_jumps[ib];
                    vt = r1_phases[a] * j.w0;
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }
                else if constexpr (Rank == 2)
                {
                    const auto &j = r2_jumps[ib];
                    const Tv *pa = r2_phases + a * 2;
                    vt = pa[0] * j.w0 + pa[1] * j.w1;
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }
                else
                {
                    const auto &j = rn_jumps[ib];
                    const Tv *w = rn_weights + j.w_offset;
                    const Tv *pa = rn_phases + a * rank;
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        vt += pa[r] * w[r];
                    }
                    src_idx = j.src_idx;
                    dst_idx = j.dst_idx;
                }

                const int64 idx_src = PURE_B_IDX(blk_src, a, src_idx);
                const int64 idx_dst = PURE_B_IDX(blk_dst, a, dst_idx);
                dst[idx_src] += src[idx_dst] * math_conj(vt);
                dst[idx_dst] += src[idx_src] * vt;
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
static FORCE_INLINE void hvec_mixed_impl(
    const BasisManager<Ti> *__restrict__ basis,
    const MixedRoute *__restrict__ routes,
    const uint64 num_routes,
    const uint16 rank,
    const GroupArena<Ti, Tv> &arena,
    const Tv *__restrict__ src,
    Tv *__restrict__ dst)
{
    const BlockDesc<Ti> *__restrict__ blocks = basis->blocks;
#pragma omp parallel
    for (uint64 i = 0; i < num_routes; ++i)
    {
        const MixedRoute &R = routes[i];
        const BlockDesc<Ti> &blk_src = blocks[R.block_src_idx];
        const BlockDesc<Ti> &blk_dst = blocks[R.block_dst_idx];

        const TransR1<Ti, Tv> *__restrict__ aj1 = nullptr;
        const TransR1<Ti, Tv> *__restrict__ bj1 = nullptr;
        const TransR2<Ti, Tv> *__restrict__ aj2 = nullptr;
        const TransR2<Ti, Tv> *__restrict__ bj2 = nullptr;
        const TransRN<Ti> *__restrict__ ajn = nullptr;
        const TransRN<Ti> *__restrict__ bjn = nullptr;
        const Tv *__restrict__ weights = nullptr;

        if constexpr (Rank == 1)
        {
            aj1 = arena.r1_jumps + R.a_jump_offset;
            bj1 = arena.r1_jumps + R.b_jump_offset;
        }
        else if constexpr (Rank == 2)
        {
            aj2 = arena.r2_jumps + R.a_jump_offset;
            bj2 = arena.r2_jumps + R.b_jump_offset;
        }
        else
        {
            ajn = arena.rn_jumps + R.a_jump_offset;
            bjn = arena.rn_jumps + R.b_jump_offset;
            weights = arena.rn_weights;
        }
#pragma omp for collapse(2) schedule(static) nowait
        for (uint32 ia = 0; ia < R.na; ++ia)
        {
            for (uint32 ib = 0; ib < R.nb; ++ib)
            {
                Tv vt{};
                Ti src_a = 0, src_b = 0, dst_a = 0, dst_b = 0;

                if constexpr (Rank == 1)
                {
                    const auto &ja = aj1[ia];
                    const auto &jb = bj1[ib];
                    vt = ja.w0 * jb.w0;
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                    dst_a = ja.dst_idx;
                    dst_b = jb.dst_idx;
                }
                else if constexpr (Rank == 2)
                {
                    const auto &ja = aj2[ia];
                    const auto &jb = bj2[ib];
                    vt = ja.w0 * jb.w0 + ja.w1 * jb.w1;
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                    dst_a = ja.dst_idx;
                    dst_b = jb.dst_idx;
                }
                else
                {
                    const auto &ja = ajn[ia];
                    const auto &jb = bjn[ib];
                    const Tv *wa = weights + ja.w_offset;
                    const Tv *wb = weights + jb.w_offset;
                    for (uint16 r = 0; r < rank; ++r)
                    {
                        vt += wa[r] * wb[r];
                    }
                    src_a = ja.src_idx;
                    src_b = jb.src_idx;
                    dst_a = ja.dst_idx;
                    dst_b = jb.dst_idx;
                }

                const int64 idx_src = MIXED_IDX(blk_src, src_a, src_b);
                const int64 idx_dst = MIXED_IDX(blk_dst, dst_a, dst_b);
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
        hvec_diag_impl<1, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    case 2:
        hvec_diag_impl<2, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    default:
        hvec_diag_impl<0, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
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
    if (!num_routes)
        return;
    switch (rank)
    {
    case 1:
        hvec_pure_a_impl<1, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    case 2:
        hvec_pure_a_impl<2, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    default:
        hvec_pure_a_impl<0, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
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
    if (!num_routes)
        return;
    switch (rank)
    {
    case 1:
        hvec_pure_b_impl<1, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    case 2:
        hvec_pure_b_impl<2, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    default:
        hvec_pure_b_impl<0, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
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
        hvec_mixed_impl<1, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    case 2:
        hvec_mixed_impl<2, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
        break;
    default:
        hvec_mixed_impl<0, Ti, Tv>(basis, routes, num_routes, rank, arena, src, dst);
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
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst[i] = {};
    }

    for (uint64 g = 0; g < net->ngs; ++g)
    {
        const uint8 type = net->excit_types[g];

        switch (type)
        {
        case 0:
            hvec_diag<Ti, Tv>(
                basis, net->mixed_routes[g], net->num_mixed_routes[g],
                net->group_ranks[g], net->arenas[g], src, dst);
            break;
        case 1:
            hvec_pure_a<Ti, Tv>(
                basis, net->pure_a_routes[g], net->num_pure_a_routes[g],
                net->group_ranks[g], net->arenas[g], src, dst);
            break;
        case 2:
            hvec_pure_b<Ti, Tv>(
                basis, net->pure_b_routes[g], net->num_pure_b_routes[g],
                net->group_ranks[g], net->arenas[g], src, dst);
            break;
        case 3:
            hvec_mixed<Ti, Tv>(
                basis, net->mixed_routes[g], net->num_mixed_routes[g],
                net->group_ranks[g], net->arenas[g], src, dst);
            break;
        default:
            std::cerr << "Error: Unexpected type = " << static_cast<int>(type)
                      << " at group g = " << g << " when hvec_svd"
                      << std::endl;
            break;
        }
    }
}
