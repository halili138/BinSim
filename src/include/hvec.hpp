#pragma once
#include "group_batched_framework.hpp"

template <typename Tv>
struct HVecBatchedOp
{
    struct ThreadState {};

    static constexpr bool SkipLowerBlocks = false;
    static constexpr bool SkipSameBlockAReverse = false;
    static constexpr bool SkipSameBlockBReverseForPureB = false;

    const Tv *src_vec;
    Tv *dst_vec;

    FORCE_INLINE ThreadState make_thread_state(int64) const { return {}; }

    template <typename Ti>
    FORCE_INLINE void diag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &, Tv &, Tv vt, int64 di) const
    {
        hvec_update<Tv>(src_vec + di, dst_vec + di, vt);
    }

    template <typename Ti>
    FORCE_INLINE void offdiag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &, Tv &, Tv vt, int64 si, int64 di) const
    {
        hvec_update<Tv>(src_vec + si, dst_vec + di, vt);
    }

    FORCE_INLINE void commit_local(ThreadState &, int64, Tv) const {}

    template <typename Ti>
    FORCE_INLINE void finish_thread(ThreadState &, const SVDGroup_OTF<Ti, Tv> *, int64) const {}
};


template <int Rank, int TypeCode, typename Ti, typename Tv>
static inline void gather_contract_batched_impl(
    const BasisView<Ti> &view,
    const SVDGroup_OTF<Ti, Tv> *groups,
    int64 num_groups,
    const Tv *src_vec,
    Tv *dst_vec)
{
    BasisManager<Ti> basis_stub;
    basis_stub.view = view;
    const HVecBatchedOp<Tv> op{src_vec, dst_vec};
    group_contract_batched_impl<Rank, TypeCode>(&basis_stub, groups, num_groups, op);
}

template <typename Ti, typename Tv>
void contract_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, const Tv *src_vec, Tv *dst_vec)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
    {
        dst_vec[i] = {};
    }

    const HVecBatchedOp<Tv> op{src_vec, dst_vec};
    dispatch_group_chunks_by_rank<0>(basis, net->diag_groups, op);
    dispatch_group_chunks_by_rank<1>(basis, net->pure_a_groups, op);
    dispatch_group_chunks_by_rank<2>(basis, net->pure_b_groups, op);
    dispatch_group_chunks_by_rank<3>(basis, net->mixed_groups, op);
}

template <typename Ti, typename Tv>
void get_diags_elements(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, Tv *diags)
{
    const SVDGroup_OTF<Ti, Tv> &group = net->diag_groups[0];
    const int rank = group.rank;
    const int num_za = group.num_za;
    const int num_zb = group.num_zb;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa0 = group.wa;
    const Tv *wb0 = group.wb;
    const BlockDesc<Ti> *blocks = basis->blocks;
    const int64 num_blocks = basis->num_blocks;
    const int max_a_count = (int)basis->max_a_count;
    const int max_b_count = (int)basis->max_b_count;

#pragma omp parallel
    {
        std::vector<Tv> local_a_phase(max_a_count * rank);
        std::vector<Tv> local_b_phase(max_b_count * rank);

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const BlockDesc<Ti> &block = blocks[block_idx];
            const Ti *astrs = block.astrs;
            const Ti *bstrs = block.bstrs;
            const int a_count = block.num_a;
            const int b_count = block.num_b;
            Tv *pa0 = local_a_phase.data();
            Tv *pb0 = local_b_phase.data();

            for (int i = 0; i < a_count; ++i)
            {
                precompute_phase<0, Ti, Tv>(astrs[i], zas, num_za, wa0, pa0 + i, max_a_count, rank);
            }

            for (int i = 0; i < b_count; ++i)
            {
                precompute_phase<0, Ti, Tv>(bstrs[i], zbs, num_zb, wb0, pb0 + i, max_b_count, rank);
            }

            const Tv *pa = local_a_phase.data();
            const Tv *pb = local_b_phase.data();

#pragma omp for collapse(2) schedule(static) nowait
            for (int a = 0; a < a_count; ++a)
            {
                for (int b = 0; b < b_count; ++b)
                {
                    const Tv vt = compute_coeff<0, Tv>(a, b, pa, pb, max_a_count, max_b_count, rank);
                    const int64 i = block.offset + (int64)a * b_count + b;
                    diags[i] += vt;
                }
            }
        }
    }
}
