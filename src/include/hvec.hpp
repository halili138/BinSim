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
