#pragma once
#include "group_batched_framework.hpp"

template <typename Tv>
struct BatchTranBatchedOp
{
    struct ThreadState
    {
        std::vector<Tv> values;
    };

    static constexpr bool SkipLowerBlocks = true;
    static constexpr bool SkipSameBlockAReverse = true;
    static constexpr bool SkipSameBlockBReverseForPureB = true;

    const Tv *lp;
    const Tv *rp;
    Tv *trans;

    FORCE_INLINE ThreadState make_thread_state(int64 num_groups) const
    {
        return ThreadState{std::vector<Tv>(num_groups, Tv{})};
    }

    template <typename Ti>
    FORCE_INLINE void diag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &, Tv &local_res, Tv vt, int64 di) const
    {
        local_res += math_conj(lp[di] * vt) * rp[di];
    }

    template <typename Ti>
    FORCE_INLINE void offdiag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &, Tv &local_res, Tv vt, int64 si, int64 di) const
    {
        tran_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt);
    }

    FORCE_INLINE void commit_local(ThreadState &state, int64 g, Tv local_res) const
    {
        state.values[g] += local_res;
    }

    template <typename Ti>
    FORCE_INLINE void finish_thread(ThreadState &state, const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups) const
    {
#pragma omp critical
        {
            for (int64 g = 0; g < num_groups; ++g)
                trans[groups[g].original_idx] += state.values[g];
        }
    }
};

template <typename Ti, typename Tv>
void tran_pool_network_batched_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net,
    const Tv *lp, const Tv *rp, Tv *trans)
{
    std::fill(trans, trans + net->num_groups, Tv{});

    const BatchTranBatchedOp<Tv> op{lp, rp, trans};
    dispatch_group_chunks_by_rank<0>(basis, net->diag_groups, op);
    dispatch_group_chunks_by_rank<1>(basis, net->pure_a_groups, op);
    dispatch_group_chunks_by_rank<2>(basis, net->pure_b_groups, op);
    dispatch_group_chunks_by_rank<3>(basis, net->mixed_groups, op);
}
