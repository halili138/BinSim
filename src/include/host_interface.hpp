#pragma once
#include "host_framework.hpp"
#include <type_traits>

template <int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE typename Op::Result dispatch_contract_group_by_rank(const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group, Op op)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;

    switch (rank)
    {
    case 1:
        return otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
    case 2:
        return otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
    default:
        return otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
    }
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE auto dispatch_contract_network(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, Op op)
{
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, TypeCode, pos);
    using Result = decltype(dispatch_contract_group_by_rank<TypeCode>(basis, *group, op));
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << TypeCode << " in dispatch_contract_network" << std::endl;
        if constexpr (std::is_void_v<Result>)
            return;
        else
            return Result{};
    }

    return dispatch_contract_group_by_rank<TypeCode>(basis, *group, op);
}

template <typename Ti, typename Tv, typename Op>
static FORCE_INLINE auto dispatch_contract_network(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, Op op)
{
    const uint8 type = net->excit_types[idx];

    switch (type)
    {
    case 0:
        return dispatch_contract_network<0>(basis, net, idx, op);
    case 1:
        return dispatch_contract_network<1>(basis, net, idx, op);
    case 2:
        return dispatch_contract_network<2>(basis, net, idx, op);
    case 3:
        return dispatch_contract_network<3>(basis, net, idx, op);
    default:
    {
        using Result = decltype(dispatch_contract_network<0>(basis, net, idx, op));
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in dispatch_contract_network" << std::endl;
        if constexpr (std::is_void_v<Result>)
            return;
        else
            return Result{};
    }
    }
}

template <typename Tv>
struct DiagElementsContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    Tv *diags;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        diags[di] += vt;
    }

    FORCE_INLINE void offdiag(Result &, Tv, int64, int64) const {}
};

template <typename Ti, typename Tv>
void get_diags_elements(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, Tv *diags)
{
    if (net->diag_groups.empty())
        return;

    const DiagElementsContractOp<Tv> op{diags};
    (void)dispatch_contract_group_by_rank<0>(basis, net->diag_groups[0], op);
}

template <typename Tv>
struct ExpmContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double cd;
    double co;
    Tv *vec;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        vec[di] *= fast_diag_exp<Tv>(vt, theta);
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        expm_update<Tv>(vec + si, vec + di, vt, cd, co);
    }
};

template <typename Ti, typename Tv>
void expm_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *vec)
{
    const ExpmContractOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), vec};
    (void)dispatch_contract_network(basis, net, idx, op);
}

template <typename Tv>
struct GradContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = true;

    double theta;
    double cd;
    double co;
    const Tv *lp;
    const Tv *rp;

    FORCE_INLINE void diag(Result &res, Tv vt, int64 di) const
    {
        const Tv du = fast_diag_grad<Tv>(vt, theta);
        res += math_conj(lp[di] * du) * rp[di];
    }

    FORCE_INLINE void offdiag(Result &res, Tv vt, int64 si, int64 di) const
    {
        grad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, cd, co);
    }
};

template <typename Ti, typename Tv>
Tv grad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const GradContractOp<Tv> op{theta, -std::sin(theta), std::cos(theta), lp, rp};
    return dispatch_contract_network(basis, net, idx, op);
}

template <typename Tv>
struct TVecContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    const Tv *src_vec;
    Tv *dst_vec;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        dst_vec[di] = src_vec[di] * vt;
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        tvec_update<Tv>(src_vec + si, src_vec + di, dst_vec + si, dst_vec + di, vt);
    }
};

template <typename Ti, typename Tv>
void tvec_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, const Tv *src_vec, Tv *dst_vec)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        dst_vec[i] = {};

    const TVecContractOp<Tv> op{src_vec, dst_vec};
    (void)dispatch_contract_network(basis, net, idx, op);
}

template <typename Tv>
struct BackgradContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = true;

    double theta;
    double ecd;
    double eco;
    double gcd;
    double gco;
    Tv *lp;
    Tv *rp;

    FORCE_INLINE void diag(Result &res, Tv vt, int64 di) const
    {
        const Tv u = fast_diag_exp<Tv>(vt, -theta);
        const Tv du = fast_diag_grad<Tv>(vt, theta);
        lp[di] *= u;
        res += math_conj(lp[di] * du) * rp[di];
        rp[di] *= u;
    }

    FORCE_INLINE void offdiag(Result &res, Tv vt, int64 si, int64 di) const
    {
        backgrad_update<Tv>(res, lp + si, lp + di, rp + si, rp + di, vt, ecd, eco, gcd, gco);
    }
};

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const BackgradContractOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp};
    return dispatch_contract_network(basis, net, idx, op);
}

template <typename Tv>
struct BacktranContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double ecd;
    double eco;
    Tv *lp;
    Tv *rp;
    Tv *bp;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        const Tv u = fast_diag_exp<Tv>(vt, theta);
        lp[di] *= u;
        rp[di] *= u;
        bp[di] = lp[di] * vt;
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        backtran_update<Tv>(lp + si, lp + di, rp + si, rp + di, bp + si, bp + di, vt, ecd, eco);
    }
};

template <typename Ti, typename Tv>
void backtran_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp, Tv *bp)
{
#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        bp[i] = {};

    const BacktranContractOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp};
    (void)dispatch_contract_network(basis, net, idx, op);
}

template <typename Tv>
struct BatchedExpmContractOp
{
    using Result = Tv;
    static constexpr bool Accumulates = false;

    double theta;
    double cd;
    double co;
    Tv *matrix;
    int ld;
    int num_vecs;

    FORCE_INLINE void diag(Result &, Tv vt, int64 di) const
    {
        expm_batch_update_diag<Tv>(matrix, ld, num_vecs, di, fast_diag_exp<Tv>(vt, theta));
    }

    FORCE_INLINE void offdiag(Result &, Tv vt, int64 si, int64 di) const
    {
        expm_batch_update_matrix<Tv>(matrix, ld, num_vecs, si, di, vt, cd, co);
    }
};

template <typename Ti, typename Tv>
void batchexpm_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *matrix, int ld, int num_vecs)
{
    const BatchedExpmContractOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), matrix, ld, num_vecs};
    (void)dispatch_contract_network(basis, net, idx, op);
}

template <int TypeCode, typename Ti, typename Tv, typename Op>
static FORCE_INLINE void dispatch_group_chunks_by_rank(const BasisManager<Ti> *basis, const std::vector<SVDGroup_OTF<Ti, Tv>> &groups, const Op &op)
{
    const int64 total_ngs = groups.size();
    if (total_ngs == 0)
        return;

    const SVDGroup_OTF<Ti, Tv> *groups_ptr = groups.data();
    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = normalized_dispatch_rank(groups, start);
        const int64 end = next_rank_chunk_end(groups, start);
        const SVDGroup_OTF<Ti, Tv> *chunk_ptr = groups_ptr + start;
        const int64 chunk_size = end - start;

        switch (dispatch_rank)
        {
        case 1:
            otf_contract_batched_groups_impl<1, TypeCode>(basis, chunk_ptr, chunk_size, op);
            break;
        case 2:
            otf_contract_batched_groups_impl<2, TypeCode>(basis, chunk_ptr, chunk_size, op);
            break;
        default:
            otf_contract_batched_groups_impl<0, TypeCode>(basis, chunk_ptr, chunk_size, op);
            break;
        }

        start = end;
    }
}

template <typename Tv>
struct HVecBatchedOp
{
    struct ThreadState
    {
    };

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
static FORCE_INLINE void gather_contract_batched_impl(const BasisView<Ti> &view, const SVDGroup_OTF<Ti, Tv> *groups, int64 num_groups, const Tv *src_vec, Tv *dst_vec)
{
    BasisManager<Ti> basis_stub;
    basis_stub.view = view;
    const HVecBatchedOp<Tv> op{src_vec, dst_vec};
    otf_contract_batched_groups_impl<Rank, TypeCode>(&basis_stub, groups, num_groups, op);
}

template <typename Ti, typename Tv>
void hvec_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, const Tv *src_vec, Tv *dst_vec)
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

template <typename Tv>
struct BatchGradBatchedOp
{
    struct ThreadState
    {
        std::vector<Tv> values;
    };

    static constexpr bool SkipLowerBlocks = true;
    static constexpr bool SkipSameBlockAReverse = true;
    static constexpr bool SkipSameBlockBReverseForPureB = true;

    const double *thetas;
    const Tv *lp;
    const Tv *rp;
    Tv *grads;

    FORCE_INLINE ThreadState make_thread_state(int64 num_groups) const
    {
        return ThreadState{std::vector<Tv>(num_groups, Tv{})};
    }

    template <typename Ti>
    FORCE_INLINE void diag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &group, Tv &local_res, Tv vt, int64 di) const
    {
        const double theta = thetas[group.original_idx];
        const Tv du = fast_diag_grad<Tv>(vt, theta);
        local_res += math_conj(lp[di] * du) * rp[di];
    }

    template <typename Ti>
    FORCE_INLINE void offdiag(ThreadState &, int64, const SVDGroup_OTF<Ti, Tv> &group, Tv &local_res, Tv vt, int64 si, int64 di) const
    {
        const double theta = thetas[group.original_idx];
        grad_update<Tv>(local_res, lp + si, lp + di, rp + si, rp + di, vt, -std::sin(theta), std::cos(theta));
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
                grads[groups[g].original_idx] += state.values[g];
        }
    }
};

template <typename Ti, typename Tv>
void batchgrad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, const double *thetas, const Tv *lp, const Tv *rp, Tv *grads)
{
    std::fill(grads, grads + net->num_groups, Tv{});

    const BatchGradBatchedOp<Tv> op{thetas, lp, rp, grads};
    dispatch_group_chunks_by_rank<0>(basis, net->diag_groups, op);
    dispatch_group_chunks_by_rank<1>(basis, net->pure_a_groups, op);
    dispatch_group_chunks_by_rank<2>(basis, net->pure_b_groups, op);
    dispatch_group_chunks_by_rank<3>(basis, net->mixed_groups, op);
}

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
void batchtran_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, const Tv *lp, const Tv *rp, Tv *trans)
{
    std::fill(trans, trans + net->num_groups, Tv{});

    const BatchTranBatchedOp<Tv> op{lp, rp, trans};
    dispatch_group_chunks_by_rank<0>(basis, net->diag_groups, op);
    dispatch_group_chunks_by_rank<1>(basis, net->pure_a_groups, op);
    dispatch_group_chunks_by_rank<2>(basis, net->pure_b_groups, op);
    dispatch_group_chunks_by_rank<3>(basis, net->mixed_groups, op);
}
