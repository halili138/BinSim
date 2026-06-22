#include "otf_contract.hpp"

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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *vec)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const ExpmContractOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), vec};

    switch (rank)
    {
    case 1:
        (void)otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
        break;
    case 2:
        (void)otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
        break;
    default:
        (void)otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
        break;
    }
}

template <typename Ti, typename Tv>
void expm_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *vec)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd_network_otf" << std::endl;
        return;
    }

    switch (type)
    {
    case 0:
        dispatch_expm_contract_group<0>(basis, *group, theta, vec);
        break;
    case 1:
        dispatch_expm_contract_group<1>(basis, *group, theta, vec);
        break;
    case 2:
        dispatch_expm_contract_group<2>(basis, *group, theta, vec);
        break;
    case 3:
        dispatch_expm_contract_group<3>(basis, *group, theta, vec);
        break;
    }
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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const GradContractOp<Tv> op{theta, -std::sin(theta), std::cos(theta), lp, rp};

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

template <typename Ti, typename Tv>
Tv grad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, const Tv *lp, const Tv *rp)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in grad_svd" << std::endl;
        return {};
    }

    switch (type)
    {
    case 0:
        return dispatch_grad_contract_group<0>(basis, *group, theta, lp, rp);
    case 1:
        return dispatch_grad_contract_group<1>(basis, *group, theta, lp, rp);
    case 2:
        return dispatch_grad_contract_group<2>(basis, *group, theta, lp, rp);
    case 3:
        return dispatch_grad_contract_group<3>(basis, *group, theta, lp, rp);
    default:
        return {};
    }
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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_tvec_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    const Tv *src_vec, Tv *dst_vec)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const TVecContractOp<Tv> op{src_vec, dst_vec};

    switch (rank)
    {
    case 1:
        (void)otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
        break;
    case 2:
        (void)otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
        break;
    default:
        (void)otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
        break;
    }
}

template <typename Ti, typename Tv>
void tvec_svd_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, const Tv *src_vec, Tv *dst_vec)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in tvec_svd" << std::endl;
        return;
    }

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        dst_vec[i] = {};

    switch (type)
    {
    case 0:
        dispatch_tvec_contract_group<0>(basis, *group, src_vec, dst_vec);
        break;
    case 1:
        dispatch_tvec_contract_group<1>(basis, *group, src_vec, dst_vec);
        break;
    case 2:
        dispatch_tvec_contract_group<2>(basis, *group, src_vec, dst_vec);
        break;
    case 3:
        dispatch_tvec_contract_group<3>(basis, *group, src_vec, dst_vec);
        break;
    }
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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_backgrad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const BackgradContractOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), -std::sin(theta), std::cos(theta), lp, rp};

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

template <typename Ti, typename Tv>
Tv backgrad_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in grad_svd" << std::endl;
        return {};
    }

    switch (type)
    {
    case 0:
        return dispatch_backgrad_contract_group<0>(basis, *group, theta, lp, rp);
    case 1:
        return dispatch_backgrad_contract_group<1>(basis, *group, theta, lp, rp);
    case 2:
        return dispatch_backgrad_contract_group<2>(basis, *group, theta, lp, rp);
    case 3:
        return dispatch_backgrad_contract_group<3>(basis, *group, theta, lp, rp);
    default:
        return {};
    }
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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_backtran_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp, Tv *bp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const BacktranContractOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp};

    switch (rank)
    {
    case 1:
        (void)otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
        break;
    case 2:
        (void)otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
        break;
    default:
        (void)otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
        break;
    }
}

template <typename Ti, typename Tv>
void backtran_svd_network_otf(const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta, Tv *lp, Tv *rp, Tv *bp)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in grad_svd" << std::endl;
        return;
    }

#pragma omp parallel for schedule(static)
    for (int64 i = 0; i < basis->dim; ++i)
        bp[i] = {};

    switch (type)
    {
    case 0:
        dispatch_backtran_contract_group<0>(basis, *group, theta, lp, rp, bp);
        break;
    case 1:
        dispatch_backtran_contract_group<1>(basis, *group, theta, lp, rp, bp);
        break;
    case 2:
        dispatch_backtran_contract_group<2>(basis, *group, theta, lp, rp, bp);
        break;
    case 3:
        dispatch_backtran_contract_group<3>(basis, *group, theta, lp, rp, bp);
        break;
    }
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

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_batched_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    const BatchedExpmContractOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), matrix, ld, num_vecs};

    switch (rank)
    {
    case 1:
        (void)otf_contract_single_group_impl<1, TypeCode>(basis, group, op);
        break;
    case 2:
        (void)otf_contract_single_group_impl<2, TypeCode>(basis, group, op);
        break;
    default:
        (void)otf_contract_single_group_impl<0, TypeCode>(basis, group, op);
        break;
    }
}

template <typename Ti, typename Tv>
void expm_svd_batched_network_otf(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *net, int64 idx, double theta,
    Tv *matrix, int ld, int num_vecs)
{
    const uint8 type = net->excit_types[idx];
    const int64 pos = net->sorted_idxs[idx];
    const SVDGroup_OTF<Ti, Tv> *group = group_by_type(net, type, pos);
    if (group == nullptr)
    {
        std::cerr << "Error: Unexpected type = " << static_cast<int>(type) << " in expm_svd_batched_network_otf" << std::endl;
        return;
    }

    switch (type)
    {
    case 0:
        dispatch_expm_batched_contract_group<0>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 1:
        dispatch_expm_batched_contract_group<1>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 2:
        dispatch_expm_batched_contract_group<2>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    case 3:
        dispatch_expm_batched_contract_group<3>(basis, *group, theta, matrix, ld, num_vecs);
        break;
    }
}
