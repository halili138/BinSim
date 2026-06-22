#pragma once
#include "otf_contract.hpp"

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

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv launch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    const GradContractOp<Tv> op{theta, -std::sin(theta), std::cos(theta), lp, rp};
    return otf_contract_single_group_impl<Rank, TypeCode>(basis, group, op);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE Tv dispatch_grad_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, const Tv *lp, const Tv *rp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        return launch_grad_contract_group<1, TypeCode>(basis, group, theta, lp, rp);
    case 2:
        return launch_grad_contract_group<2, TypeCode>(basis, group, theta, lp, rp);
    default:
        return launch_grad_contract_group<0, TypeCode>(basis, group, theta, lp, rp);
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
