#pragma once
#include "otf_contract.hpp"

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

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void launch_backtran_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp, Tv *bp)
{
    const BacktranContractOp<Tv> op{theta, std::cos(theta) - 1.0, -std::sin(theta), lp, rp, bp};
    (void)otf_contract_single_group_impl<Rank, TypeCode>(basis, group, op);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_backtran_contract_group(
    const BasisManager<Ti> *basis, const SVDGroup_OTF<Ti, Tv> &group,
    double theta, Tv *lp, Tv *rp, Tv *bp)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        launch_backtran_contract_group<1, TypeCode>(basis, group, theta, lp, rp, bp);
        break;
    case 2:
        launch_backtran_contract_group<2, TypeCode>(basis, group, theta, lp, rp, bp);
        break;
    default:
        launch_backtran_contract_group<0, TypeCode>(basis, group, theta, lp, rp, bp);
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
