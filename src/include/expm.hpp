#pragma once
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

template <int Rank, int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void launch_expm_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *vec)
{
    const ExpmContractOp<Tv> op{theta, std::cos(theta) - 1.0, std::sin(theta), vec};
    (void)otf_contract_single_group_impl<Rank, TypeCode>(basis, group, op);
}

template <int TypeCode, typename Ti, typename Tv>
static FORCE_INLINE void dispatch_expm_contract_group(
    const BasisManager<Ti> *basis,
    const SVDGroup_OTF<Ti, Tv> &group,
    double theta,
    Tv *vec)
{
    const int rank = (group.rank == 1 || group.rank == 2) ? group.rank : 0;
    switch (rank)
    {
    case 1:
        launch_expm_contract_group<1, TypeCode>(basis, group, theta, vec);
        break;
    case 2:
        launch_expm_contract_group<2, TypeCode>(basis, group, theta, vec);
        break;
    default:
        launch_expm_contract_group<0, TypeCode>(basis, group, theta, vec);
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
