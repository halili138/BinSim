#pragma once
#include "core/types.hpp"
#include "basis/basis.hpp"
#include "ham/otf.hpp"
#include "core/math.hpp"
#include <ankerl/unordered_dense.h>
#include <cassert>

inline constexpr int64 GROUP_CHUNK_SIZE = 1 << 10;
inline constexpr int64 TARGET_CHUNK_SIZE = 1 << 16;

template <typename Tv>
FORCE_INLINE auto sqnorm(const Tv &v)
{
    if constexpr (std::is_arithmetic_v<Tv>)
        return v * v;
    else
        return v.real() * v.real() + v.imag() * v.imag();
}

template <typename Tv>
FORCE_INLINE bool sci_eps_check(Tv acc, Tv haa, Tv e_var, double eps)
{
    if (acc == Tv{})
        return false;
    Tv denom = e_var - haa;
    double dn_sq = sqnorm(denom);
    if (dn_sq == 0.0)
        return false;
    return sqnorm(acc) / dn_sq > eps * eps;
}

template <typename Ti, typename Tv>
FORCE_INLINE void precompute_phase_select(Ti str, const Ti *zs, int nz, const Tv *w0, Tv *ps, int stride, int rank)
{
    if (rank == 1)
    {
        Tv vt = Tv{};
        for (int i = 0; i < nz; ++i)
        {
            bool phase = popcnt(str & zs[i]) & 1;
            vt += phase ? -w0[i] : w0[i];
        }
        ps[0] = vt;
    }
    else
    {
        const Tv *w1 = w0 + nz;
        Tv v0 = Tv{}, v1 = Tv{};
        for (int i = 0; i < nz; ++i)
        {
            bool phase = popcnt(str & zs[i]) & 1;
            v0 += phase ? -w0[i] : w0[i];
            v1 += phase ? -w1[i] : w1[i];
        }
        ps[0] = v0;
        ps[stride] = v1;
    }
}

template <typename Ti, typename Tv>
static void precompute_diag_phases(
    const Ti *strs, int num_strs,
    const Ti *zs, const Tv *w, int num_zs, int rank,
    Tv *ps)
{
    for (int i = 0; i < num_strs; ++i)
    {
        Ti str = strs[i];
        Tv *pi = ps + i * rank;
        for (int r = 0; r < rank; ++r)
        {
            const Tv *wr = w + r * num_zs;
            Tv vr = {};
            for (int k = 0; k < num_zs; ++k)
            {
                bool parity = popcnt(str & zs[k]) & 1;
                vr += parity ? -wr[k] : wr[k];
            }
            pi[r] = vr;
        }
    }
}

template <typename Ti, typename Tv>
static std::vector<SVDGroup_OTF<Ti, Tv>> flatten_groups(const Network_OTF<Ti, Tv> *net)
{
    std::vector<SVDGroup_OTF<Ti, Tv>> all;
    all.insert(all.end(), net->pure_a_groups.begin(), net->pure_a_groups.end());
    all.insert(all.end(), net->pure_b_groups.begin(), net->pure_b_groups.end());
    all.insert(all.end(), net->mixed_groups.begin(), net->mixed_groups.end());
    return all;
}
