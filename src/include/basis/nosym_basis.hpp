#pragma once
#include <vector>
#include <ankerl/unordered_dense.h>
#include "select/utils.hpp"

template <typename Ti>
struct SciBasisManagerNosym
{
    Ti *all_astrs = nullptr;
    Ti *all_bstrs = nullptr;
    int64 num_a = 0;
    int64 num_b = 0;
    int64 dim = 0;
    int64 norb = 0;
    ankerl::unordered_dense::map<Ti, int> a_idx_map; // str → 0..num_a-1
    ankerl::unordered_dense::map<Ti, int> b_idx_map;

    void clear()
    {
        delete[] all_astrs;
        all_astrs = nullptr;
        delete[] all_bstrs;
        all_bstrs = nullptr;
        a_idx_map.clear();
        b_idx_map.clear();
        num_a = num_b = dim = norb = 0;
    }

    ~SciBasisManagerNosym() { clear(); }
};

template <typename Ti>
SciBasisManagerNosym<Ti> *create_sci_basis_manager_nosym(
    const Ti *astrs, int64 num_a,
    const Ti *bstrs, int64 num_b,
    int64 norb)
{
    auto *basis = new SciBasisManagerNosym<Ti>();
    basis->num_a = num_a;
    basis->num_b = num_b;
    basis->dim = num_a * num_b;
    basis->norb = norb;

    basis->all_astrs = new Ti[num_a];
    basis->all_bstrs = new Ti[num_b];
    std::copy(astrs, astrs + num_a, basis->all_astrs);
    std::copy(bstrs, bstrs + num_b, basis->all_bstrs);

    for (int64 i = 0; i < num_a; ++i)
        basis->a_idx_map[astrs[i]] = (int)i;
    for (int64 i = 0; i < num_b; ++i)
        basis->b_idx_map[bstrs[i]] = (int)i;

    return basis;
}

template <typename Ti>
void destroy_sci_basis_manager_nosym(SciBasisManagerNosym<Ti> *basis)
{
    basis->clear();
    delete basis;
}

template <typename Ti, typename Tv>
static void remap_wavefunction_nosym(
    const SciBasisManagerNosym<Ti> *old_src, const Tv *old_psi,
    const SciBasisManagerNosym<Ti> *new_src, Tv *new_psi)
{
    std::fill_n(new_psi, new_src->dim, Tv{});

    for (int64 a = 0; a < old_src->num_a; ++a)
    {
        Ti astr = old_src->all_astrs[a];
        auto it_a = new_src->a_idx_map.find(astr);
        if (it_a == new_src->a_idx_map.end())
            continue;

        const int64 new_a = it_a->second;
        for (int64 b = 0; b < old_src->num_b; ++b)
        {
            auto it_b = new_src->b_idx_map.find(old_src->all_bstrs[b]);
            if (it_b == new_src->b_idx_map.end())
                continue;

            int64 old_pos = a * old_src->num_b + b;
            int64 new_pos = new_a * new_src->num_b + it_b->second;
            new_psi[new_pos] = old_psi[old_pos];
        }
    }
}

template <typename Ti, typename Tv>
static void get_diags_elements_sci_nosym(
    const SciBasisManagerNosym<Ti> *basis,
    const Network_OTF<Ti, Tv> *net, Tv *diags)
{
    const SVDGroup_OTF<Ti, Tv> &group = net->diag_groups[0];
    const int rank = group.rank;
    const Ti *zas = group.unique_zas;
    const Ti *zbs = group.unique_zbs;
    const Tv *wa = group.wa;
    const Tv *wb = group.wb;
    const int nza = group.num_za;
    const int nzb = group.num_zb;
    const int na = (int)basis->num_a;
    const int nb = (int)basis->num_b;

#pragma omp parallel
    {
        std::vector<Tv> a_phase(na * rank);
        std::vector<Tv> b_phase(nb * rank);

        Tv *pa = a_phase.data();
        Tv *pb = b_phase.data();

#pragma omp for
        for (int a = 0; a < na; ++a)
            precompute_phase<0, Ti, Tv>(basis->all_astrs[a], zas, nza, wa, pa + a, na, rank);

#pragma omp for
        for (int b = 0; b < nb; ++b)
            precompute_phase<0, Ti, Tv>(basis->all_bstrs[b], zbs, nzb, wb, pb + b, nb, rank);

#pragma omp for collapse(2) schedule(static)
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b)
                diags[(int64)a * nb + b] += compute_coeff<0, Tv>(a, b, pa, pb, na, nb, rank);
    }
}
