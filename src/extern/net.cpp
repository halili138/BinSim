#include "net.hpp"
#include <omp.h>
#include <algorithm>
#include <iostream>

extern "C"
{
    void *create_svd_network(
        void *basis_ptr,
        int64 ncs,
        int64 ngs,
        const uint32 *axs,
        const uint32 *bxs,
        const uint32 *azs,
        const uint32 *bzs,
        const double *cs,
        const int64 *gs,
        const int64 *ranks,
        const int64 *num_as,
        const int64 *num_bs,
        const uint32 *flat_azs,
        const uint32 *flat_bzs,
        const double *flat_wa,
        const double *flat_wb,
        const int64 *orbsym)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        SVDNetwork *net = new SVDNetwork();

        net->azs = new uint32[ncs]();
        net->bzs = new uint32[ncs]();
        net->cs = new double[ncs]();
        net->gs = new uint64[ngs + 1]();
        net->ngs = ngs;

        std::copy(azs, azs + ncs, net->azs);
        std::copy(bzs, bzs + ncs, net->bzs);
        std::copy(cs, cs + ncs, net->cs);
        std::copy(gs, gs + ngs + 1, net->gs);

        net->excit_types = new uint8[ngs]();
        net->group_ranks = new uint16[ngs]();
        net->arenas = new GroupArena[ngs]();

        net->pure_a_routes = new PureRoute *[ngs]();
        net->num_pure_a_routes = new uint64[ngs]();
        net->pure_b_routes = new PureRoute *[ngs]();
        net->num_pure_b_routes = new uint64[ngs]();
        net->mixed_routes = new MixedRoute *[ngs]();
        net->num_mixed_routes = new uint64[ngs]();

        std::vector<int64> off_az(ngs, 0), off_bz(ngs, 0), off_wa(ngs, 0), off_wb(ngs, 0);
        int64 caz = 0, cbz = 0, cwa = 0, cwb = 0;
        for (int64 g = 0; g < ngs; ++g)
        {
            off_az[g] = caz;
            off_bz[g] = cbz;
            off_wa[g] = cwa;
            off_wb[g] = cwb;
            caz += num_as[g];
            cbz += num_bs[g];
            cwa += num_as[g] * ranks[g];
            cwb += num_bs[g] * ranks[g];
        }

#pragma omp parallel for schedule(dynamic)
        for (int64 g = 0; g < ngs; ++g)
        {
            uint32 ax = axs[g], bx = bxs[g];
            int64 rank = ranks[g];
            net->group_ranks[g] = static_cast<uint16>(rank);

            int type = 0;
            if (ax != 0 && bx == 0)
                type = 1;
            else if (ax == 0 && bx != 0)
                type = 2;
            else if (ax != 0 && bx != 0)
                type = 3;

            net->excit_types[g] = static_cast<uint8>(type);

            TempArena temp;

            switch (type)
            {
            case 1:
                build_pure_a(
                    g, ax, rank,
                    num_as[g], num_bs[g],
                    off_az[g], off_bz[g], off_wa[g], off_wb[g],
                    flat_azs, flat_bzs, flat_wa, flat_wb, orbsym,
                    basis, temp);
                break;
            case 2:
                build_pure_b(
                    g, bx, rank,
                    num_as[g], num_bs[g],
                    off_az[g], off_bz[g], off_wa[g], off_wb[g],
                    flat_azs, flat_bzs, flat_wa, flat_wb, orbsym,
                    basis, temp);
                break;
            case 3:
                build_mixed(
                    g, ax, bx, rank,
                    num_as[g], num_bs[g],
                    off_az[g], off_bz[g], off_wa[g], off_wb[g],
                    flat_azs, flat_bzs, flat_wa, flat_wb, orbsym,
                    basis, temp);
                break;
            default:
                break;
            }

            GroupArena &arena = net->arenas[g];

            arena.num_r1_jumps = temp.r1_jumps.size();
            arena.r1_jumps = arena.num_r1_jumps ? new TransR1[arena.num_r1_jumps] : nullptr;
            if (arena.num_r1_jumps)
                std::copy(temp.r1_jumps.begin(), temp.r1_jumps.end(), arena.r1_jumps);

            arena.num_r1_phases = temp.r1_phases.size();
            arena.r1_phases = arena.num_r1_phases ? new double[arena.num_r1_phases] : nullptr;
            if (arena.num_r1_phases)
                std::copy(temp.r1_phases.begin(), temp.r1_phases.end(), arena.r1_phases);

            arena.num_r2_jumps = temp.r2_jumps.size();
            arena.r2_jumps = arena.num_r2_jumps ? new TransR2[arena.num_r2_jumps] : nullptr;
            if (arena.num_r2_jumps)
                std::copy(temp.r2_jumps.begin(), temp.r2_jumps.end(), arena.r2_jumps);

            arena.num_r2_phases = temp.r2_phases.size();
            arena.r2_phases = arena.num_r2_phases ? new double[arena.num_r2_phases] : nullptr;
            if (arena.num_r2_phases)
                std::copy(temp.r2_phases.begin(), temp.r2_phases.end(), arena.r2_phases);

            arena.num_rn_jumps = temp.rn_jumps.size();
            arena.rn_jumps = arena.num_rn_jumps ? new TransRN[arena.num_rn_jumps] : nullptr;
            if (arena.num_rn_jumps)
                std::copy(temp.rn_jumps.begin(), temp.rn_jumps.end(), arena.rn_jumps);

            arena.num_rn_weights = temp.rn_weights.size();
            arena.rn_weights = arena.num_rn_weights ? new double[arena.num_rn_weights] : nullptr;
            if (arena.num_rn_weights)
                std::copy(temp.rn_weights.begin(), temp.rn_weights.end(), arena.rn_weights);

            arena.num_rn_phases = temp.rn_phases.size();
            arena.rn_phases = arena.num_rn_phases ? new double[arena.num_rn_phases] : nullptr;
            if (arena.num_rn_phases)
                std::copy(temp.rn_phases.begin(), temp.rn_phases.end(), arena.rn_phases);

            switch (type)
            {
            case 1:
                net->num_pure_a_routes[g] = temp.pure_routes.size();
                net->pure_a_routes[g] = net->num_pure_a_routes[g] ? new PureRoute[net->num_pure_a_routes[g]] : nullptr;
                if (net->num_pure_a_routes[g])
                    std::copy(temp.pure_routes.begin(), temp.pure_routes.end(), net->pure_a_routes[g]);
                break;
            case 2:
                net->num_pure_b_routes[g] = temp.pure_routes.size();
                net->pure_b_routes[g] = net->num_pure_b_routes[g] ? new PureRoute[net->num_pure_b_routes[g]] : nullptr;
                if (net->num_pure_b_routes[g])
                    std::copy(temp.pure_routes.begin(), temp.pure_routes.end(), net->pure_b_routes[g]);
                break;
            case 3:
                net->num_mixed_routes[g] = temp.mixed_routes.size();
                net->mixed_routes[g] = net->num_mixed_routes[g] ? new MixedRoute[net->num_mixed_routes[g]] : nullptr;
                if (net->num_mixed_routes[g])
                    std::copy(temp.mixed_routes.begin(), temp.mixed_routes.end(), net->mixed_routes[g]);
                break;
            default:
                break;
            }
        }

        return static_cast<void *>(net);
    }

    void destroy_svd_network(void *net_ptr)
    {
        if (!net_ptr)
            return;
        SVDNetwork *net = static_cast<SVDNetwork *>(net_ptr);

        if (net->arenas)
        {
            for (uint64 g = 0; g < net->ngs; ++g)
            {
                GroupArena &a = net->arenas[g];
                delete[] a.r1_jumps;
                delete[] a.r1_phases;
                delete[] a.r2_jumps;
                delete[] a.r2_phases;
                delete[] a.rn_jumps;
                delete[] a.rn_weights;
                delete[] a.rn_phases;

                if (net->pure_a_routes)
                    delete[] net->pure_a_routes[g];
                if (net->pure_b_routes)
                    delete[] net->pure_b_routes[g];
                if (net->mixed_routes)
                    delete[] net->mixed_routes[g];
            }
            delete[] net->arenas;
        }

        delete[] net->excit_types;
        delete[] net->group_ranks;
        delete[] net->pure_a_routes;
        delete[] net->num_pure_a_routes;
        delete[] net->pure_b_routes;
        delete[] net->num_pure_b_routes;
        delete[] net->mixed_routes;
        delete[] net->num_mixed_routes;

        delete net;
    }

    void get_diagonal_elements_svd_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        double *__restrict__ diags)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        const SVDNetwork *net = static_cast<const SVDNetwork *>(net_ptr);

        const uint32 *azs = net->azs;
        const uint32 *bzs = net->bzs;
        const double *cs = net->cs;
        const uint64 *gs = net->gs;
        const uint8 *types = net->excit_types;

        for (uint64 g = 0; g < net->ngs; ++g)
        {
            if (types[g] == 0)
            {
                const uint64 lb = gs[g];
                const uint64 rb = gs[g + 1];
#pragma omp parallel
                for (int64 i = 0; i < basis->num_blocks; ++i)
                {
                    const BlockDesc &block = basis->blocks[i];
#pragma omp for schedule(guided)
                    for (int64 a = 0; a < block.num_a; ++a)
                    {
                        uint32 astr = block.astrs[a];
                        int64 row_ptr = block.offset + a * block.num_b;
                        for (int64 b = 0; b < block.num_b; ++b)
                        {
                            uint32 bstr = block.bstrs[b];

                            double vt = 0.0;
                            for (uint64 k = lb; k < rb; ++k)
                            {
                                vt += cs[k] * phase(azs[k] & astr) * phase(bzs[k] & bstr);
                            }

                            diags[row_ptr + b] += vt;
                        }
                    }
                }
            }
        }
    }

    void hvec_svd_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        const SVDNetwork *net = static_cast<const SVDNetwork *>(net_ptr);

        const uint32 *azs = net->azs;
        const uint32 *bzs = net->bzs;
        const double *cs = net->cs;
        const uint64 *gs = net->gs;
        const uint8 *types = net->excit_types;

#pragma omp parallel for schedule(static)
        for (int64 i = 0; i < basis->dim; ++i)
        {
            dst[i] = 0.0;
        }

        for (uint64 g = 0; g < net->ngs; ++g)
        {
            const uint8 type = types[g];
            const uint64 lb = gs[g];
            const uint64 rb = gs[g + 1];
            const uint64 n_terms = rb - lb;

            if (type != 0 && n_terms == 0)
                continue;

            switch (type)
            {
            case 0:
                apply_diag_terms(
                    basis,
                    azs + lb, bzs + lb, cs + lb, n_terms,
                    src, dst);
                break;
            case 1:
                hvec_pure_a(
                    basis,
                    net->pure_a_routes[g],
                    net->num_pure_a_routes[g],
                    net->group_ranks[g],
                    net->arenas[g],
                    src, dst);
                break;
            case 2:
                hvec_pure_b(
                    basis,
                    net->pure_b_routes[g],
                    net->num_pure_b_routes[g],
                    net->group_ranks[g],
                    net->arenas[g],
                    src, dst);
                break;
            case 3:
                hvec_mixed(
                    basis,
                    net->mixed_routes[g],
                    net->num_mixed_routes[g],
                    net->group_ranks[g],
                    net->arenas[g],
                    src, dst);
                break;
            default:
                std::cerr << "Error: Unexpected type = " << type
                          << " at group g = " << g << " when hvec_svd"
                          << std::endl;
                break;
            }
        }
    }

    void tvec_svd_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        double *__restrict__ vec)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        const SVDNetwork *net = static_cast<const SVDNetwork *>(net_ptr);

        int type = net->excit_types[idx];

        switch (type)
        {
        case 1:
            tvec_pure_a(
                basis,
                net->pure_a_routes[idx],
                net->num_pure_a_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, vec);
            break;
        case 2:
            tvec_pure_b(
                basis,
                net->pure_b_routes[idx],
                net->num_pure_b_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, vec);
            break;
        case 3:
            tvec_mixed(
                basis,
                net->mixed_routes[idx],
                net->num_mixed_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, vec);
            break;
        default:
            std::cerr << "Error: Unexpected type = " << type
                      << " when tvec_svd"
                      << std::endl;
            break;
        }
    }

    double grad_svd_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const int64 idx,
        const double theta,
        const double *__restrict__ lp,
        const double *__restrict__ rp)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);
        const SVDNetwork *net = static_cast<const SVDNetwork *>(net_ptr);

        int type = net->excit_types[idx];

        switch (type)
        {
        case 1:
            return grad_pure_a(
                basis,
                net->pure_a_routes[idx],
                net->num_pure_a_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, lp, rp);
        case 2:
            return grad_pure_b(
                basis,
                net->pure_b_routes[idx],
                net->num_pure_b_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, lp, rp);
        case 3:
            return grad_mixed(
                basis,
                net->mixed_routes[idx],
                net->num_mixed_routes[idx],
                net->group_ranks[idx],
                net->arenas[idx],
                theta, lp, rp);
        default:
            std::cerr << "Error: Unexpected type = " << type
                      << " when expect_svd"
                      << std::endl;
            return 0.0;
        }
    }
}
