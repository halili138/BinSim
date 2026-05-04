#pragma once
#include "net.hpp"

template <typename Ti,
          typename Tv>
void build_pure_a(
    int64 g, Ti ax,
    int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb,
    const BasisManager<Ti> *basis,
    TempArena<Ti, Tv> &temp)
{
    int64 axsym = get_string_sym(ax, basis->orbsym);
    const Tv *wa = flat_wa + offset_wa;
    const Ti *za = flat_azs + offset_az;
    const Tv *wb = flat_wb + offset_wb;
    const Ti *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[i];

        int64 block_idx = (block_src.asym ^ axsym) * basis->num_irreps + block_src.bsym;
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[j];
        int64 valid_jumps = 0;

        uint64 start_j = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());

        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            Ti astr_src = block_src.astrs[ia];
            Ti astr_dst = astr_src ^ ax;

            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            append_jump_temp<Ti, Tv>(
                temp, rank,
                static_cast<Ti>(ia), static_cast<Ti>(ja),
                astr_src, na,
                wa, za);

            valid_jumps++;
        }

        if (valid_jumps == 0)
            continue;

        uint64 p_idx = (rank == 1) ? temp.r1_phases.size() : ((rank == 2) ? temp.r2_phases.size() : temp.rn_phases.size());

        PureRoute route{};
        route.block_src_idx = static_cast<uint16>(i);
        route.block_dst_idx = static_cast<uint16>(j);
        route.n = static_cast<uint32>(valid_jumps);
        route.jump_offset = start_j;
        route.phase_offset = p_idx;

        for (int64 b = 0; b < block_src.num_b; ++b)
        {
            append_phase_temp<Ti, Tv>(temp, rank, block_src.bstrs[b], nb, wb, zb);
        }

        temp.pure_routes.push_back(route);
    }
}

template <typename Ti,
          typename Tv>
void build_pure_b(
    int64 g, Ti bx,
    int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb,
    const BasisManager<Ti> *basis,
    TempArena<Ti, Tv> &temp)
{
    int64 bxsym = get_string_sym(bx, basis->orbsym);

    const Tv *wa = flat_wa + offset_wa;
    const Ti *za = flat_azs + offset_az;
    const Tv *wb = flat_wb + offset_wb;
    const Ti *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[i];

        int64 block_idx = block_src.asym * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[j];
        int64 valid_jumps = 0;

        uint64 start_j = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());

        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            Ti bstr_src = block_src.bstrs[ib];
            Ti bstr_dst = bstr_src ^ bx;

            if (bstr_src > bstr_dst)
                continue;

            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            append_jump_temp<Ti, Tv>(
                temp, rank,
                static_cast<Ti>(ib), static_cast<Ti>(jb),
                bstr_src, nb,
                wb, zb);

            valid_jumps++;
        }

        if (valid_jumps == 0)
            continue;

        uint64 p_idx = (rank == 1) ? temp.r1_phases.size() : ((rank == 2) ? temp.r2_phases.size() : temp.rn_phases.size());

        PureRoute route{};
        route.block_src_idx = static_cast<uint16>(i);
        route.block_dst_idx = static_cast<uint16>(j);
        route.n = static_cast<uint32>(valid_jumps);
        route.jump_offset = start_j;
        route.phase_offset = p_idx;

        for (int64 a = 0; a < block_src.num_a; ++a)
        {
            append_phase_temp<Ti, Tv>(temp, rank, block_src.astrs[a], na, wa, za);
        }

        temp.pure_routes.push_back(route);
    }
}

template <typename Ti,
          typename Tv>
void build_mixed(
    int64 g, Ti ax, Ti bx,
    int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb,
    const BasisManager<Ti> *basis,
    TempArena<Ti, Tv> &temp)
{
    int64 axsym = get_string_sym(ax, basis->orbsym);
    int64 bxsym = get_string_sym(bx, basis->orbsym);

    const Tv *wa = flat_wa + offset_wa;
    const Ti *za = flat_azs + offset_az;
    const Tv *wb = flat_wb + offset_wb;
    const Ti *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc<Ti> &block_src = basis->blocks[i];

        int64 block_idx = (block_src.asym ^ axsym) * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc<Ti> &block_dst = basis->blocks[j];

        uint64 start_j_a = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());
        uint64 start_w_a = temp.rn_weights.size();

        int64 valid_na = 0;
        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            Ti astr_src = block_src.astrs[ia];
            Ti astr_dst = astr_src ^ ax;

            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            append_jump_temp<Ti, Tv>(
                temp, rank,
                static_cast<Ti>(ia), static_cast<Ti>(ja),
                astr_src, na,
                wa, za);

            valid_na++;
        }

        if (valid_na == 0)
            continue;

        uint64 start_j_b = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());
        int64 valid_nb = 0;
        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            Ti bstr_src = block_src.bstrs[ib];
            Ti bstr_dst = bstr_src ^ bx;

            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            append_jump_temp<Ti, Tv>(
                temp, rank,
                static_cast<Ti>(ib), static_cast<Ti>(jb), bstr_src, nb,
                wb, zb);

            valid_nb++;
        }

        if (valid_nb == 0)
        {
            temp.rollback_jumps(rank, start_j_a, start_w_a);
            continue;
        }

        MixedRoute route{};
        route.na = static_cast<uint32>(valid_na);
        route.nb = static_cast<uint32>(valid_nb);
        route.block_src_idx = static_cast<uint16>(i);
        route.block_dst_idx = static_cast<uint16>(j);
        route.a_jump_offset = start_j_a;
        route.b_jump_offset = start_j_b;

        temp.mixed_routes.push_back(route);
    }
}

template <typename Ti, typename Tv>
SVDNetwork<Ti, Tv> *create_svd_network(
    const BasisManager<Ti> *basis,
    int64 ngs,
    const Ti *axs,
    const Ti *bxs,
    const int64 *ranks,
    const int64 *num_as,
    const int64 *num_bs,
    const Ti *flat_azs,
    const Ti *flat_bzs,
    const Tv *flat_wa,
    const Tv *flat_wb)
{
    SVDNetwork<Ti, Tv> *net = new SVDNetwork<Ti, Tv>();

    net->ngs = ngs;
    net->excit_types = new uint8[ngs]();
    net->group_ranks = new uint16[ngs]();
    net->arenas = new GroupArena<Ti, Tv>[ngs]();

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
        Ti ax = axs[g], bx = bxs[g];
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

        TempArena<Ti, Tv> temp;

        switch (type)
        {
        case 1:
            build_pure_a<Ti, Tv>(
                g, ax, rank,
                num_as[g], num_bs[g],
                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                flat_azs, flat_bzs, flat_wa, flat_wb,
                basis, temp);
            break;
        case 2:
            build_pure_b<Ti, Tv>(
                g, bx, rank,
                num_as[g], num_bs[g],
                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                flat_azs, flat_bzs, flat_wa, flat_wb,
                basis, temp);
            break;
        case 0:
        case 3:
            build_mixed<Ti, Tv>(
                g, ax, bx, rank,
                num_as[g], num_bs[g],
                off_az[g], off_bz[g], off_wa[g], off_wb[g],
                flat_azs, flat_bzs, flat_wa, flat_wb,
                basis, temp);
            break;
        default:
            break;
        }

        GroupArena<Ti, Tv> &arena = net->arenas[g];

        arena.num_r1_jumps = temp.r1_jumps.size();
        arena.r1_jumps = arena.num_r1_jumps ? new TransR1<Ti, Tv>[arena.num_r1_jumps] : nullptr;
        if (arena.num_r1_jumps)
            std::copy(temp.r1_jumps.begin(), temp.r1_jumps.end(), arena.r1_jumps);

        arena.num_r1_phases = temp.r1_phases.size();
        arena.r1_phases = arena.num_r1_phases ? new Tv[arena.num_r1_phases] : nullptr;
        if (arena.num_r1_phases)
            std::copy(temp.r1_phases.begin(), temp.r1_phases.end(), arena.r1_phases);

        arena.num_r2_jumps = temp.r2_jumps.size();
        arena.r2_jumps = arena.num_r2_jumps ? new TransR2<Ti, Tv>[arena.num_r2_jumps] : nullptr;
        if (arena.num_r2_jumps)
            std::copy(temp.r2_jumps.begin(), temp.r2_jumps.end(), arena.r2_jumps);

        arena.num_r2_phases = temp.r2_phases.size();
        arena.r2_phases = arena.num_r2_phases ? new Tv[arena.num_r2_phases] : nullptr;
        if (arena.num_r2_phases)
            std::copy(temp.r2_phases.begin(), temp.r2_phases.end(), arena.r2_phases);

        arena.num_rn_jumps = temp.rn_jumps.size();
        arena.rn_jumps = arena.num_rn_jumps ? new TransRN<Ti>[arena.num_rn_jumps] : nullptr;
        if (arena.num_rn_jumps)
            std::copy(temp.rn_jumps.begin(), temp.rn_jumps.end(), arena.rn_jumps);

        arena.num_rn_weights = temp.rn_weights.size();
        arena.rn_weights = arena.num_rn_weights ? new Tv[arena.num_rn_weights] : nullptr;
        if (arena.num_rn_weights)
            std::copy(temp.rn_weights.begin(), temp.rn_weights.end(), arena.rn_weights);

        arena.num_rn_phases = temp.rn_phases.size();
        arena.rn_phases = arena.num_rn_phases ? new Tv[arena.num_rn_phases] : nullptr;
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
        case 0:
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

    return net;
}

template <typename Ti, typename Tv>
void destroy_svd_network(SVDNetwork<Ti, Tv> *net)
{
    if (net->arenas)
    {
        for (uint64 g = 0; g < net->ngs; ++g)
        {
            GroupArena<Ti, Tv> &a = net->arenas[g];
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
