#include "net.hpp"

FORCE_INLINE double calc_w_r1(
    uint32 str, int64 len,
    const double *w, const uint32 *z)
{
    double res = 0.0;
    for (int64 k = 0; k < len; ++k)
        res += w[k] * phase(z[k] & str);
    return res;
}

FORCE_INLINE void calc_w_r2(
    uint32 str, int64 len,
    const double *w, const uint32 *z,
    double &w0, double &w1)
{
    w0 = 0.0;
    w1 = 0.0;
    for (int64 k = 0; k < len; ++k)
    {
        int p = phase(z[k] & str);
        w0 += w[k] * p;
        w1 += w[k + len] * p;
    }
}

FORCE_INLINE void push_w_rn(
    uint32 str, int64 len, int rank,
    const double *w, const uint32 *z,
    std::vector<double> &out_vec)
{
    for (int r = 0; r < rank; ++r)
    {
        double res = 0.0;
        const double *wr = w + r * len;
        for (int64 k = 0; k < len; ++k)
            res += wr[k] * phase(z[k] & str);
        out_vec.push_back(res);
    }
}

FORCE_INLINE void append_jump_temp(
    TempArena &temp, int rank,
    uint32 src_idx, uint32 dst_idx, uint32 str, int64 len,
    const double *w, const uint32 *z)
{
    if (rank == 1)
    {
        temp.r1_jumps.push_back({src_idx, dst_idx, calc_w_r1(str, len, w, z)});
    }
    else if (rank == 2)
    {
        double w0, w1;
        calc_w_r2(str, len, w, z, w0, w1);
        temp.r2_jumps.push_back({src_idx, dst_idx, w0, w1});
    }
    else
    {
        uint64 w_offset = temp.rn_weights.size();
        push_w_rn(str, len, rank, w, z, temp.rn_weights);
        temp.rn_jumps.push_back({src_idx, dst_idx, w_offset});
    }
}

FORCE_INLINE void append_phase_temp(
    TempArena &temp, int rank, uint32 str, int64 len,
    const double *w, const uint32 *z)
{
    if (rank == 1)
    {
        temp.r1_phases.push_back(calc_w_r1(str, len, w, z));
    }
    else if (rank == 2)
    {
        double p0, p1;
        calc_w_r2(str, len, w, z, p0, p1);
        temp.r2_phases.push_back(p0);
        temp.r2_phases.push_back(p1);
    }
    else
    {
        push_w_rn(str, len, rank, w, z, temp.rn_phases);
    }
}

void build_pure_a(
    int64 g, uint32 ax, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp)
{
    int64 axsym = get_string_sym(ax, orbsym);
    const double *wa = flat_wa + offset_wa;
    const uint32 *za = flat_azs + offset_az;
    const double *wb = flat_wb + offset_wb;
    const uint32 *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 block_idx = (block_src.asym ^ axsym) * basis->num_irreps + block_src.bsym;
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];
        int64 valid_jumps = 0;

        uint64 start_j = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());

        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            uint32 astr_src = block_src.astrs[ia];
            uint32 astr_dst = astr_src ^ ax;

            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            append_jump_temp(
                temp, rank,
                static_cast<uint32>(ia), static_cast<uint32>(ja), astr_src, na,
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
            append_phase_temp(temp, rank, block_src.bstrs[b], nb, wb, zb);
        }

        temp.pure_routes.push_back(route);
    }
}

void build_pure_b(
    int64 g, uint32 bx, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp)
{
    int64 bxsym = get_string_sym(bx, orbsym);
    const double *wa = flat_wa + offset_wa;
    const uint32 *za = flat_azs + offset_az;
    const double *wb = flat_wb + offset_wb;
    const uint32 *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 block_idx = block_src.asym * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];
        int64 valid_jumps = 0;

        uint64 start_j = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());

        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            uint32 bstr_src = block_src.bstrs[ib];
            uint32 bstr_dst = bstr_src ^ bx;

            if (bstr_src > bstr_dst)
                continue;

            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            append_jump_temp(
                temp, rank,
                static_cast<uint32>(ib), static_cast<uint32>(jb), bstr_src, nb,
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
            append_phase_temp(temp, rank, block_src.astrs[a], na, wa, za);
        }

        temp.pure_routes.push_back(route);
    }
}

void build_mixed(
    int64 g, uint32 ax, uint32 bx, int rank, int64 na, int64 nb,
    int64 offset_az, int64 offset_bz,
    int64 offset_wa, int64 offset_wb,
    const uint32 *flat_azs, const uint32 *flat_bzs,
    const double *flat_wa, const double *flat_wb,
    const int64 *orbsym, const BasisManager *basis, TempArena &temp)
{
    int64 axsym = get_string_sym(ax, orbsym);
    int64 bxsym = get_string_sym(bx, orbsym);
    const double *wa = flat_wa + offset_wa;
    const uint32 *za = flat_azs + offset_az;
    const double *wb = flat_wb + offset_wb;
    const uint32 *zb = flat_bzs + offset_bz;

    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 block_idx = (block_src.asym ^ axsym) * basis->num_irreps + (block_src.bsym ^ bxsym);
        int64 j = basis->block_map[block_idx];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];

        uint64 start_j_a = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());
        uint64 start_w_a = temp.rn_weights.size();

        int64 valid_na = 0;
        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            uint32 astr_src = block_src.astrs[ia];
            uint32 astr_dst = astr_src ^ ax;

            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            append_jump_temp(
                temp, rank,
                static_cast<uint32>(ia), static_cast<uint32>(ja), astr_src, na,
                wa, za);

            valid_na++;
        }

        if (valid_na == 0)
            continue;

        uint64 start_j_b = (rank == 1) ? temp.r1_jumps.size() : ((rank == 2) ? temp.r2_jumps.size() : temp.rn_jumps.size());
        int64 valid_nb = 0;
        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            uint32 bstr_src = block_src.bstrs[ib];
            uint32 bstr_dst = bstr_src ^ bx;

            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            append_jump_temp(
                temp, rank,
                static_cast<uint32>(ib), static_cast<uint32>(jb), bstr_src, nb,
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
