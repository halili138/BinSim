#include <cstdint>
#include <complex>
#include <vector>
#include <omp.h>
#include <bit>
#include <algorithm>
#include <iostream>
#include <cstring>

#define FORCE_INLINE inline __attribute__((always_inline))

using uint32 = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;

FORCE_INLINE int phase(uint32 x)
{
    return 1 - 2 * (std::popcount(x) & 1);
}

FORCE_INLINE int64 find_index(const uint32 *arr, int64 len, uint32 val)
{
    const uint32 *it = std::lower_bound(arr, arr + len, val);

    if (it != arr + len && *it == val)
    {
        return std::distance(arr, it);
    }

    return -1;
}

FORCE_INLINE int64 get_string_sym(uint32 str, const int64 *orbsym)
{
    int64 sym = 0;
    int64 pos = 0;
    while (str > 0)
    {
        if (str & 1)
        {
            sym ^= *(orbsym + pos);
        }
        str >>= 1;
        ++pos;
    }

    return sym;
}

FORCE_INLINE uint32 next_combination(uint32 v)
{
    if (v == 0)
        return 0;
    uint32 c = (v & -v);
    uint32 r = v + c;
    return (((r ^ v) >> 2) / c) | r;
}

struct BlockDesc
{
    int64 asym, bsym, num_a, num_b;
    const uint32 *astrs, *bstrs;
    int64 offset;
};

struct BasisManager
{
    std::vector<std::vector<uint32>> astrs_vec;
    std::vector<std::vector<uint32>> bstrs_vec;
    std::vector<BlockDesc> blocks;
    std::vector<std::vector<int64>> block_map;
    int64 dim;
};

struct SpinElement
{
    int64 ptr;
    uint32 str;
};

struct Trans
{
    SpinElement src;
    SpinElement dst;
};

struct PureRoute
{
    int64 block_src_idx; // 记录对应的 basis->blocks 索引
    std::vector<Trans> jumps;
};

struct MixedRoute
{
    std::vector<Trans> a_jumps;
    std::vector<Trans> b_jumps;
};

struct Network
{
    std::vector<int> excit_types; // 0: Diag, 1: Pure A, 2: Pure B, 3: Mix

    // 大小均为 nxs，如果是 type 1，就在 pure_a_routes[i] 里存数据
    std::vector<std::vector<PureRoute>> pure_a_routes;
    std::vector<std::vector<PureRoute>> pure_b_routes;
    std::vector<std::vector<MixedRoute>> mixed_routes;
};

static void build_pure_a_routes(
    const uint32 ax,
    const int64 *__restrict__ orbsym,
    const BasisManager *__restrict__ basis,
    std::vector<PureRoute> &routes)
{
    int64 axsym = get_string_sym(ax, orbsym);

    for (int64 i = 0; i < basis->blocks.size(); ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 asym_dst = block_src.asym ^ axsym;
        int64 bsym_dst = block_src.bsym; // β 无激发，对称性不变

        int64 j = basis->block_map[asym_dst][bsym_dst];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];

        PureRoute U;
        U.block_src_idx = i; // 记录来源 block

        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            uint32 astr_src = block_src.astrs[ia];
            uint32 astr_dst = astr_src ^ ax;

            // 原逻辑：确保只算一侧，避免厄米矩阵重复计算
            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            SpinElement src = {block_src.offset + ia * block_src.num_b, astr_src};
            SpinElement dst = {block_dst.offset + ja * block_dst.num_b, astr_dst};

            U.jumps.push_back({src, dst});
        }

        if (!U.jumps.empty())
        {
            routes.push_back(U);
        }
    }
}

static void build_pure_b_routes(
    const uint32 bx,
    const int64 *__restrict__ orbsym,
    const BasisManager *__restrict__ basis,
    std::vector<PureRoute> &routes)
{
    int64 bxsym = get_string_sym(bx, orbsym);

    for (int64 i = 0; i < basis->blocks.size(); ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 asym_dst = block_src.asym; // α 无激发，对称性不变
        int64 bsym_dst = block_src.bsym ^ bxsym;

        int64 j = basis->block_map[asym_dst][bsym_dst];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];

        PureRoute U;
        U.block_src_idx = i; // 记录来源 block

        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            uint32 bstr_src = block_src.bstrs[ib];
            uint32 bstr_dst = bstr_src ^ bx;

            // 根据原逻辑：纯 β 激发且 ax == 0 时，启用过滤
            if (bstr_src > bstr_dst)
                continue;

            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            SpinElement src = {ib, bstr_src};
            SpinElement dst = {jb, bstr_dst};

            U.jumps.push_back({src, dst});
        }

        if (!U.jumps.empty())
        {
            routes.push_back(U);
        }
    }
}

static void build_mixed_routes(
    const uint32 ax,
    const uint32 bx,
    const int64 *__restrict__ orbsym,
    const BasisManager *__restrict__ basis,
    std::vector<MixedRoute> &routes)
{
    int64 axsym = get_string_sym(ax, orbsym);
    int64 bxsym = get_string_sym(bx, orbsym);

    for (int64 i = 0; i < basis->blocks.size(); ++i)
    {
        const BlockDesc &block_src = basis->blocks[i];

        int64 asym_dst = block_src.asym ^ axsym;
        int64 bsym_dst = block_src.bsym ^ bxsym;

        int64 j = basis->block_map[asym_dst][bsym_dst];
        if (j == -1)
            continue;

        const BlockDesc &block_dst = basis->blocks[j];

        MixedRoute U;

        for (int64 ia = 0; ia < block_src.num_a; ++ia)
        {
            uint32 astr_src = block_src.astrs[ia];
            uint32 astr_dst = astr_src ^ ax;

            // 因为是 mix，ax 肯定不为 0
            if (astr_src > astr_dst)
                continue;

            int64 ja = find_index(block_dst.astrs, block_dst.num_a, astr_dst);
            if (ja == -1)
                continue;

            SpinElement src = {block_src.offset + ia * block_src.num_b, astr_src};
            SpinElement dst = {block_dst.offset + ja * block_dst.num_b, astr_dst};

            U.a_jumps.push_back({src, dst});
        }

        if (U.a_jumps.empty())
            continue;

        for (int64 ib = 0; ib < block_src.num_b; ++ib)
        {
            uint32 bstr_src = block_src.bstrs[ib];
            uint32 bstr_dst = bstr_src ^ bx;

            // 因为 ax != 0 且 bx != 0，原代码逻辑此处的 bstr_src > bstr_dst 不会触发。
            // 混合算符的厄米去重已经由 alpha 端承担了。
            int64 jb = find_index(block_dst.bstrs, block_dst.num_b, bstr_dst);
            if (jb == -1)
                continue;

            SpinElement src = {ib, bstr_src};
            SpinElement dst = {jb, bstr_dst};

            U.b_jumps.push_back({src, dst});
        }

        if (U.b_jumps.empty())
            continue;

        routes.push_back(U);
    }
}

static void apply_diag_terms(
    const BasisManager *__restrict__ basis,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
#pragma omp parallel
    {
        std::vector<int> phase_a(n_terms);
        int *__restrict__ pa = phase_a.data();
        const int *__restrict__ pa_cp = phase_a.data();

        for (const BlockDesc &block : basis->blocks)
        {
            const int64 num_b = block.num_b;
#pragma omp for schedule(guided) nowait
            for (int64 a = 0; a < block.num_a; ++a)
            {
                const uint32 astr = *(block.astrs + a);
                const int64 row_ptr = block.offset + a * num_b;

                for (int64 k = 0; k < n_terms; ++k)
                {
                    *(pa + k) = phase(*(azs + k) & astr);
                }

                for (int64 b = 0; b < num_b; ++b)
                {
                    const uint32 bstr = *(block.bstrs + b);
                    double vt = 0.0;

                    for (int64 k = 0; k < n_terms; ++k)
                    {
                        vt += *(cs + k) * *(pa_cp + k) * phase(*(bzs + k) & bstr);
                    }

                    *(dst + row_ptr + b) += *(src + row_ptr + b) * vt;
                }
            }
        }
    }
}

static void apply_pure_a_routes(
    const BasisManager *__restrict__ basis,
    const std::vector<PureRoute> &routes,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
#pragma omp parallel
    {
        std::vector<int> phase_a_src(n_terms);
        std::vector<int> phase_a_dst(n_terms);
        int *__restrict__ pa_src = phase_a_src.data();
        int *__restrict__ pa_dst = phase_a_dst.data();
        const int *__restrict__ pa_src_cp = phase_a_src.data();
        const int *__restrict__ pa_dst_cp = phase_a_dst.data();

        for (const PureRoute &R : routes)
        {
            const BlockDesc &block = basis->blocks[R.block_src_idx];
            const int64 num_b = block.num_b;

#pragma omp for schedule(guided) nowait
            for (const Trans &jump_a : R.jumps)
            {
                for (int64 k = 0; k < n_terms; ++k)
                {
                    const uint32 azk = *(azs + k);
                    *(pa_src + k) = phase(azk & jump_a.src.str);
                    *(pa_dst + k) = phase(azk & jump_a.dst.str);
                }

                const int64 ptr_src = jump_a.src.ptr;
                const int64 ptr_dst = jump_a.dst.ptr;

                for (int64 ib = 0; ib < num_b; ++ib)
                {
                    const uint32 bstr = block.bstrs[ib];
                    double i2j = 0.0, j2i = 0.0;

                    for (int64 k = 0; k < n_terms; ++k)
                    {
                        const int p_b = phase(*(bzs + k) & bstr);
                        const double ck = *(cs + k);
                        j2i += *(pa_dst_cp + k) * p_b * ck;
                        i2j += *(pa_src_cp + k) * p_b * ck;
                    }

                    const int64 gid_src = ptr_src + ib;
                    const int64 gid_dst = ptr_dst + ib;

                    *(dst + gid_src) += *(src + gid_dst) * j2i;
                    *(dst + gid_dst) += *(src + gid_src) * i2j;
                }
            }
        }
    }
}

static void apply_pure_b_routes(
    const BasisManager *__restrict__ basis,
    const std::vector<PureRoute> &routes,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
#pragma omp parallel
    {
        std::vector<int> phase_a(n_terms);
        int *__restrict__ pa = phase_a.data();
        const int *__restrict__ pa_cp = phase_a.data();

        for (const PureRoute &R : routes)
        {
            const BlockDesc &block = basis->blocks[R.block_src_idx];
            const int64 num_b = block.num_b;

#pragma omp for schedule(guided) nowait
            for (int64 a = 0; a < block.num_a; ++a)
            {
                const uint32 astr = block.astrs[a];
                const int64 row_ptr = block.offset + a * num_b;

                for (int64 k = 0; k < n_terms; ++k)
                {
                    *(pa + k) = phase(*(azs + k) & astr);
                }

                for (const Trans &jump_b : R.jumps)
                {
                    double i2j = 0.0, j2i = 0.0;

                    for (int64 k = 0; k < n_terms; ++k)
                    {
                        const int pa_k = *(pa_cp + k);
                        const uint32 bzk = *(bzs + k);
                        const double ck = *(cs + k);
                        j2i += pa_k * phase(bzk & jump_b.dst.str) * ck;
                        i2j += pa_k * phase(bzk & jump_b.src.str) * ck;
                    }

                    const int64 idx_src = row_ptr + jump_b.src.ptr;
                    const int64 idx_dst = row_ptr + jump_b.dst.ptr;

                    *(dst + idx_src) += *(src + idx_dst) * j2i;
                    *(dst + idx_dst) += *(src + idx_src) * i2j;
                }
            }
        }
    }
}

static void apply_mixed_routes(
    const std::vector<MixedRoute> &routes,
    const uint32 *__restrict__ azs,
    const uint32 *__restrict__ bzs,
    const double *__restrict__ cs,
    const int64 n_terms,
    const double *__restrict__ src,
    double *__restrict__ dst)
{
#pragma omp parallel
    {
        std::vector<int> phase_a_src(n_terms);
        std::vector<int> phase_a_dst(n_terms);
        int *__restrict__ pa_src = phase_a_src.data();
        int *__restrict__ pa_dst = phase_a_dst.data();
        const int *__restrict__ pa_src_cp = phase_a_src.data();
        const int *__restrict__ pa_dst_cp = phase_a_dst.data();

        for (const MixedRoute &R : routes)
        {
#pragma omp for schedule(guided) nowait
            for (const Trans &jump_a : R.a_jumps)
            {
                for (int64 k = 0; k < n_terms; ++k)
                {
                    const uint32 azk = *(azs + k);
                    *(pa_src + k) = phase(azk & jump_a.src.str);
                    *(pa_dst + k) = phase(azk & jump_a.dst.str);
                }

                for (const Trans &jump_b : R.b_jumps)
                {
                    double i2j = 0.0, j2i = 0.0;
                    for (int64 k = 0; k < n_terms; ++k)
                    {
                        const uint32 bzk = *(bzs + k);
                        const double ck = *(cs + k);
                        j2i += *(pa_dst_cp + k) * phase(bzk & jump_b.dst.str) * ck;
                        i2j += *(pa_src_cp + k) * phase(bzk & jump_b.src.str) * ck;
                    }

                    const int64 idx_src = jump_a.src.ptr + jump_b.src.ptr;
                    const int64 idx_dst = jump_a.dst.ptr + jump_b.dst.ptr;

                    *(dst + idx_src) += *(src + idx_dst) * j2i;
                    *(dst + idx_dst) += *(src + idx_src) * i2j;
                }
            }
        }
    }
}

extern "C"
{
    int64 get_subspace_dim(void *basis_ptr)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        return basis->dim;
    }

    void *create_basis_manager(
        const int64 norb,
        const int64 na,
        const int64 nb,
        const int64 total_sym,
        const int64 *__restrict__ orbsym,
        const int64 num_irreps)
    {
        BasisManager *basis = new BasisManager();
        try
        {
            basis->astrs_vec.resize(num_irreps);
            basis->bstrs_vec.resize(num_irreps);
            basis->block_map.assign(num_irreps, std::vector<int64>(num_irreps, -1));
            basis->dim = 0;

            uint32 a_str = (1UL << na) - 1;
            uint32 a_max = 1UL << norb;
            while (a_str < a_max)
            {
                int64 sym = get_string_sym(a_str, orbsym);
                if (sym < num_irreps)
                    basis->astrs_vec[sym].push_back(a_str);
                if (na == 0)
                    break;
                a_str = next_combination(a_str);
            }

            uint32 b_str = (1UL << nb) - 1;
            uint32 b_max = 1UL << norb;
            while (b_str < b_max)
            {
                int64 sym = get_string_sym(b_str, orbsym);
                if (sym < num_irreps)
                    basis->bstrs_vec[sym].push_back(b_str);
                if (nb == 0)
                    break;
                b_str = next_combination(b_str);
            }

            for (int64 asym = 0; asym < num_irreps; ++asym)
            {
                int64 bsym = total_sym ^ asym;
                if (bsym >= num_irreps)
                    continue;

                const auto &astrs = basis->astrs_vec[asym];
                const auto &bstrs = basis->bstrs_vec[bsym];

                if (!astrs.empty() && !bstrs.empty())
                {
                    BlockDesc block;
                    block.asym = asym;
                    block.bsym = bsym;
                    block.num_a = static_cast<int64>(astrs.size());
                    block.num_b = static_cast<int64>(bstrs.size());
                    block.astrs = astrs.data();
                    block.bstrs = bstrs.data();
                    block.offset = basis->dim;

                    basis->block_map[asym][bsym] = static_cast<int>(basis->blocks.size());
                    basis->blocks.push_back(block);
                    basis->dim += block.num_a * block.num_b;
                }
            }
        }
        catch (...)
        {
            delete basis;
            throw;
        }
        return static_cast<void *>(basis);
    }

    void destroy_basis_manager(void *basis_ptr)
    {
        if (basis_ptr)
        {
            delete static_cast<BasisManager *>(basis_ptr);
        }
    }

    void set_det_coeff(
        void *__restrict__ basis_ptr,
        const uint32 target_astr,
        const uint32 target_bstr,
        const double coeff,
        const int64 *__restrict__ orbsym,
        double *vec)
    {
        const BasisManager *basis = static_cast<const BasisManager *>(basis_ptr);

        int64 asym = get_string_sym(target_astr, orbsym);
        int64 bsym = get_string_sym(target_bstr, orbsym);

        if (asym >= basis->block_map.size() || bsym >= basis->block_map[asym].size())
            return;

        int64 block_idx = basis->block_map[asym][bsym];
        if (block_idx == -1)
            return;

        const BlockDesc &block = basis->blocks[block_idx];

        int64 ia = find_index(block.astrs, block.num_a, target_astr);
        if (ia == -1)
            return;

        int64 ib = find_index(block.bstrs, block.num_b, target_bstr);
        if (ib == -1)
            return;

        int64 gid = block.offset + ia * block.num_b + ib;

        *(vec + gid) += coeff;
    }

    void *build_routing_network(
        void *__restrict__ basis_ptr,
        const uint32 *__restrict__ axs,
        const uint32 *__restrict__ bxs,
        const int64 nxs,
        const int64 *__restrict__ orbsym)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);

        auto *net = new Network();
        net->excit_types.resize(nxs);
        net->pure_a_routes.resize(nxs);
        net->pure_b_routes.resize(nxs);
        net->mixed_routes.resize(nxs);

#pragma omp parallel for schedule(dynamic)
        for (int64 i = 0; i < nxs; ++i)
        {
            const uint32 ax = *(axs + i);
            const uint32 bx = *(bxs + i);

            if (ax == 0 && bx == 0)
            {
                net->excit_types[i] = 0; // Diag
            }
            else if (ax != 0 && bx == 0)
            {
                net->excit_types[i] = 1; // Pure A
                build_pure_a_routes(ax, orbsym, basis, net->pure_a_routes[i]);
            }
            else if (ax == 0 && bx != 0)
            {
                net->excit_types[i] = 2; // Pure B
                build_pure_b_routes(bx, orbsym, basis, net->pure_b_routes[i]);
            }
            else
            {
                net->excit_types[i] = 3; // Mix
                build_mixed_routes(ax, bx, orbsym, basis, net->mixed_routes[i]);
            }
        }
        return static_cast<void *>(net);
    }

    void destroy_routing_network(void *net_ptr)
    {
        if (net_ptr)
        {
            delete static_cast<Network *>(net_ptr);
        }
    }

    void hvec_network(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 *__restrict__ gs,
        const int64 ngs,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        const Network *net = static_cast<Network *>(net_ptr);

        for (int64 g = 0; g < ngs; ++g)
        {
            const int64 lb = *(gs + g);
            const int64 rb = *(gs + g + 1);
            int64 n_terms = rb - lb;
            if (n_terms == 0)
                continue;
            int route_type = net->excit_types[g];

            switch (route_type)
            {
            case 0:
                apply_diag_terms(
                    basis,
                    azs + lb, bzs + lb, cs + lb,
                    n_terms,
                    src, dst);
                break;
            case 1:
                apply_pure_a_routes(
                    basis, net->pure_a_routes[g],
                    azs + lb, bzs + lb, cs + lb,
                    n_terms,
                    src, dst);
                break;
            case 2:
                apply_pure_b_routes(
                    basis, net->pure_b_routes[g],
                    azs + lb, bzs + lb, cs + lb,
                    n_terms,
                    src, dst);
                break;
            case 3:
                apply_mixed_routes(
                    net->mixed_routes[g],
                    azs + lb, bzs + lb, cs + lb,
                    n_terms,
                    src, dst);
                break;
            default:
                std::cerr << "Error: Unexpected type = " << route_type
                          << " at group g = " << g << " when hvec"
                          << std::endl;
                break;
            }
        }
    }

    void get_diagonal_elements(
        void *__restrict__ basis_ptr,
        void *__restrict__ net_ptr,
        const uint32 *__restrict__ azs,
        const uint32 *__restrict__ bzs,
        const double *__restrict__ cs,
        const int64 *__restrict__ gs,
        const int64 ngs,
        double *__restrict__ diags)
    {
        const BasisManager *basis = static_cast<BasisManager *>(basis_ptr);
        const Network *net = static_cast<Network *>(net_ptr);

        for (int64 g = 0; g < ngs; ++g)
        {
            if (net->excit_types[g] == 0)
            {
                const int64 lb = *(gs + g);
                const int64 rb = *(gs + g + 1);
#pragma omp parallel
                {
                    for (int64 i = 0; i < basis->blocks.size(); ++i)
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
                                double vt = {};
                                for (int64 k = lb; k < rb; ++k)
                                {
                                    vt += *(cs + k) *
                                          phase(*(azs + k) & astr) *
                                          phase(*(bzs + k) & bstr);
                                }

                                *(diags + row_ptr + b) += vt;
                            }
                        }
                    }
                }
            }
        }
    }
}
