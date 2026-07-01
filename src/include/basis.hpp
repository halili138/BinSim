#pragma once
#include "common.hpp"
#include <unordered_map>
#include <numeric>
#include <cmath>

template <typename T>
struct BlockDesc
{
    int64 asym, bsym, num_a, num_b;
    const T *astrs, *bstrs;
    int64 offset;
};

template <typename Ti>
struct BasisView
{
    const BlockDesc<Ti> *blocks;
    const BlockDesc<Ti> *src_blocks;
    int64 num_blocks;
    int max_a_count, max_b_count;
    const int64 *block_map;
    int64 num_irreps;
    const int *a_idx_map;
    const int *b_idx_map;
    const int64 *src_offsets;
};

template <typename T>
struct BasisManager
{
    T *all_astrs = nullptr;
    T *all_bstrs = nullptr;

    T **astrs_vec = nullptr;
    T **bstrs_vec = nullptr;

    int64 *num_astrs = nullptr;
    int64 *num_bstrs = nullptr;

    BlockDesc<T> *blocks = nullptr;
    int64 num_blocks = {};

    int64 *orbsym = nullptr;
    int64 *block_map = nullptr;
    int64 num_irreps = {};
    int64 physical_num_irreps = {};  // 0 = non-partitioned; >0 = physical irrep count
    int64 total_sym = {};

    int64 dim = {};
    int64 norb = {};
    int max_a_count = {};
    int max_b_count = {};

    int *a_idx_map = nullptr;
    int *b_idx_map = nullptr;

    std::vector<int64> _src_offsets;
    BasisView<T> view;

    void _init_view()
    {
        int64 total = num_irreps * num_irreps;
        _src_offsets.resize(total, -1);
        for (int64 i = 0; i < num_blocks; ++i)
        {
            const auto &blk = blocks[i];
            int64 bid = blk.asym * num_irreps + blk.bsym;
            _src_offsets[bid] = blk.offset;
        }
        view = BasisView<T>{blocks, nullptr, num_blocks,
                            max_a_count, max_b_count,
                            block_map, num_irreps,
                            a_idx_map, b_idx_map, _src_offsets.data()};
    }

    void clear()
    {
        delete[] all_astrs;
        all_astrs = nullptr;
        delete[] all_bstrs;
        all_bstrs = nullptr;
        delete[] astrs_vec;
        astrs_vec = nullptr;
        delete[] bstrs_vec;
        bstrs_vec = nullptr;
        delete[] num_astrs;
        num_astrs = nullptr;
        delete[] num_bstrs;
        num_bstrs = nullptr;
        delete[] blocks;
        blocks = nullptr;
        delete[] orbsym;
        orbsym = nullptr;
        delete[] block_map;
        block_map = nullptr;
        delete[] a_idx_map;
        a_idx_map = nullptr;
        delete[] b_idx_map;
        b_idx_map = nullptr;
        num_blocks = 0;
        num_irreps = 0;
        physical_num_irreps = 0;
        total_sym = 0;
        dim = 0;
        norb = 0;
        max_a_count = 0;
        max_b_count = 0;
    }
};

template <typename Ti>
int64 get_subspace_dim_tmpl(const BasisManager<Ti> *basis)
{
    return basis->dim;
}

template <typename Ti>
void *create_basis_manager_tmpl(
    const int64 norb,
    const int64 na,
    const int64 nb,
    const int64 total_sym,
    const int64 *__restrict__ orbsym,
    const int64 num_irreps)
{
    BasisManager<Ti> *basis = new BasisManager<Ti>();

    try
    {
        basis->num_irreps = num_irreps;
        basis->physical_num_irreps = num_irreps;  // non-partitioned: physical == virtual
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;
        basis->total_sym = total_sym;

        basis->num_astrs = new int64[num_irreps]();
        basis->num_bstrs = new int64[num_irreps]();

        Ti a_str = (static_cast<Ti>(1) << na) - 1;
        Ti a_max = static_cast<Ti>(1) << norb;
        int64 total_a_strings = 0;
        while (a_str < a_max)
        {
            int64 sym = get_string_sym(a_str, orbsym);
            if (sym < num_irreps)
            {
                basis->num_astrs[sym]++;
                total_a_strings++;
            }
            if (na == 0)
                break;
            a_str = next_combination(a_str);
        }

        Ti b_str = (static_cast<Ti>(1) << nb) - 1;
        Ti b_max = static_cast<Ti>(1) << norb;
        int64 total_b_strings = 0;
        while (b_str < b_max)
        {
            int64 sym = get_string_sym(b_str, orbsym);
            if (sym < num_irreps)
            {
                basis->num_bstrs[sym]++;
                total_b_strings++;
            }
            if (nb == 0)
                break;
            b_str = next_combination(b_str);
        }

        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym < num_irreps && basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
            {
                basis->num_blocks++;
            }
        }

        basis->all_astrs = new Ti[total_a_strings];
        basis->all_bstrs = new Ti[total_b_strings];

        basis->astrs_vec = new Ti *[num_irreps];
        basis->bstrs_vec = new Ti *[num_irreps];

        basis->blocks = new BlockDesc<Ti>[basis->num_blocks];
        basis->orbsym = new int64[norb];
        std::copy(orbsym, orbsym + norb, basis->orbsym);
        basis->block_map = new int64[num_irreps * num_irreps];
        std::fill_n(basis->block_map, num_irreps * num_irreps, -1);

        int64 a_offset = 0;
        int64 b_offset = 0;
        for (int64 i = 0; i < num_irreps; ++i)
        {
            basis->astrs_vec[i] = basis->all_astrs + a_offset;
            a_offset += basis->num_astrs[i];

            basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
            b_offset += basis->num_bstrs[i];
        }

        int64 *a_idx = new int64[num_irreps]();
        a_str = (static_cast<Ti>(1) << na) - 1;
        while (a_str < a_max)
        {
            int64 sym = get_string_sym(a_str, orbsym);
            if (sym < num_irreps)
            {
                basis->astrs_vec[sym][a_idx[sym]++] = a_str;
            }
            if (na == 0)
                break;
            a_str = next_combination(a_str);
        }
        delete[] a_idx;

        int64 *b_idx = new int64[num_irreps]();
        b_str = (static_cast<Ti>(1) << nb) - 1;
        while (b_str < b_max)
        {
            int64 sym = get_string_sym(b_str, orbsym);
            if (sym < num_irreps)
            {
                basis->bstrs_vec[sym][b_idx[sym]++] = b_str;
            }
            if (nb == 0)
                break;
            b_str = next_combination(b_str);
        }
        delete[] b_idx;

        int64 block_counter = 0;
        basis->dim = 0;
        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym >= num_irreps)
                continue;

            if (basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
            {
                BlockDesc<Ti> &block = basis->blocks[block_counter];
                block.asym = asym;
                block.bsym = bsym;
                block.num_a = basis->num_astrs[asym];
                block.num_b = basis->num_bstrs[bsym];
                block.astrs = basis->astrs_vec[asym];
                block.bstrs = basis->bstrs_vec[bsym];
                block.offset = basis->dim;

                basis->block_map[asym * num_irreps + bsym] = block_counter;

                block_counter++;
                basis->dim += block.num_a * block.num_b;
            }
        }

        basis->max_a_count = 0;
        basis->max_b_count = 0;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            if (basis->blocks[i].num_a > basis->max_a_count)
                basis->max_a_count = basis->blocks[i].num_a;
            if (basis->blocks[i].num_b > basis->max_b_count)
                basis->max_b_count = basis->blocks[i].num_b;
        }

        int32 map_size = 1 << norb;
        int32 *a_map = new int32[map_size];
        int32 *b_map = new int32[map_size];
        std::fill(a_map, a_map + map_size, -1);
        std::fill(b_map, b_map + map_size, -1);

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            for (int32 a = 0; a < basis->blocks[i].num_a; ++a)
                a_map[basis->blocks[i].astrs[a]] = a;
            for (int32 b = 0; b < basis->blocks[i].num_b; ++b)
                b_map[basis->blocks[i].bstrs[b]] = b;
        }
        basis->a_idx_map = a_map;
        basis->b_idx_map = b_map;
        basis->_init_view();
    }
    catch (...)
    {
        basis->clear();
        delete basis;
        throw;
    }

    return static_cast<void *>(basis);
}

template <typename Ti,
          typename Tv>
void set_det_coeff(
    const BasisManager<Ti> *basis,
    const Ti target_astr,
    const Ti target_bstr,
    const Tv coeff,
    Tv *vec)
{
    int64 asym = get_string_sym(target_astr, basis->orbsym);
    int64 bsym = get_string_sym(target_bstr, basis->orbsym);

    if (asym >= basis->num_irreps || bsym >= basis->num_irreps)
        return;

    int64 block_idx = basis->block_map[asym * basis->num_irreps + bsym];
    if (block_idx == -1)
        return;

    const BlockDesc<Ti> &block = basis->blocks[block_idx];

    int64 ia = find_index(block.astrs, block.num_a, target_astr);
    if (ia == -1)
        return;

    int64 ib = find_index(block.bstrs, block.num_b, target_bstr);
    if (ib == -1)
        return;

    int64 gid = block.offset + ia * block.num_b + ib;

    *(vec + gid) += coeff;
}

template <typename Ti>
void *create_custom_basis_manager_tmpl(
    int64 norb,
    const Ti *input_astrs, int64 num_astrs_total,
    const Ti *input_bstrs, int64 num_bstrs_total,
    const int64 *orbsym, int64 total_sym, int64 num_irreps)
{
    BasisManager<Ti> *basis = new BasisManager<Ti>();

    try
    {
        basis->num_irreps = num_irreps;
        basis->physical_num_irreps = num_irreps;  // non-partitioned: physical == virtual
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;

        basis->num_astrs = new int64[num_irreps]();
        basis->num_bstrs = new int64[num_irreps]();

        for (int64 i = 0; i < num_astrs_total; ++i)
        {
            int64 sym = get_string_sym(input_astrs[i], orbsym);
            if (sym < num_irreps)
                basis->num_astrs[sym]++;
        }

        for (int64 i = 0; i < num_bstrs_total; ++i)
        {
            int64 sym = get_string_sym(input_bstrs[i], orbsym);
            if (sym < num_irreps)
                basis->num_bstrs[sym]++;
        }

        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym < num_irreps && basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
            {
                basis->num_blocks++;
            }
        }

        basis->all_astrs = new Ti[num_astrs_total];
        basis->all_bstrs = new Ti[num_bstrs_total];

        basis->astrs_vec = new Ti *[num_irreps];
        basis->bstrs_vec = new Ti *[num_irreps];

        basis->blocks = new BlockDesc<Ti>[basis->num_blocks];
        basis->orbsym = new int64[norb];
        std::copy(orbsym, orbsym + norb, basis->orbsym);
        basis->block_map = new int64[num_irreps * num_irreps];
        std::fill_n(basis->block_map, num_irreps * num_irreps, -1);

        int64 a_offset = 0;
        int64 b_offset = 0;
        for (int64 i = 0; i < num_irreps; ++i)
        {
            basis->astrs_vec[i] = basis->all_astrs + a_offset;
            a_offset += basis->num_astrs[i];

            basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
            b_offset += basis->num_bstrs[i];
        }

        int64 *a_idx = new int64[num_irreps]();
        for (int64 i = 0; i < num_astrs_total; ++i)
        {
            int64 sym = get_string_sym(input_astrs[i], orbsym);
            if (sym < num_irreps)
            {
                basis->astrs_vec[sym][a_idx[sym]++] = input_astrs[i];
            }
        }
        delete[] a_idx;

        int64 *b_idx = new int64[num_irreps]();
        for (int64 i = 0; i < num_bstrs_total; ++i)
        {
            int64 sym = get_string_sym(input_bstrs[i], orbsym);
            if (sym < num_irreps)
            {
                basis->bstrs_vec[sym][b_idx[sym]++] = input_bstrs[i];
            }
        }
        delete[] b_idx;

        for (int64 i = 0; i < num_irreps; ++i)
        {
            if (basis->num_astrs[i] > 0)
                std::sort(basis->astrs_vec[i], basis->astrs_vec[i] + basis->num_astrs[i]);

            if (basis->num_bstrs[i] > 0)
                std::sort(basis->bstrs_vec[i], basis->bstrs_vec[i] + basis->num_bstrs[i]);
        }

        int64 block_counter = 0;
        basis->dim = 0;
        for (int64 asym = 0; asym < num_irreps; ++asym)
        {
            int64 bsym = total_sym ^ asym;
            if (bsym >= num_irreps)
                continue;

            if (basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
            {
                BlockDesc<Ti> &block = basis->blocks[block_counter];
                block.asym = asym;
                block.bsym = bsym;
                block.num_a = basis->num_astrs[asym];
                block.num_b = basis->num_bstrs[bsym];
                block.astrs = basis->astrs_vec[asym];
                block.bstrs = basis->bstrs_vec[bsym];
                block.offset = basis->dim;

                basis->block_map[asym * num_irreps + bsym] = block_counter;

                block_counter++;
                basis->dim += block.num_a * block.num_b;
            }
        }

        basis->max_a_count = 0;
        basis->max_b_count = 0;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            if (basis->blocks[i].num_a > basis->max_a_count)
                basis->max_a_count = basis->blocks[i].num_a;
            if (basis->blocks[i].num_b > basis->max_b_count)
                basis->max_b_count = basis->blocks[i].num_b;
        }

        int32 map_size = 1 << norb;
        int32 *a_map = new int32[map_size];
        int32 *b_map = new int32[map_size];
        std::fill(a_map, a_map + map_size, -1);
        std::fill(b_map, b_map + map_size, -1);

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            for (int32 a = 0; a < basis->blocks[i].num_a; ++a)
                a_map[basis->blocks[i].astrs[a]] = a;
            for (int32 b = 0; b < basis->blocks[i].num_b; ++b)
                b_map[basis->blocks[i].bstrs[b]] = b;
        }
        basis->a_idx_map = a_map;
        basis->b_idx_map = b_map;
        basis->_init_view();
    }
    catch (...)
    {
        basis->clear();
        delete basis;
        throw;
    }

    return static_cast<void *>(basis);
}

template <typename Ti>
void *create_partitioned_basis_manager_tmpl(
    const int64 norb,
    const int64 na,
    const int64 nb,
    const int64 physical_total_sym,
    const int64 *__restrict__ physical_orbsym,
    const int64 *__restrict__ virtual_orbsym,
    const int64 physical_num_irreps,
    const int64 virtual_num_irreps)
{
    BasisManager<Ti> *basis = new BasisManager<Ti>();

    try
    {
        if ((physical_num_irreps & (physical_num_irreps - 1)) != 0 ||
            (virtual_num_irreps & (virtual_num_irreps - 1)) != 0)
            throw std::runtime_error("partitioned basis requires power-of-two physical and virtual irreps");

        int64 physical_bits = 0;
        while ((static_cast<int64>(1) << physical_bits) < physical_num_irreps)
            physical_bits++;

        const int64 combined_num_irreps = physical_num_irreps * virtual_num_irreps;
        auto combine_sym = [physical_bits](int64 psym, int64 vsym)
        {
            return psym | (vsym << physical_bits);
        };
        auto physical_part = [physical_num_irreps](int64 combined_sym)
        {
            return combined_sym & (physical_num_irreps - 1);
        };

        basis->num_irreps = combined_num_irreps;
        basis->physical_num_irreps = physical_num_irreps;
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;
        basis->total_sym = physical_total_sym;

        basis->num_astrs = new int64[combined_num_irreps]();
        basis->num_bstrs = new int64[combined_num_irreps]();
        basis->orbsym = new int64[norb];
        for (int64 i = 0; i < norb; ++i)
            basis->orbsym[i] = combine_sym(physical_orbsym[i], virtual_orbsym[i]);

        Ti a_str = (static_cast<Ti>(1) << na) - 1;
        Ti a_max = static_cast<Ti>(1) << norb;
        int64 total_a_strings = 0;
        while (a_str < a_max)
        {
            int64 psym = get_string_sym(a_str, physical_orbsym);
            int64 vsym = get_string_sym(a_str, virtual_orbsym);
            int64 sym = combine_sym(psym, vsym);
            if (psym < physical_num_irreps && vsym < virtual_num_irreps)
            {
                basis->num_astrs[sym]++;
                total_a_strings++;
            }
            if (na == 0)
                break;
            a_str = next_combination(a_str);
        }

        Ti b_str = (static_cast<Ti>(1) << nb) - 1;
        Ti b_max = static_cast<Ti>(1) << norb;
        int64 total_b_strings = 0;
        while (b_str < b_max)
        {
            int64 psym = get_string_sym(b_str, physical_orbsym);
            int64 vsym = get_string_sym(b_str, virtual_orbsym);
            int64 sym = combine_sym(psym, vsym);
            if (psym < physical_num_irreps && vsym < virtual_num_irreps)
            {
                basis->num_bstrs[sym]++;
                total_b_strings++;
            }
            if (nb == 0)
                break;
            b_str = next_combination(b_str);
        }

        for (int64 asym = 0; asym < combined_num_irreps; ++asym)
        {
            int64 a_phys = physical_part(asym);
            int64 b_phys_required = physical_total_sym ^ a_phys;
            for (int64 bsym = 0; bsym < combined_num_irreps; ++bsym)
            {
                if (physical_part(bsym) == b_phys_required &&
                    basis->num_astrs[asym] > 0 && basis->num_bstrs[bsym] > 0)
                    basis->num_blocks++;
            }
        }

        basis->all_astrs = new Ti[total_a_strings];
        basis->all_bstrs = new Ti[total_b_strings];
        basis->astrs_vec = new Ti *[combined_num_irreps];
        basis->bstrs_vec = new Ti *[combined_num_irreps];
        basis->blocks = new BlockDesc<Ti>[basis->num_blocks];
        basis->block_map = new int64[combined_num_irreps * combined_num_irreps];
        std::fill_n(basis->block_map, combined_num_irreps * combined_num_irreps, -1);

        int64 a_offset = 0;
        int64 b_offset = 0;
        for (int64 i = 0; i < combined_num_irreps; ++i)
        {
            basis->astrs_vec[i] = basis->all_astrs + a_offset;
            a_offset += basis->num_astrs[i];
            basis->bstrs_vec[i] = basis->all_bstrs + b_offset;
            b_offset += basis->num_bstrs[i];
        }

        std::vector<int64> a_idx(combined_num_irreps, 0), b_idx(combined_num_irreps, 0);
        a_str = (static_cast<Ti>(1) << na) - 1;
        while (a_str < a_max)
        {
            int64 sym = get_string_sym(a_str, basis->orbsym);
            if (sym < combined_num_irreps)
                basis->astrs_vec[sym][a_idx[sym]++] = a_str;
            if (na == 0)
                break;
            a_str = next_combination(a_str);
        }
        b_str = (static_cast<Ti>(1) << nb) - 1;
        while (b_str < b_max)
        {
            int64 sym = get_string_sym(b_str, basis->orbsym);
            if (sym < combined_num_irreps)
                basis->bstrs_vec[sym][b_idx[sym]++] = b_str;
            if (nb == 0)
                break;
            b_str = next_combination(b_str);
        }

        int64 block_counter = 0;
        for (int64 asym = 0; asym < combined_num_irreps; ++asym)
        {
            int64 a_phys = physical_part(asym);
            int64 b_phys_required = physical_total_sym ^ a_phys;
            for (int64 bsym = 0; bsym < combined_num_irreps; ++bsym)
            {
                if (physical_part(bsym) != b_phys_required ||
                    basis->num_astrs[asym] == 0 || basis->num_bstrs[bsym] == 0)
                    continue;

                BlockDesc<Ti> &block = basis->blocks[block_counter];
                block.asym = asym;
                block.bsym = bsym;
                block.num_a = basis->num_astrs[asym];
                block.num_b = basis->num_bstrs[bsym];
                block.astrs = basis->astrs_vec[asym];
                block.bstrs = basis->bstrs_vec[bsym];
                block.offset = basis->dim;
                basis->block_map[asym * combined_num_irreps + bsym] = block_counter;
                basis->dim += block.num_a * block.num_b;
                block_counter++;
            }
        }

        basis->max_a_count = 0;
        basis->max_b_count = 0;
        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            basis->max_a_count = std::max(basis->max_a_count, (int)basis->blocks[i].num_a);
            basis->max_b_count = std::max(basis->max_b_count, (int)basis->blocks[i].num_b);
        }

        int32 map_size = 1 << norb;
        int32 *a_map = new int32[map_size];
        int32 *b_map = new int32[map_size];
        std::fill(a_map, a_map + map_size, -1);
        std::fill(b_map, b_map + map_size, -1);
        for (int64 i = 0; i < combined_num_irreps; ++i)
        {
            for (int32 a = 0; a < basis->num_astrs[i]; ++a)
                a_map[basis->astrs_vec[i][a]] = a;
            for (int32 b = 0; b < basis->num_bstrs[i]; ++b)
                b_map[basis->bstrs_vec[i][b]] = b;
        }
        basis->a_idx_map = a_map;
        basis->b_idx_map = b_map;
        basis->_init_view();
    }
    catch (...)
    {
        basis->clear();
        delete basis;
        throw;
    }

    return static_cast<void *>(basis);
}

struct GlobalMemMap
{
    int mpi_rank;
    int mpi_size;
    int64 local_dim;
    std::vector<int> block_to_rank;
    std::vector<int64> block_local_offsets;
};

template <typename Ti>
GlobalMemMap *build_global_map(const BasisManager<Ti> *basis, int mpi_rank, int mpi_size)
{
    GlobalMemMap *gmap = new GlobalMemMap();
    gmap->mpi_rank = mpi_rank;
    gmap->mpi_size = mpi_size;

    int64 num_irreps = basis->num_irreps;
    int64 max_h = num_irreps * num_irreps;
    gmap->block_to_rank.assign(max_h, -1);
    gmap->block_local_offsets.assign(max_h, -1);

    std::vector<int64> rank_loads(mpi_size, 0);
    std::vector<int64> current_local_offsets(mpi_size, 0);

    struct BInfo
    {
        int64 h;
        int64 size;
    };
    std::vector<BInfo> binfo;
    for (int64 i = 0; i < basis->num_blocks; ++i)
    {
        int64 h = basis->blocks[i].asym * num_irreps + basis->blocks[i].bsym;
        binfo.push_back({h, basis->blocks[i].num_a * basis->blocks[i].num_b});
    }

    // 贪心算法分配波函数块 — 先按物理对称性扇区分组，再按尺寸负载均衡
    std::sort(binfo.begin(), binfo.end(), [](const BInfo &a, const BInfo &b)
              { return a.size > b.size; });

    const int64 pnirp = basis->physical_num_irreps;
    const int64 pmask = pnirp - 1;
    const int64 nirp  = basis->num_irreps;

    // Group blocks by physical sector
    std::unordered_map<int64, std::vector<BInfo>> phys_blocks;
    std::unordered_map<int64, int64> phys_size;
    int64 total_size = 0;
    for (const auto &bi : binfo)
    {
        const int64 ph = ((bi.h / nirp) & pmask) * pnirp + ((bi.h % nirp) & pmask);
        phys_blocks[ph].push_back(bi);
        phys_size[ph] += bi.size;
        total_size += bi.size;
    }

    std::vector<std::pair<int64, int64>> ranked_phys(phys_size.begin(), phys_size.end());
    std::sort(ranked_phys.begin(), ranked_phys.end(),
              [](auto &a, auto &b) { return a.second > b.second; });

    int ranks_done = 0;
    for (size_t si = 0; si < ranked_phys.size(); ++si)
    {
        const auto &[ph, psize] = ranked_phys[si];
        bool is_last = (si == ranked_phys.size() - 1);
        int K = is_last ? std::max(1, mpi_size - ranks_done)
                        : std::max(1, (int)std::llround((double)mpi_size * (double)psize / (double)total_size));
        if (K > mpi_size - ranks_done)
            K = mpi_size - ranks_done;

        // Find K least-loaded ranks
        std::vector<int> cand_ranks(mpi_size);
        std::iota(cand_ranks.begin(), cand_ranks.end(), 0);
        std::partial_sort(cand_ranks.begin(), cand_ranks.begin() + K, cand_ranks.end(),
                          [&](int a, int b) { return rank_loads[a] < rank_loads[b]; });

        // Within this physical sector, distribute blocks greedily across the K ranks
        auto &blocks = phys_blocks[ph];
        for (const auto &bi : blocks)
        {
            int tgt = cand_ranks[0];
            for (int k = 1; k < K; ++k)
                if (rank_loads[cand_ranks[k]] < rank_loads[tgt])
                    tgt = cand_ranks[k];
            gmap->block_to_rank[bi.h] = tgt;
            gmap->block_local_offsets[bi.h] = current_local_offsets[tgt];
            current_local_offsets[tgt] += bi.size;
            rank_loads[tgt] += bi.size;
        }
        ranks_done += K;
    }

    gmap->local_dim = current_local_offsets[mpi_rank];
    return gmap;
}

struct PackJob
{
    int64 src_offset;
    int64 dst_offset;
    int64 size;
};
