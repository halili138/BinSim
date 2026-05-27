#pragma once

#include <cstdint>
#include <bit>
#include <algorithm>
#include <complex>
#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <map>

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using complexf64 = std::complex<double>;

#define FORCE_INLINE inline __attribute__((always_inline))

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

template <typename T>
FORCE_INLINE T math_conj(const T &x)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        return x;
    }
    else
    {
        return std::conj(x);
    }
}

template <typename T>
FORCE_INLINE int phase(T x)
{
    return 1 - 2 * (std::popcount(x) & 1);
}

template <typename T>
FORCE_INLINE int64 find_index(const T *arr, int64 len, T val)
{
    const T *it = std::lower_bound(arr, arr + len, val);
    if (it != arr + len && *it == val)
    {
        return std::distance(arr, it);
    }
    return -1;
}

template <typename T>
FORCE_INLINE int64 get_string_sym(T str, const int64 *orbsym)
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

template <typename T>
FORCE_INLINE T next_combination(T v)
{
    if (v == 0)
        return 0;
    T c = (v & -v);
    T r = v + c;
    return (((r ^ v) >> 2) / c) | r;
}

template <typename T>
struct BlockDesc
{
    int64 asym, bsym, num_a, num_b;
    const T *astrs, *bstrs;
    int64 offset;
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

    int64 dim = {};
    int64 norb = {};
    int64 max_a_count = {};
    int64 max_b_count = {};

    int *a_idx_map = nullptr;
    int *b_idx_map = nullptr;

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
        basis->norb = norb;
        basis->dim = 0;
        basis->num_blocks = 0;

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
    }
    catch (...)
    {
        basis->clear();
        delete basis;
        throw;
    }

    return static_cast<void *>(basis);
}
