#pragma once

#include <cstdint>
#include <bit>
#include <algorithm>
#include <complex>
#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>

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

template <typename Tv>
FORCE_INLINE Tv fast_diag_exp(const Tv &vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        // 实数域（分子体系）：对角线严格为 0，exp(0) == 1.0
        // 为了极致性能，编译器遇到 type=double 会直接把整个内层循环优化为 1.0！
        return static_cast<Tv>(1.0);
    }
    else
    {
        // 复数域（周期性体系）：对角线必定是纯虚数，直接提取虚部
        double val = vt.imag() * theta;

        // 使用实数的 cos 和 sin，彻底避开昂贵的 __cexp 库函数调用
        return Tv(std::cos(val), std::sin(val));
    }
}

template <typename Tv>
FORCE_INLINE Tv fast_diag_grad(const Tv &vt, double theta)
{
    if constexpr (std::is_arithmetic_v<Tv>)
    {
        // 实数域下导数也必定为 0
        return static_cast<Tv>(0.0);
    }
    else
    {
        // 导数公式: dU = v_t * exp(v_t * theta)
        double val = vt.imag() * theta;
        Tv u(std::cos(val), std::sin(val));
        return vt * u;
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
    T *all_astrs;
    T *all_bstrs;

    T **astrs_vec;
    T **bstrs_vec;

    int64 *num_astrs;
    int64 *num_bstrs;

    BlockDesc<T> *blocks;
    int64 num_blocks;

    int64 *orbsym;
    int64 *block_map;
    int64 num_irreps;

    int64 dim;
};

template <typename Ti>
int64 get_subspace_dim_tmpl(const BasisManager<Ti> *basis)
{
    return basis->dim;
}

template <typename Ti>
void destroy_basis_manager_tmpl(BasisManager<Ti> *basis)
{
    delete[] basis->all_astrs;
    delete[] basis->all_bstrs;
    delete[] basis->astrs_vec;
    delete[] basis->bstrs_vec;
    delete[] basis->num_astrs;
    delete[] basis->num_bstrs;
    delete[] basis->blocks;
    delete[] basis->orbsym;
    delete[] basis->block_map;
    delete basis;
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

    basis->all_astrs = nullptr;
    basis->all_bstrs = nullptr;
    basis->astrs_vec = nullptr;
    basis->bstrs_vec = nullptr;
    basis->num_astrs = nullptr;
    basis->num_bstrs = nullptr;
    basis->blocks = nullptr;
    basis->orbsym = nullptr;
    basis->block_map = nullptr;

    try
    {
        basis->num_irreps = num_irreps;
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
    }
    catch (...)
    {
        destroy_basis_manager_tmpl<Ti>(basis);
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

template <typename Ti, typename Tv>
void compute_diagonal_elements_raw(
    const BasisManager<Ti> *__restrict__ basis,
    const Ti *__restrict__ azs,
    const Ti *__restrict__ bzs,
    const Tv *__restrict__ cs,
    const int64 nterms,
    Tv *__restrict__ diags)
{
    if (nterms == 0)
        return;

#pragma omp parallel
    {
        bool *parity_a = new bool[nterms];

        for (int64 i = 0; i < basis->num_blocks; ++i)
        {
            const BlockDesc<Ti> &block = basis->blocks[i];
            const int64 num_a = block.num_a;
            const int64 num_b = block.num_b;
#pragma omp for schedule(guided) nowait
            for (int64 a = 0; a < num_a; ++a)
            {
                const Ti astr = block.astrs[a];
                const int64 row_ptr = block.offset + a * num_b;

                for (int64 k = 0; k < nterms; ++k)
                {
                    parity_a[k] = std::popcount(azs[k] & astr) & 1;
                }

                for (int64 b = 0; b < num_b; ++b)
                {
                    const Ti bstr = block.bstrs[b];

                    Tv vt = {};
                    for (int64 k = 0; k < nterms; ++k)
                    {
                        const bool parity_b = std::popcount(bzs[k] & bstr) & 1;
                        const bool parity = parity_a[k] ^ parity_b;

                        if constexpr (std::is_arithmetic_v<Tv>)
                        {
                            vt += parity ? -cs[k] : cs[k];
                        }
                        else
                        {
                            Tv val = parity ? -cs[k] : cs[k];
                            vt += Tv(val.real(), 0.0);
                        }
                    }

                    diags[row_ptr + b] += vt;
                }
            }
        }

        delete[] parity_a;
    }
}
