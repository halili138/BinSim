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
#include <stdexcept>

#define FORCE_INLINE inline __attribute__((always_inline))

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using complexf64 = std::complex<double>;

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0.0, 0.0))

FORCE_INLINE int popcnt(uint8 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint16 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint32 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint64 v) { return std::popcount(v); }

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

template <typename Ti> FORCE_INLINE Ti get_zero();
template <> FORCE_INLINE uint8 get_zero<uint8>() { return 0; }
template <> FORCE_INLINE uint16 get_zero<uint16>() { return 0; }
template <> FORCE_INLINE uint32 get_zero<uint32>() { return 0; }
template <> FORCE_INLINE uint64 get_zero<uint64>() { return 0; }

template <typename Ti> FORCE_INLINE Ti get_one();
template <> FORCE_INLINE uint8 get_one<uint8>() { return 1; }
template <> FORCE_INLINE uint16 get_one<uint16>() { return 1; }
template <> FORCE_INLINE uint32 get_one<uint32>() { return 1; }
template <> FORCE_INLINE uint64 get_one<uint64>() { return 1; }

FORCE_INLINE uint64 fold_for_hash(uint8 v) { return v; }
FORCE_INLINE uint64 fold_for_hash(uint16 v) { return v; }
FORCE_INLINE uint64 fold_for_hash(uint32 v) { return v; }
FORCE_INLINE uint64 fold_for_hash(uint64 v) { return v; }
