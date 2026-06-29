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
