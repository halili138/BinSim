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
#include "bitintegers.hpp"

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
    static_assert(is_supported_bit_uint_v<T>, "phase<T> requires a supported unsigned bit-integer type");
    return 1 - 2 * (popcnt(x) & 1);
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
    static_assert(is_supported_bit_uint_v<T>, "get_string_sym<T> requires a supported unsigned bit-integer type");
    int64 sym = 0;
    while (str != get_zero<T>())
    {
        const int64 pos = ctz_gen(str);
        sym ^= *(orbsym + pos);
        str = str & (str - get_one<T>());
    }
    return sym;
}

template <typename T>
FORCE_INLINE T next_combination(T v)
{
    static_assert(is_supported_bit_uint_v<T>, "next_combination<T> requires a supported unsigned bit-integer type");
    if (v == get_zero<T>())
        return get_zero<T>();
    T c = (v & (get_zero<T>() - v));
    T r = v + c;
    return (((r ^ v) >> (ctz_gen(c) + 2)) | r);
}
