#pragma once

#include <cstdint>
#include <bit>
#include <complex>
#include <type_traits>
#include <immintrin.h>

#define FORCE_INLINE inline __attribute__((always_inline))
#define FORCE_INLINE_CONSTEXPR constexpr FORCE_INLINE

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using uint128 = __uint128_t;

struct uint256
{
    uint128 lo;
    uint128 hi;

    constexpr uint256 operator<<(int p) const noexcept
    {
        if (p <= 0)
            return *this;
        if (p < 128)
            return {lo << p, (hi << p) | (lo >> (128 - p))};
        if (p < 256)
            return {0, lo << (p - 128)};
        return {0, 0};
    }

    constexpr uint256 operator|(const uint256 &o) const noexcept { return {lo | o.lo, hi | o.hi}; }

    constexpr uint256 operator~() const noexcept { return {~lo, ~hi}; }

    constexpr uint256 operator^(const uint256 &o) const noexcept { return {lo ^ o.lo, hi ^ o.hi}; }

    constexpr uint256 operator&(const uint256 &o) const noexcept { return {lo & o.lo, hi & o.hi}; }

    constexpr bool operator==(const uint256 &o) const noexcept { return lo == o.lo && hi == o.hi; }

    constexpr bool operator!=(const uint256 &o) const noexcept { return lo != o.lo || hi != o.hi; }

    constexpr bool operator<(const uint256 &o) const noexcept
    {
        if (hi != o.hi)
            return hi < o.hi;
        return lo < o.lo;
    }

    constexpr bool operator>(const uint256 &o) const noexcept { return o < *this; }

    constexpr bool operator<=(const uint256 &o) const noexcept { return !(*this > o); }

    constexpr bool operator>=(const uint256 &o) const noexcept { return !(*this < o); }

    constexpr uint256 operator-(const uint256 &o) const noexcept
    {
        if (lo < o.lo)
            return {lo - o.lo, hi - o.hi - static_cast<uint128>(1)};
        return {lo - o.lo, hi - o.hi};
    }
};

FORCE_INLINE int popcnt(uint8 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint16 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint32 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint64 v) { return std::popcount(v); }
FORCE_INLINE int popcnt(uint128 v)
{
    return std::popcount(static_cast<uint64>(v >> 64)) +
           std::popcount(static_cast<uint64>(v));
}
FORCE_INLINE int popcnt(const uint256 &v)
{
    return popcnt(v.lo) + popcnt(v.hi);
}

FORCE_INLINE uint32 zip_even_bit_bmi2(uint64 v)
{
    return static_cast<uint32>(_pext_u64(v, 0x5555555555555555ULL));
}
FORCE_INLINE uint64 zip_even_bit_bmi2(uint128 v)
{
    uint64 lo = static_cast<uint64>(v);
    uint64 hi = static_cast<uint64>(v >> 64);

    uint64 res_lo = _pext_u64(lo, 0x5555555555555555ULL);
    uint64 res_hi = _pext_u64(hi, 0x5555555555555555ULL);

    return (res_hi << 32) | res_lo;
}
FORCE_INLINE uint128 zip_even_bit_bmi2(const uint256 &v)
{
    // 复用前面的 uint128 重载，返回的是 uint64
    uint64 res_lo = zip_even_bit_bmi2(v.lo);
    uint64 res_hi = zip_even_bit_bmi2(v.hi);

    // 将高 64 位结果左移 64 位并拼接
    return (static_cast<uint128>(res_hi) << 64) | res_lo;
}

FORCE_INLINE uint32 zip_odd_bit_bmi2(uint64 v)
{
    return static_cast<uint32>(_pext_u64(v, 0xAAAAAAAAAAAAAAAAULL));
}
FORCE_INLINE uint64 zip_odd_bit_bmi2(uint128 v)
{
    uint64 lo = static_cast<uint64>(v);
    uint64 hi = static_cast<uint64>(v >> 64);

    uint64 res_lo = _pext_u64(lo, 0xAAAAAAAAAAAAAAAAULL);
    uint64 res_hi = _pext_u64(hi, 0xAAAAAAAAAAAAAAAAULL);

    return (res_hi << 32) | res_lo;
}
FORCE_INLINE uint128 zip_odd_bit_bmi2(const uint256 &v)
{
    uint64 res_lo = zip_odd_bit_bmi2(v.lo);
    uint64 res_hi = zip_odd_bit_bmi2(v.hi);

    return (static_cast<uint128>(res_hi) << 64) | res_lo;
}

FORCE_INLINE uint64 uzip_even_bit_bmi2(uint32 v)
{
    return _pdep_u64(static_cast<uint64>(v), 0x5555555555555555ULL);
}
FORCE_INLINE uint128 uzip_even_bit_bmi2(uint64 v)
{
    uint64 lo = v & 0xFFFFFFFFULL; // 取低 32 位
    uint64 hi = v >> 32;           // 取高 32 位

    uint64 res_lo = _pdep_u64(lo, 0x5555555555555555ULL);
    uint64 res_hi = _pdep_u64(hi, 0x5555555555555555ULL);

    return (static_cast<uint128>(res_hi) << 64) | res_lo;
}
FORCE_INLINE uint256 uzip_even_bit_bmi2(uint128 v)
{
    uint128 res_lo = uzip_even_bit_bmi2(static_cast<uint64>(v));
    uint128 res_hi = uzip_even_bit_bmi2(static_cast<uint64>(v >> 64));

    return {res_lo, res_hi}; // 依赖于 uint256 的构造 {lo, hi}
}

FORCE_INLINE uint64 uzip_odd_bit_bmi2(uint32 v)
{
    return _pdep_u64(static_cast<uint64>(v), 0xAAAAAAAAAAAAAAAAULL);
}
FORCE_INLINE uint128 uzip_odd_bit_bmi2(uint64 v)
{
    uint64 lo = v & 0xFFFFFFFFULL;
    uint64 hi = v >> 32;

    uint64 res_lo = _pdep_u64(lo, 0xAAAAAAAAAAAAAAAAULL);
    uint64 res_hi = _pdep_u64(hi, 0xAAAAAAAAAAAAAAAAULL);

    return (static_cast<uint128>(res_hi) << 64) | res_lo;
}
FORCE_INLINE uint256 uzip_odd_bit_bmi2(uint128 v)
{
    uint128 res_lo = uzip_odd_bit_bmi2(static_cast<uint64>(v));
    uint128 res_hi = uzip_odd_bit_bmi2(static_cast<uint64>(v >> 64));

    return {res_lo, res_hi};
}

FORCE_INLINE_CONSTEXPR int ctz_gen(uint32 v) noexcept { return std::countr_zero(v); }
FORCE_INLINE_CONSTEXPR int ctz_gen(uint64 v) noexcept { return std::countr_zero(v); }
FORCE_INLINE_CONSTEXPR int ctz_gen(uint128 v) noexcept
{
    uint64 lo = static_cast<uint64>(v);
    uint64 hi = static_cast<uint64>(v >> 64);
    return lo ? std::countr_zero(lo) : 64 + std::countr_zero(hi);
}

template <typename Ti>
FORCE_INLINE_CONSTEXPR Ti get_zero() noexcept;
template <>
FORCE_INLINE_CONSTEXPR uint8 get_zero<uint8>() noexcept { return 0; }
template <>
FORCE_INLINE_CONSTEXPR uint16 get_zero<uint16>() noexcept { return 0; }
template <>
FORCE_INLINE_CONSTEXPR uint32 get_zero<uint32>() noexcept { return 0; }
template <>
FORCE_INLINE_CONSTEXPR uint64 get_zero<uint64>() noexcept { return 0; }

template <typename Ti>
FORCE_INLINE_CONSTEXPR Ti get_one() noexcept;
template <>
FORCE_INLINE_CONSTEXPR uint8 get_one<uint8>() noexcept { return 1; }
template <>
FORCE_INLINE_CONSTEXPR uint16 get_one<uint16>() noexcept { return 1; }
template <>
FORCE_INLINE_CONSTEXPR uint32 get_one<uint32>() noexcept { return 1; }
template <>
FORCE_INLINE_CONSTEXPR uint64 get_one<uint64>() noexcept { return 1; }

template <>
FORCE_INLINE_CONSTEXPR uint128 get_zero<uint128>() noexcept { return 0; }
template <>
FORCE_INLINE_CONSTEXPR uint256 get_zero<uint256>() noexcept { return {0, 0}; }
template <>
FORCE_INLINE_CONSTEXPR uint128 get_one<uint128>() noexcept { return 1; }
template <>
FORCE_INLINE_CONSTEXPR uint256 get_one<uint256>() noexcept { return {1, 0}; }

template <typename T>
struct half_width;
template <>
struct half_width<uint16>
{
    using type = uint8;
};
template <>
struct half_width<uint32>
{
    using type = uint16;
};
template <>
struct half_width<uint64>
{
    using type = uint32;
};
template <>
struct half_width<uint128>
{
    using type = uint64;
};
template <>
struct half_width<uint256>
{
    using type = uint128;
};
template <typename T>
using half_width_t = typename half_width<T>::type;

template <typename T>
struct double_width;
template <>
struct double_width<uint8>
{
    using type = uint16;
};
template <>
struct double_width<uint16>
{
    using type = uint32;
};
template <>
struct double_width<uint32>
{
    using type = uint64;
};
template <>
struct double_width<uint64>
{
    using type = uint128;
};
template <>
struct double_width<uint128>
{
    using type = uint256;
};
template <typename T>
using double_width_t = typename double_width<T>::type;

FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(uint8 v) noexcept { return v; }
FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(uint16 v) noexcept { return v; }
FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(uint32 v) noexcept { return v; }
FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(uint64 v) noexcept { return v; }
FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(uint128 v) noexcept
{
    return static_cast<uint64>(v >> 64) ^ static_cast<uint64>(v);
}
FORCE_INLINE_CONSTEXPR uint64 fold_for_hash(const uint256 &v) noexcept
{
    return fold_for_hash(v.hi) ^ fold_for_hash(v.lo);
}

FORCE_INLINE_CONSTEXPR uint32 mix_hash(uint64 h) noexcept
{
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return static_cast<uint32>(h);
}

static_assert((uint256{1, 0} << 0) == uint256{1, 0});
static_assert((uint256{1, 0} << 1) == uint256{2, 0});
static_assert((uint256{1, 0} << 127) == uint256{static_cast<uint128>(1) << 127, 0});
static_assert((uint256{1, 0} << 128) == uint256{0, 1});
static_assert((uint256{1, 0} << 255) == uint256{0, static_cast<uint128>(1) << 127});
static_assert((uint256{1, 0} << 256) == uint256{0, 0});
static_assert(get_zero<uint256>() == uint256{0, 0});
static_assert(get_one<uint256>() == uint256{1, 0});
static_assert(fold_for_hash((static_cast<uint128>(0x0123456789abcdefULL) << 64) |
                            static_cast<uint128>(0xfedcba9876543210ULL)) == 0xffffffffffffffffULL);
static_assert(mix_hash(0x123456789abcdef0ULL) == 0x8ec5b906U);
static_assert(ctz_gen(uint32{0}) == 32);
static_assert(ctz_gen(uint64{0}) == 64);
static_assert(ctz_gen(uint128{0}) == 128);
