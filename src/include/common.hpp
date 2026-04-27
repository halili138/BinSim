#pragma once

#include <cstdint>
#include <bit>
#include <algorithm>

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;

#define FORCE_INLINE inline __attribute__((always_inline))

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
    uint32 *all_astrs;
    uint32 *all_bstrs;

    uint32 **astrs_vec;
    uint32 **bstrs_vec;

    int64 *num_astrs;
    int64 *num_bstrs;

    BlockDesc *blocks;
    int64 num_blocks;

    int64 *block_map;
    int64 num_irreps;

    int64 dim;
};
