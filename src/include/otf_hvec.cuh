#pragma once
#include <cuda_runtime.h>
#include "common.hpp"

// CUDA 兼容的复数共轭
template <typename T>
__device__ __forceinline__ T math_conj_dev(const T &x)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        return x;
    }
    else
    {
        return T(x.real(), -x.imag());
    }
}

template <typename T>
__device__ __forceinline__ int64 get_string_sym_dev(T str, const int64 *orbsym)
{
    int64 sym = 0;
    int64 pos = 0;
    while (str > 0)
    {
        if (str & 1)
            sym ^= orbsym[pos];
        str >>= 1;
        ++pos;
    }
    return sym;
}

// 展平的 Basis 视图 (存放在 GPU 端)
template <typename Ti>
struct BasisViewDev
{
    int num_blocks;
    const int64 *block_offsets;
    const int *block_num_a;
    const int *block_num_b;
    const int *block_asym;
    const int *block_bsym;

    const Ti *astrs_flat;
    const int64 *astrs_start_idx;

    const Ti *bstrs_flat;
    const int64 *bstrs_start_idx;

    const int *block_map;
    const int64 *orbsym;
    int num_irreps;
};

template <int Rank, typename Ti, typename Tv>
__global__ void mixed_prep_b_kernel(
    BasisViewDev<Ti> basis,
    Ti bx,
    const Tv *w0, const Ti *zbs, int num_zb,
    const int *b_idx_map,
    int *valid_b_counts,
    int *src_b_arr, int *dst_b_arr, Tv *pb_arr,
    int max_b_count, int rank)
{
    int dst_block_idx = blockIdx.x;
    int thread_id = threadIdx.x;

    // 共享内存计数器，用于当前 Block 分配连续的 B 索引
    __shared__ int shared_valid_count;
    if (thread_id == 0)
        shared_valid_count = 0;

    __syncthreads();

    // 混合算符的对称性匹配 (仅检查 B 侧可能还不够，但完整合法性在 A 侧进一步卡死)
    const int dst_b_count = basis.block_num_b[dst_block_idx];
    const int64 b_start = basis.bstrs_start_idx[dst_block_idx];

    // Block 内的线程共同遍历当前 dst_block 的所有 B 弦
    for (int i = thread_id; i < dst_b_count; i += blockDim.x)
    {
        const Ti dst_str_b = basis.bstrs_flat[b_start + i];
        const Ti src_str_b = dst_str_b ^ bx;
        const int src_b_idx = b_idx_map[src_str_b];

        if (src_b_idx == -1)
            continue;

        // Mixed 算符 B 侧无需去重 (去重在 A 侧执行)
        // 原子获取写入位置 (仅在 Shared Memory 极快)
        const int pos = atomicAdd(&shared_valid_count, 1);

        int base_offset = dst_block_idx * max_b_count;
        src_b_arr[base_offset + pos] = src_b_idx;
        dst_b_arr[base_offset + pos] = i;
        Tv *pb0 = pb_arr + base_offset;

        // 计算相位 (利用 CUDA 硬件级别的 popcount 指令)
        if constexpr (Rank == 1)
        {
            Tv pt0 = {};
            for (int k = 0; k < num_zb; ++k)
            {
                const bool parity = __popcll(src_str_b & zbs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
            }
            pb0[pos] = pt0;
        }
        else if constexpr (Rank == 2)
        {
            const Tv *w1 = w0 + num_zb;
            Tv pt0 = {}, pt1 = {};
            for (int k = 0; k < num_zb; ++k)
            {
                const bool parity = __popcll(src_str_b & zbs[k]) & 1;
                pt0 += parity ? -w0[k] : w0[k];
                pt1 += parity ? -w1[k] : w1[k];
            }
            pb0[pos] = pt0;
            pb0[pos + max_b_count] = pt1; // 纯 SOA 布局
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                const Tv *wr = w0 + r * num_zb;
                Tv ptn = {};
                for (int k = 0; k < num_zb; ++k)
                {
                    const bool parity = __popcll(src_str_b & zbs[k]) & 1;
                    ptn += parity ? -wr[k] : wr[k];
                }
                pb0[pos + r * max_b_count] = ptn;
            }
        }
    }

    __syncthreads();

    // 将最终合法的 B 数量写回全局内存
    if (thread_id == 0)
        valid_b_counts[dst_block_idx] = shared_valid_count;
}

template <int Rank, typename Ti, typename Tv>
__global__ void mixed_contract_kernel(
    BasisViewDev<Ti> basis,
    Ti ax, Ti bx,
    const Tv *w0_a, const Ti *zas, int num_za,
    const int *a_idx_map,
    const int *valid_b_counts,
    const int *src_b_arr, const int *dst_b_arr, const Tv *pb_arr,
    Tv *vec,
    double cd, double co,
    int max_b_count, int rank)
{
    int dst_block_idx = blockIdx.x;
    int valid_b = valid_b_counts[dst_block_idx];

    // 如果当前 Block 没有合法的 B 跳转，直接退出
    if (valid_b == 0)
        return;

    int axsym = get_string_sym_dev(ax, basis.orbsym);
    int bxsym = get_string_sym_dev(bx, basis.orbsym);
    int bid = (basis.block_asym[dst_block_idx] ^ axsym) * basis.num_irreps + (basis.block_bsym[dst_block_idx] ^ bxsym);
    int src_block_idx = basis.block_map[bid];

    if (src_block_idx == -1 || src_block_idx < dst_block_idx)
        return;
    bool is_same_block = (src_block_idx == dst_block_idx);

    int dst_a_count = basis.block_num_a[dst_block_idx];
    int64 a_start = basis.astrs_start_idx[dst_block_idx];

    // 并行映射到 A 侧的行
    for (int i = threadIdx.x; i < dst_a_count; i += blockDim.x)
    {
        Ti dst_str_a = basis.astrs_flat[a_start + i];
        Ti src_str_a = dst_str_a ^ ax;
        int src_a_idx = a_idx_map[src_str_a];

        if (src_a_idx == -1)
            continue;

        // 核心：同区块去重，只计算上三角方向，防止互为 src/dst 被旋转两次
        if (is_same_block && src_a_idx < i)
            continue;

        // 计算 A 侧相位
        Tv pa0 = {}, pa1 = {};
        Tv pan[64] = {}; // 动态 rank 存寄存器数组

        if constexpr (Rank == 1)
        {
            for (int k = 0; k < num_za; ++k)
                pa0 += (__popcll(src_str_a & zas[k]) & 1) ? -w0_a[k] : w0_a[k];
        }
        else if constexpr (Rank == 2)
        {
            const Tv *w1_a = w0_a + num_za;
            for (int k = 0; k < num_za; ++k)
            {
                bool parity = __popcll(src_str_a & zas[k]) & 1;
                pa0 += parity ? -w0_a[k] : w0_a[k];
                pa1 += parity ? -w1_a[k] : w1_a[k];
            }
        }
        else
        {
            for (int r = 0; r < rank; ++r)
            {
                Tv ptn = {};
                const Tv *wr = w0_a + r * num_za;
                for (int k = 0; k < num_za; ++k)
                    ptn += (__popcll(src_str_a & zas[k]) & 1) ? -wr[k] : wr[k];
                pan[r] = ptn;
            }
        }

        // 行偏移基址提取
        int64 src_row_offset = basis.block_offsets[src_block_idx] + src_a_idx * basis.block_num_b[src_block_idx];
        int64 dst_row_offset = basis.block_offsets[dst_block_idx] + i * basis.block_num_b[dst_block_idx];
        int base_b_offset = dst_block_idx * max_b_count;

        // 【极速内层】：利用预先准备的 B 数据，A 线程顺序执行列的收缩
        // 这里的 pb_arr 是 Broadcast Uniform 读取，所有 Warp 线程命中恒定的 L1 Cache，速度极快！
        for (int vb = 0; vb < valid_b; ++vb)
        {
            Tv vt = {};
            if constexpr (Rank == 1)
            {
                vt = pa0 * pb_arr[base_b_offset + vb];
            }
            else if constexpr (Rank == 2)
            {
                vt = pa0 * pb_arr[base_b_offset + vb] +
                     pa1 * pb_arr[base_b_offset + max_b_count + vb];
            }
            else
            {
                for (int r = 0; r < rank; ++r)
                {
                    vt += pan[r] * pb_arr[base_b_offset + r * max_b_count + vb];
                }
            }

            Tv vd = 1.0 + cd * (vt * math_conj_dev(vt));
            Tv vo_fwd = co * vt;
            Tv vo_rev = co * math_conj_dev(vt);

            int64 si = src_row_offset + src_b_arr[base_b_offset + vb];
            int64 di = dst_row_offset + dst_b_arr[base_b_offset + vb];

            // 安全原地操作：因为外层由 Host 下发单个 Group 保证隔离
            Tv vi = vec[si];
            Tv vj = vec[di];
            vec[si] = vi * vd - vj * vo_rev;
            vec[di] = vj * vd + vi * vo_fwd;
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void mixed_hvec_gather_contract_kernel(
    BasisViewDev<Ti> basis,
    const SVDGroupViewDev<Ti, Tv> groups,
    int batch_start,
    int cur_batch_size,
    const int *a_idx_map,
    const int *valid_b_counts, // [num_blocks * BATCH_SIZE]
    const int *src_b_arr,      // [num_blocks * BATCH_SIZE * max_b_count]
    const int *dst_b_arr,
    const Tv *pb_arr,
    const Tv *src_vec,
    Tv *dst_vec,
    int max_b_count)
{
    // Grid.x = dst_block_idx, Grid.y 和 Thread.x 共同映射 dst_a_idx
    int dst_block_idx = blockIdx.x;
    int dst_a_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (dst_a_idx >= basis.block_num_a[dst_block_idx])
        return;

    // 提前计算 Dst 的行指针 (该线程专属的写入区域！)
    Tv *dst_row_ptr = dst_vec + basis.block_offsets[dst_block_idx] + dst_a_idx * basis.block_num_b[dst_block_idx];
    Ti dst_str_a = basis.astrs_flat[basis.astrs_start_idx[dst_block_idx] + dst_a_idx];

    // 在线程内部串行处理整个 Batch
    for (int batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx)
    {
        // 从全局打平的数组中获取当前 Group 针对当前 Block 的 valid_b
        int cache_idx = dst_block_idx * cur_batch_size + batch_idx;
        int valid_b = valid_b_counts[cache_idx];
        if (valid_b == 0)
            continue;

        int group_idx = batch_start + batch_idx;
        Ti ax = groups.ax_array[group_idx];
        Ti src_str_a = dst_str_a ^ ax;
        int src_a_idx = a_idx_map[src_str_a];

        if (src_a_idx == -1)
            continue;

        // 查找 src_block (对称性映射)
        int axsym = get_string_sym_dev(ax, basis.orbsym);
        int bxsym = get_string_sym_dev(groups.bx_array[group_idx], basis.orbsym);
        int bid = (basis.block_asym[dst_block_idx] ^ axsym) * basis.num_irreps + (basis.block_bsym[dst_block_idx] ^ bxsym);
        int src_block_idx = basis.block_map[bid];

        const Tv *src_row_ptr = src_vec + basis.block_offsets[src_block_idx] + src_a_idx * basis.block_num_b[src_block_idx];

        // 提取 A 侧参数
        int num_za = groups.num_za_array[group_idx];
        const Ti *zas = groups.flat_zas + groups.za_start_idx[group_idx];
        const Tv *wa0 = groups.flat_wa + groups.wa_start_idx[group_idx];

        // 计算 A 相位
        Tv pa0 = {}, pa1 = {};
        Tv pan[64] = {};
        if constexpr (Rank == 1)
        {
            for (int k = 0; k < num_za; ++k)
                pa0 += (__popcll(src_str_a & zas[k]) & 1) ? -wa0[k] : wa0[k];
        }
        else if constexpr (Rank == 2)
        {
            const Tv *wa1 = wa0 + num_za;
            for (int k = 0; k < num_za; ++k)
            {
                bool parity = __popcll(src_str_a & zas[k]) & 1;
                pa0 += parity ? -wa0[k] : wa0[k];
                pa1 += parity ? -wa1[k] : wa1[k];
            }
        }
        else
        {
            for (int r = 0; r < groups.rank_array[group_idx]; ++r)
            {
                Tv ptn = {};
                const Tv *wr = wa0 + r * num_za;
                for (int k = 0; k < num_za; ++k)
                    ptn += (__popcll(src_str_a & zas[k]) & 1) ? -wr[k] : wr[k];
                pan[r] = ptn;
            }
        }

        int b_base = cache_idx * max_b_count;
        int rank = groups.rank_array[group_idx];

        // 【极速最内层】：纯 Gather 累加，绝对无锁！
        for (int vb = 0; vb < valid_b; ++vb)
        {
            Tv vt = {};
            if constexpr (Rank == 1)
            {
                vt = pa0 * pb_arr[b_base + vb];
            }
            else if constexpr (Rank == 2)
            {
                vt = pa0 * pb_arr[b_base + vb] + pa1 * pb_arr[b_base + max_b_count + vb];
            }
            else
            {
                for (int r = 0; r < rank; ++r)
                {
                    vt += pan[r] * pb_arr[b_base + r * max_b_count + vb];
                }
            }

            int src_b = src_b_arr[b_base + vb];
            int dst_b = dst_b_arr[b_base + vb];

            // 安全的常规累加：线程独占 dst_row_ptr
            dst_row_ptr[dst_b] += src_row_ptr[src_b] * vt;
        }
    }
}

