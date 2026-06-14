#pragma once
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_diag_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;
    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    const Tv *src_vec_bid = src_vec + basis.block_offsets[bid];
    Tv *dst_vec_bid = dst_vec + basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;

        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;

        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);

        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = (a < a_tile_end);
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;
        const Tv *src_base = valid_a ? (src_vec_bid + (int64)a * n_b) : nullptr;

        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);
            const int total_sh_elements = current_chunk_groups * current_tile_b;

            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;
                const int g = chunk_start_g + g_offset;
                const Ti bstr = bstrs_tile_start[b_offset];
                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];
                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(bstr, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    const int g = chunk_start_g + g_offset;
                    const int nza = groups.num_zas[g];
                    const Ti *zas = groups.flat_zas + groups.za_start[g];
                    const Tv *wa = groups.flat_wa + groups.wa_start[g];
                    const int rank = groups.ranks[g];

                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(astr, zas, nza, wa, pa, 1, rank);
                    const Tv *pb = sh_pb + (g_offset * TILE_B);

                    if (current_tile_b == TILE_B)
                    {
#pragma unroll
                        for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                        {
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            accum[b_offset] += __ldg(&src_base[b_tile_start + b_offset]) * vt;
                        }
                    }
                    else
                    {
                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                            accum[b_offset] += __ldg(&src_base[b_tile_start + b_offset]) * vt;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (valid_a)
        {
            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            {
                const int actual_b_idx = b_tile_start + b_offset;
                dst_base[actual_b_idx] += accum[b_offset];
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_mixed_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;
    const int *a_idx_map = basis.astr2idx;
    const int *b_idx_map = basis.bstr2idx;

    constexpr int SHARED_MEM_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1 * TILE_B
        : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;

    constexpr int BATCH_SIZE =
        Rank == 1   ? BATCH_SIZE_SH1
        : Rank == 2 ? BATCH_SIZE_SH2
                    : BATCH_SIZE_SH3;

    constexpr int IDX_MEM_SIZE = BATCH_SIZE * TILE_B;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sa_b[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid_b[BATCH_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;

    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;

    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    Tv *dst_vec_bid = dst_vec + basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;

        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;

        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);

        // 【关键1：提取当前线程对应的 a，进行越界保护】
        // 由于 TILE_A 是 256，blockDim.x 也是 256，这里 a 恰好由 1 个线程固定处理
        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = (a < a_tile_end);
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;

        // 【关键2：极其重要的寄存器缓存！用于累加跨越所有 group 的结果】
        // TILE_B 为 32，刚好占用 32 个寄存器，完全足够，杜绝显存写放大！
        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            // -------------- [这里必须保持所有线程协作加载 Shared Memory] --------------
            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = (asym ^ groups.asyms[g]) * nirp + (bsym ^ groups.bsyms[g]);
                const int src_bid = basis.block_map[h];
                sh_src_bid[g_offset] = src_bid;
                sh_valid_b[g_offset] = (src_bid == -1) ? 0 : n_b;
            }
            __syncthreads();

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;

                if (sh_valid_b[g_offset] == 0)
                    continue;

                const int g = chunk_start_g + g_offset;
                const Ti dst_b_str = bstrs_tile_start[b_offset];
                const Ti bx = groups.bxs[g];
                const Ti sas_b = dst_b_str ^ bx;

                const int sh_flat_offset = g_offset * TILE_B + b_offset;
                sh_sa_b[sh_flat_offset] = b_idx_map[sas_b];

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(sas_b, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }
            __syncthreads();
            // ----------------------------------------------------------------------

            // 【开始正式计算】只有负责有效 a 的线程才会进入计算
            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const Ti ax = groups.axs[g];
                    int sa = -1;
                    Ti sas_a = 0;

                    if (ax == 0)
                    {
                        sa = a;
                        sas_a = astr;
                    }
                    else
                    {
                        sas_a = astr ^ ax;
                        sa = a_idx_map[sas_a];
                    }

                    if (sa != -1)
                    {
                        const int nza = groups.num_zas[g];
                        const Ti *zas = groups.flat_zas + groups.za_start[g];
                        const Tv *wa = groups.flat_wa + groups.wa_start[g];
                        const int rank = groups.ranks[g];

                        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                        Tv pa[STACK_SIZE] = {};
                        compute_phase_dev<Rank, Ti, Tv>(sas_a, zas, nza, wa, pa, 1, rank);

                        const int sbi = sh_src_bid[g_offset];
                        const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * basis.block_num_b[sbi];
                        const Tv *pb = sh_pb + (g_offset * TILE_B);
                        const int *sh_sa_b_task = sh_sa_b + g_offset * TILE_B;

                        if (current_tile_b == TILE_B)
                        {
// 对于 99% 的完整 Tile，强制编译器将 32 次循环完全展开成一长串 FMA (融合乘加) 指令
#pragma unroll
                            for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                            {
                                const int sa_b = sh_sa_b_task[b_offset];
                                if (sa_b != -1)
                                {
                                    const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                    accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                                }
                            }
                        }
                        else
                        {
                            // 仅在处理最后边界残缺的 Tile 时，使用常规循环
                            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                            {
                                const int sa_b = sh_sa_b_task[b_offset];
                                if (sa_b != -1)
                                {
                                    const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                    accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                                }
                            }
                        }
                    }
                }
            }
            // 必须等待同一 Block 内所有人都处理完这个 Chunk
            __syncthreads();
        }

        // 【关键3：整个大循环全部结束，向显存只发起 1 次写入！】
        if (valid_a)
        {
            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            {
                const int actual_b_idx = b_tile_start + b_offset;
                dst_base[actual_b_idx] += accum[b_offset];
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_pure_a_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;
    const int *a_idx_map = basis.astr2idx;

    constexpr int SHARED_MEM_SIZE = Rank == 1 ? BATCH_SIZE_SH1 * TILE_B : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                                                                                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE_SH1 : Rank == 2 ? BATCH_SIZE_SH2
                                                                      : BATCH_SIZE_SH3;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid_b[BATCH_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;

    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;

    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    Tv *dst_vec_bid = dst_vec + basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;

        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;

        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);

        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = (a < a_tile_end);
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;

        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = (asym ^ groups.asyms[g]) * nirp + bsym;
                const int src_bid = basis.block_map[h];

                sh_src_bid[g_offset] = src_bid;
                sh_valid_b[g_offset] = (src_bid == -1) ? 0 : n_b;
            }

            __syncthreads();

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;

                if (sh_valid_b[g_offset] == 0)
                    continue;

                const int g = chunk_start_g + g_offset;
                const Ti bstr = bstrs_tile_start[b_offset];
                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(bstr, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const Ti ax = groups.axs[g];
                    int sa = -1;
                    Ti sas_a = 0;

                    if (ax == 0)
                    {
                        sa = a;
                        sas_a = astr;
                    }
                    else
                    {
                        sas_a = astr ^ ax;
                        sa = a_idx_map[sas_a];
                    }

                    if (sa != -1)
                    {
                        const int nza = groups.num_zas[g];
                        const Ti *zas = groups.flat_zas + groups.za_start[g];
                        const Tv *wa = groups.flat_wa + groups.wa_start[g];
                        const int rank = groups.ranks[g];

                        constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                        Tv pa[STACK_SIZE] = {};
                        compute_phase_dev<Rank, Ti, Tv>(sas_a, zas, nza, wa, pa, 1, rank);

                        const int sbi = sh_src_bid[g_offset];
                        const int src_n_b = basis.block_num_b[sbi];
                        const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * src_n_b;
                        const Tv *pb = sh_pb + (g_offset * TILE_B);

                        if (current_tile_b == TILE_B)
                        {
#pragma unroll
                            for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                // 注意：直接访问 b_tile_start + b_offset，去除了查表和条件判断
                                accum[b_offset] += __ldg(&src_base[b_tile_start + b_offset]) * vt;
                            }
                        }
                        else
                        {
                            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                accum[b_offset] += __ldg(&src_base[b_tile_start + b_offset]) * vt;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (valid_a)
        {
            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            {
                const int actual_b_idx = b_tile_start + b_offset;
                dst_base[actual_b_idx] += accum[b_offset];
            }
        }
    }
}

template <int Rank, typename Ti, typename Tv>
__global__ void hvec_gather_pure_b_kernel(
    const BasisSliceDev<Ti> basis,
    const GroupsSliceDev<Ti, Tv> groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int bid = basis.target_bids ? basis.target_bids[blockIdx.x] : blockIdx.x;
    const int total_groups = groups.num_groups;
    const int *b_idx_map = basis.bstr2idx;

    constexpr int SHARED_MEM_SIZE = Rank == 1 ? BATCH_SIZE_SH1 * TILE_B : Rank == 2 ? BATCH_SIZE_SH2 * TILE_B * 2
                                                                                    : BATCH_SIZE_SH3 * TILE_B * KERNEL_MAX_RANK;
    constexpr int BATCH_SIZE = Rank == 1 ? BATCH_SIZE_SH1 : Rank == 2 ? BATCH_SIZE_SH2
                                                                      : BATCH_SIZE_SH3;
    constexpr int IDX_MEM_SIZE = BATCH_SIZE * TILE_B;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sa_b[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid_b[BATCH_SIZE];

    const int num_chunks = (total_groups + BATCH_SIZE - 1) / BATCH_SIZE;
    const int n_a = basis.block_num_a[bid];
    const int n_b = basis.block_num_b[bid];
    const int asym = basis.block_asym[bid];
    const int bsym = basis.block_bsym[bid];
    const int nirp = basis.num_irreps;

    const int num_b_tiles = (n_b + TILE_B - 1) / TILE_B;
    const int num_a_tiles = (n_a + TILE_A - 1) / TILE_A;
    const int total_tiles = num_b_tiles * num_a_tiles;

    const Ti *astrs = basis.astrs_flat + basis.astrs_start[bid];
    const Ti *bstrs = basis.bstrs_flat + basis.bstrs_start[bid];
    Tv *dst_vec_bid = dst_vec + basis.block_offsets[bid];

    for (int task_idx = blockIdx.y; task_idx < total_tiles; task_idx += gridDim.y)
    {
        const int b_tile_idx = task_idx % num_b_tiles;
        const int a_tile_idx = task_idx / num_b_tiles;

        const int b_tile_start = b_tile_idx * TILE_B;
        const int current_tile_b = min(TILE_B, n_b - b_tile_start);
        const Ti *bstrs_tile_start = bstrs + b_tile_start;

        const int a_tile_start = a_tile_idx * TILE_A;
        const int a_tile_end = min(n_a, a_tile_start + TILE_A);

        const int a = a_tile_start + threadIdx.x;
        const bool valid_a = (a < a_tile_end);
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;

        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
            {
                const int g = chunk_start_g + g_offset;
                const int h = asym * nirp + (bsym ^ groups.bsyms[g]);
                const int src_bid = basis.block_map[h];
                sh_src_bid[g_offset] = src_bid;
                sh_valid_b[g_offset] = (src_bid == -1) ? 0 : n_b;
            }

            __syncthreads();

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;

                if (sh_valid_b[g_offset] == 0)
                    continue;

                const int g = chunk_start_g + g_offset;
                const Ti dst_b_str = bstrs_tile_start[b_offset];
                const Ti bx = groups.bxs[g];
                const Ti sas_b = dst_b_str ^ bx;

                const int sh_flat_offset = g_offset * TILE_B + b_offset;
                sh_sa_b[sh_flat_offset] = b_idx_map[sas_b];

                const Ti *zbs = groups.flat_zbs + groups.zb_start[g];
                const Tv *wb = groups.flat_wb + groups.wb_start[g];
                const int num_zb = groups.num_zbs[g];
                const int rank = groups.ranks[g];

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(sas_b, zbs, num_zb, wb, sh_pb_ptr, BATCH_SIZE * TILE_B, rank);
            }

            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if (sh_valid_b[g_offset] == 0)
                        continue;

                    const int g = chunk_start_g + g_offset;
                    const int sa = a;
                    const Ti sas_a = astr;

                    const int nza = groups.num_zas[g];
                    const Ti *zas = groups.flat_zas + groups.za_start[g];
                    const Tv *wa = groups.flat_wa + groups.wa_start[g];
                    const int rank = groups.ranks[g];

                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(sas_a, zas, nza, wa, pa, 1, rank);

                    const int sbi = sh_src_bid[g_offset];
                    const int src_n_b = basis.block_num_b[sbi];
                    const Tv *src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * src_n_b;
                    const Tv *pb = sh_pb + (g_offset * TILE_B);

                    const int sh_task_base_offset = g_offset * TILE_B;
                    const int *sh_sa_b_task = sh_sa_b + sh_task_base_offset;

                    if (current_tile_b == TILE_B)
                    {
#pragma unroll
                        for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                        {
                            const int sa_b = sh_sa_b_task[b_offset];
                            if (sa_b != -1)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                            }
                        }
                    }
                    else
                    {
                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            const int sa_b = sh_sa_b_task[b_offset];
                            if (sa_b != -1)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, rank, b_offset);
                                accum[b_offset] += __ldg(&src_base[sa_b]) * vt;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (valid_a)
        {
            for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
            {
                const int actual_b_idx = b_tile_start + b_offset;
                dst_base[actual_b_idx] += accum[b_offset];
            }
        }
    }
}

template <int TypeCode, typename Ti, typename Tv>
static inline void dispatch_chunks_by_rank_gpu(
    const BasisSliceDev<Ti> &basis_slice, int num_active_blocks,
    const GroupsViewDev<Ti, Tv> &groups,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    const int64 total_ngs = groups.num_groups;
    if (total_ngs == 0)
        return;

    int64 start = 0;
    while (start < total_ngs)
    {
        const int current_rank = groups.host_ranks[start];
        const int dispatch_rank = (current_rank == 1 || current_rank == 2) ? current_rank : 0;

        int64 end = start + 1;
        while (end < total_ngs)
        {
            const int next_rank = groups.host_ranks[end];
            const int next_dispatch_rank = (next_rank == 1 || next_rank == 2) ? next_rank : 0;
            if (next_dispatch_rank != dispatch_rank)
                break;
            end++;
        }

        const int64 chunk_size = end - start;

        GroupsSliceDev<Ti, Tv> slice;
        slice.num_groups = chunk_size;
        slice.axs = groups.axs + start;
        slice.bxs = groups.bxs + start;
        slice.asyms = groups.asyms + start;
        slice.bsyms = groups.bsyms + start;
        slice.ranks = groups.ranks + start;
        slice.num_zas = groups.num_zas + start;
        slice.num_zbs = groups.num_zbs + start;
        slice.za_start = groups.za_start + start;
        slice.zb_start = groups.zb_start + start;
        slice.wa_start = groups.wa_start + start;
        slice.wb_start = groups.wb_start + start;
        slice.flat_zas = groups.flat_zas;
        slice.flat_zbs = groups.flat_zbs;
        slice.flat_wa = groups.flat_wa;
        slice.flat_wb = groups.flat_wb;

        int block_size = 256;
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_size(num_active_blocks, num_sms * 4);

        if constexpr (TypeCode == 0)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_diag_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_diag_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_diag_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 1)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_pure_a_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_pure_a_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_pure_a_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 2)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_pure_b_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_pure_b_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_pure_b_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }
        else if constexpr (TypeCode == 3)
        {
            switch (dispatch_rank)
            {
            case 1:
                hvec_gather_mixed_kernel<1, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            case 2:
                hvec_gather_mixed_kernel<2, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            default:
                hvec_gather_mixed_kernel<0, Ti, Tv><<<grid_size, block_size>>>(basis_slice, slice, src_vec, dst_vec);
                break;
            }
        }

        start = end;
    }
}

template <typename Ti, typename Tv>
void cuda_hvec(
    const BasisViewDev<Ti> &basis,
    const NetworkDev<Ti, Tv> &net,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    cudaMemset(dst_vec, 0, basis.dim * sizeof(Tv));
    BasisSliceDev<Ti> slice = make_basis_slice(basis);
    
    dispatch_chunks_by_rank_gpu<0>(slice, basis.num_blocks, net.diag_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<1>(slice, basis.num_blocks, net.pure_a_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<2>(slice, basis.num_blocks, net.pure_b_groups, src_vec, dst_vec);
    dispatch_chunks_by_rank_gpu<3>(slice, basis.num_blocks, net.mixed_groups, src_vec, dst_vec);
}
