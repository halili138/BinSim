#pragma once
#include "cuda_utils.cuh"
#include "cuda_basis.cuh"
#include "cuda_otf.cuh"

template <int Rank, int TypeCode, typename Ti, typename Tv>
__global__ void hvec_gather_kernel(
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

    constexpr bool UsesAExcitation = TypeCode == 1 || TypeCode == 3;
    constexpr bool UsesBExcitation = TypeCode == 2 || TypeCode == 3;
    constexpr bool IsDiagonal = TypeCode == 0;
    constexpr int IDX_MEM_SIZE = UsesBExcitation ? BATCH_SIZE * TILE_B : 1;

    __shared__ Tv sh_pb[SHARED_MEM_SIZE];
    __shared__ int sh_sa_b[IDX_MEM_SIZE];
    __shared__ int sh_src_bid[BATCH_SIZE];
    __shared__ int sh_valid_group[BATCH_SIZE];

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
    const int *a_idx_map = basis.astr2idx;
    const int *b_idx_map = basis.bstr2idx;
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
        const bool valid_a = a < a_tile_end;
        const Ti astr = valid_a ? astrs[a] : 0;
        Tv *dst_base = valid_a ? (dst_vec_bid + (int64)a * n_b) : nullptr;

        Tv accum[TILE_B] = {};

        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
        {
            const int chunk_start_g = chunk_idx * BATCH_SIZE;
            const int current_chunk_groups = min(BATCH_SIZE, total_groups - chunk_start_g);

            if constexpr (!IsDiagonal)
            {
                for (int g_offset = threadIdx.x; g_offset < current_chunk_groups; g_offset += blockDim.x)
                {
                    const int g = chunk_start_g + g_offset;
                    int h;
                    if constexpr (TypeCode == 1)
                        h = (asym ^ groups.asyms[g]) * nirp + bsym;
                    else if constexpr (TypeCode == 2)
                        h = asym * nirp + (bsym ^ groups.bsyms[g]);
                    else
                        h = (asym ^ groups.asyms[g]) * nirp + (bsym ^ groups.bsyms[g]);

                    const int src_bid = basis.block_map[h];
                    sh_src_bid[g_offset] = src_bid;
                    sh_valid_group[g_offset] = src_bid != -1;
                }
                __syncthreads();
            }

            const int total_sh_elements = current_chunk_groups * current_tile_b;
            for (int sh_idx = threadIdx.x; sh_idx < total_sh_elements; sh_idx += blockDim.x)
            {
                const int g_offset = sh_idx / current_tile_b;
                const int b_offset = sh_idx % current_tile_b;

                if constexpr (!IsDiagonal)
                {
                    if (!sh_valid_group[g_offset])
                        continue;
                }

                const int g = chunk_start_g + g_offset;
                const Ti bstr = bstrs_tile_start[b_offset];
                Ti phase_bstr = bstr;
                if constexpr (UsesBExcitation)
                {
                    phase_bstr = bstr ^ groups.bxs[g];
                    sh_sa_b[g_offset * TILE_B + b_offset] = b_idx_map[phase_bstr];
                }

                Tv *sh_pb_ptr = sh_pb + (g_offset * TILE_B + b_offset);
                compute_phase_dev<Rank, Ti, Tv>(
                    phase_bstr, groups.flat_zbs + groups.zb_start[g], groups.num_zbs[g],
                    groups.flat_wb + groups.wb_start[g], sh_pb_ptr, BATCH_SIZE * TILE_B, groups.ranks[g]);
            }
            __syncthreads();

            if (valid_a)
            {
                for (int g_offset = 0; g_offset < current_chunk_groups; ++g_offset)
                {
                    if constexpr (!IsDiagonal)
                    {
                        if (!sh_valid_group[g_offset])
                            continue;
                    }

                    const int g = chunk_start_g + g_offset;
                    int sa = a;
                    Ti phase_astr = astr;
                    if constexpr (UsesAExcitation)
                    {
                        const Ti ax = groups.axs[g];
                        if (ax != 0)
                        {
                            phase_astr = astr ^ ax;
                            sa = a_idx_map[phase_astr];
                        }
                    }

                    if (sa == -1)
                        continue;

                    constexpr int STACK_SIZE = Rank == 1 ? 1 : (Rank == 2 ? 2 : 128);
                    Tv pa[STACK_SIZE] = {};
                    compute_phase_dev<Rank, Ti, Tv>(
                        phase_astr, groups.flat_zas + groups.za_start[g], groups.num_zas[g],
                        groups.flat_wa + groups.wa_start[g], pa, 1, groups.ranks[g]);

                    const Tv *src_base;
                    if constexpr (IsDiagonal)
                    {
                        src_base = src_vec_bid + (int64)a * n_b;
                    }
                    else
                    {
                        const int sbi = sh_src_bid[g_offset];
                        src_base = src_vec + basis.block_offsets[sbi] + (int64)sa * basis.block_num_b[sbi];
                    }

                    const Tv *pb = sh_pb + (g_offset * TILE_B);

                    if (current_tile_b == TILE_B)
                    {
#pragma unroll
                        for (int b_offset = 0; b_offset < TILE_B; ++b_offset)
                        {
                            int src_b = b_tile_start + b_offset;
                            if constexpr (UsesBExcitation)
                                src_b = sh_sa_b[g_offset * TILE_B + b_offset];
                            if (src_b != -1)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, groups.ranks[g], b_offset);
                                accum[b_offset] += __ldg(&src_base[src_b]) * vt;
                            }
                        }
                    }
                    else
                    {
                        for (int b_offset = 0; b_offset < current_tile_b; ++b_offset)
                        {
                            int src_b = b_tile_start + b_offset;
                            if constexpr (UsesBExcitation)
                                src_b = sh_sa_b[g_offset * TILE_B + b_offset];
                            if (src_b != -1)
                            {
                                const Tv vt = compute_coeff_dev<Rank, Tv>(pa, pb, BATCH_SIZE * TILE_B, groups.ranks[g], b_offset);
                                accum[b_offset] += __ldg(&src_base[src_b]) * vt;
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
                dst_base[b_tile_start + b_offset] += accum[b_offset];
        }
    }
}

template <int Rank, int TypeCode, typename Ti, typename Tv>
static inline void launch_hvec_chunk(
    const BasisSliceDev<Ti> &basis_slice,
    const GroupsSliceDev<Ti, Tv> &groups_slice,
    dim3 grid_size,
    int block_size,
    const Tv *__restrict__ src_vec,
    Tv *__restrict__ dst_vec)
{
    hvec_gather_kernel<Rank, TypeCode, Ti, Tv><<<grid_size, block_size>>>(basis_slice, groups_slice, src_vec, dst_vec);
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

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    constexpr int block_size = 256;
    const dim3 grid_size(num_active_blocks, num_sms * 4);

    int64 start = 0;
    while (start < total_ngs)
    {
        const int dispatch_rank = normalized_dispatch_rank(groups, start);
        const int64 end = next_rank_chunk_end(groups, start);
        const GroupsSliceDev<Ti, Tv> slice = make_groups_slice(groups, start, end - start);

        switch (dispatch_rank)
        {
        case 1:
            launch_hvec_chunk<1, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, src_vec, dst_vec);
            break;
        case 2:
            launch_hvec_chunk<2, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, src_vec, dst_vec);
            break;
        default:
            launch_hvec_chunk<0, TypeCode, Ti, Tv>(basis_slice, slice, grid_size, block_size, src_vec, dst_vec);
            break;
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
