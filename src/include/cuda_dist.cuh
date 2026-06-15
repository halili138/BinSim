#pragma once
#include <vector>
#include <algorithm>
#include "dist_hvec.hpp"
#include "cuda_hvec.cuh"

struct SubTopologyDev
{
    int64 send_dim = 0;
    int64 recv_dim = 0;
    std::vector<int> send_counts;
    std::vector<int> recv_counts;

    // --- GPU VRAM 驻留数据 ---
    int num_targets = 0;
    int *d_target_bids = nullptr;
    int64 *d_topo_offsets = nullptr;

    int num_pack_jobs = 0;
    PackJob *d_pack_jobs = nullptr;

    ~SubTopologyDev()
    {
        if (d_target_bids)
        {
            cudaFree(d_target_bids);
            d_target_bids = nullptr;
        }
        if (d_topo_offsets)
        {
            cudaFree(d_topo_offsets);
            d_topo_offsets = nullptr;
        }
        if (d_pack_jobs)
        {
            cudaFree(d_pack_jobs);
            d_pack_jobs = nullptr;
        }
    }
};

template <typename Ti, typename Tv>
SubTopologyDev *build_sub_topology_gpu_impl(
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *sub_net, const GlobalMemMap *gmap, int num_phases, int phase_idx)
{
    SubTopology *h_topo = build_sub_topology<Ti, Tv>(basis, sub_net, gmap, num_phases, phase_idx);

    SubTopologyDev *d_topo = new SubTopologyDev();

    d_topo->send_dim = h_topo->send_dim;
    d_topo->recv_dim = h_topo->recv_dim;
    d_topo->send_counts = std::move(h_topo->send_counts);
    d_topo->recv_counts = std::move(h_topo->recv_counts);

    // =========================================================================
    // 【核心修复】：将 CPU 松散的对称性 ID (h) 映射为 GPU 紧凑的 Block Index (bid)
    // =========================================================================

    // 1. 映射 target_bids
    d_topo->num_targets = h_topo->target_blocks.size();
    std::vector<int> gpu_target_bids;
    gpu_target_bids.reserve(d_topo->num_targets);
    for (int h : h_topo->target_blocks)
    {
        // 通过 block_map 将物理 h 转为连续的 bid
        gpu_target_bids.push_back((int)basis->block_map[h]);
    }

    // 2. 映射 topo_offsets
    std::vector<int64> gpu_topo_offsets(basis->num_blocks, -1);
    for (int i = 0; i < basis->num_blocks; ++i)
    {
        int64 h = basis->blocks[i].asym * basis->num_irreps + basis->blocks[i].bsym;
        // 把索引为 h 的偏移量，塞进索引为 bid (即 i) 的数组中
        gpu_topo_offsets[i] = h_topo->block_offsets_in_cache[h];
    }
    // =========================================================================

    d_topo->num_pack_jobs = h_topo->pack_jobs.size();

    // 将映射后的干净数据上传到 GPU
    if (d_topo->num_targets > 0)
    {
        d_topo->d_target_bids = up(gpu_target_bids.data(), d_topo->num_targets);
    }

    if (basis->num_blocks > 0)
    {
        d_topo->d_topo_offsets = up(gpu_topo_offsets.data(), basis->num_blocks);
    }

    if (d_topo->num_pack_jobs > 0)
    {
        d_topo->d_pack_jobs = up(h_topo->pack_jobs.data(), d_topo->num_pack_jobs);
    }

    delete h_topo; // 清理 CPU 临时拓扑

    return d_topo;
}

// =================================================================
// GPU 高速显存打包内核
// =================================================================
template <typename Tv>
__global__ void pack_send_buffer_gpu_kernel(const PackJob *jobs, int num_jobs, const Tv *__restrict__ local_v, Tv *__restrict__ send_buffer)
{
    const int job_idx = blockIdx.x;
    if (job_idx < num_jobs)
    {
        PackJob job = jobs[job_idx];
        for (int64 i = threadIdx.x; i < job.size; i += blockDim.x)
        {
            send_buffer[job.dst_offset + i] = local_v[job.src_offset + i];
        }
    }
}
