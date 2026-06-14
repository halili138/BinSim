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
    const BasisManager<Ti> *basis, const Network_OTF<Ti, Tv> *sub_net, const GlobalMemMap *gmap)
{
    SubTopology *h_topo = build_sub_topology<Ti, Tv>(basis, sub_net, gmap);

    SubTopologyDev *d_topo = new SubTopologyDev();

    // 剥离并转移 MPI 通信所需的元数据 (Host -> Host)
    // 既然 h_topo 马上要销毁, 使用 std::move 避免 vector 的深拷贝! 
    d_topo->send_dim = h_topo->send_dim;
    d_topo->recv_dim = h_topo->recv_dim;
    d_topo->send_counts = std::move(h_topo->send_counts);
    d_topo->recv_counts = std::move(h_topo->recv_counts);

    // 将寻址映射与打包任务上传到显存 (Host -> Device VRAM)
    d_topo->num_targets = h_topo->target_blocks.size();
    d_topo->num_pack_jobs = h_topo->pack_jobs.size();

    if (d_topo->num_targets > 0)
    {
        d_topo->d_target_bids = up(h_topo->target_blocks.data(), d_topo->num_targets);
    }

    int64 offsets_size = h_topo->block_offsets_in_cache.size();
    if (offsets_size > 0)
    {
        d_topo->d_topo_offsets = up(h_topo->block_offsets_in_cache.data(), offsets_size);
    }

    if (d_topo->num_pack_jobs > 0)
    {
        d_topo->d_pack_jobs = up(h_topo->pack_jobs.data(), d_topo->num_pack_jobs);
    }

    // 销毁临时的 CPU 拓扑对象, 防止内存泄漏! 
    delete h_topo;

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
