#pragma once
#include "cuda_framework.cuh"
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

struct CudaSingleGroupSchedule
{
    int max_tasks = 0;
    int64 total_tasks = 0;
    int64 rectangular_tasks = 0;
    CudaSingleGroupTask *dev_tasks = nullptr;

    CudaSingleGroupSchedule() = default;
    CudaSingleGroupSchedule(const CudaSingleGroupSchedule &) = delete;
    CudaSingleGroupSchedule &operator=(const CudaSingleGroupSchedule &) = delete;

    CudaSingleGroupSchedule(CudaSingleGroupSchedule &&other) noexcept
    {
        *this = std::move(other);
    }

    CudaSingleGroupSchedule &operator=(CudaSingleGroupSchedule &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            max_tasks = other.max_tasks;
            total_tasks = other.total_tasks;
            rectangular_tasks = other.rectangular_tasks;
            dev_tasks = other.dev_tasks;
            other.max_tasks = 0;
            other.total_tasks = 0;
            other.rectangular_tasks = 0;
            other.dev_tasks = nullptr;
        }
        return *this;
    }

    ~CudaSingleGroupSchedule() { clear(); }

    void clear()
    {
        if (dev_tasks)
        {
            cudaFree(dev_tasks);
            dev_tasks = nullptr;
        }
    }

    bool compact_enabled() const { return dev_tasks != nullptr && total_tasks > 0; }

    double early_return_rate() const
    {
        return rectangular_tasks == 0 ? 0.0 : double(rectangular_tasks - total_tasks) / double(rectangular_tasks);
    }
};

struct CudaMultiGroupSchedule
{
    int64 total_tiles = 0;
    int64 rectangular_tasks = 0;
    CudaMultiGroupTask *dev_tasks = nullptr;

    CudaMultiGroupSchedule() = default;
    CudaMultiGroupSchedule(const CudaMultiGroupSchedule &) = delete;
    CudaMultiGroupSchedule &operator=(const CudaMultiGroupSchedule &) = delete;

    CudaMultiGroupSchedule(CudaMultiGroupSchedule &&other) noexcept
    {
        *this = std::move(other);
    }

    CudaMultiGroupSchedule &operator=(CudaMultiGroupSchedule &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            total_tiles = other.total_tiles;
            rectangular_tasks = other.rectangular_tasks;
            dev_tasks = other.dev_tasks;
            other.total_tiles = 0;
            other.rectangular_tasks = 0;
            other.dev_tasks = nullptr;
        }
        return *this;
    }

    ~CudaMultiGroupSchedule() { clear(); }

    void clear()
    {
        if (dev_tasks)
        {
            cudaFree(dev_tasks);
            dev_tasks = nullptr;
        }
    }

    bool compact_enabled() const { return dev_tasks != nullptr && total_tiles > 0; }
};

template <typename Ti>
static inline CudaMultiGroupSchedule cuda_multi_group_schedule(const BasisViewDev<Ti> &basis, int num_sms)
{
    CudaMultiGroupSchedule schedule;
    std::vector<CudaMultiGroupTask> host_tasks;

    for (int active_block_idx = 0; active_block_idx < basis.num_blocks; ++active_block_idx)
    {
        const int num_a_tiles = (basis.host_block_num_a[active_block_idx] + TILE_A - 1) / TILE_A;
        const int num_b_tiles = (basis.host_block_num_b[active_block_idx] + TILE_B - 1) / TILE_B;
        const int block_tiles = num_a_tiles * num_b_tiles;
        for (int tile_idx = 0; tile_idx < block_tiles; ++tile_idx)
            host_tasks.push_back(CudaMultiGroupTask{active_block_idx, tile_idx});
    }

    schedule.total_tiles = static_cast<int64>(host_tasks.size());
    schedule.rectangular_tasks = static_cast<int64>(basis.num_blocks) * num_sms * 4;

    const bool grid_fits = schedule.total_tiles <= std::numeric_limits<unsigned int>::max();
    const bool compact_is_smaller = schedule.total_tiles < schedule.rectangular_tasks;
    const bool high_idle = schedule.rectangular_tasks > 0 && schedule.total_tiles * 2 < schedule.rectangular_tasks;
    if (!host_tasks.empty() && grid_fits && compact_is_smaller && high_idle)
    {
        CUDA_CHECK(cudaMalloc(&schedule.dev_tasks, host_tasks.size() * sizeof(CudaMultiGroupTask)));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks, host_tasks.data(), host_tasks.size() * sizeof(CudaMultiGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
}

template <typename Ti>
static inline CudaSingleGroupSchedule cuda_single_group_schedule(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = 256;
    CudaSingleGroupSchedule schedule;
    std::vector<CudaSingleGroupTask> host_tasks;

    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        const int block_tasks = num_a_tiles * num_b_tiles;
        schedule.max_tasks = std::max(schedule.max_tasks, block_tasks);
        for (int task_idx = 0; task_idx < block_tasks; ++task_idx)
            host_tasks.push_back(CudaSingleGroupTask{bid, task_idx});
    }

    schedule.total_tasks = static_cast<int64>(host_tasks.size());
    schedule.rectangular_tasks = static_cast<int64>(basis.num_blocks) * schedule.max_tasks;

    if (!host_tasks.empty() && schedule.total_tasks < schedule.rectangular_tasks)
    {
        CUDA_CHECK(cudaMalloc(&schedule.dev_tasks, host_tasks.size() * sizeof(CudaSingleGroupTask)));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks, host_tasks.data(), host_tasks.size() * sizeof(CudaSingleGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
}

template <typename Ti>
static inline int cuda_single_group_max_tasks(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = 256;
    int max_tasks = 0;
    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        max_tasks = std::max(max_tasks, num_a_tiles * num_b_tiles);
    }
    return max_tasks;
}