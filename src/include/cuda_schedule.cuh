#pragma once
#include "cuda_framework.cuh"
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

template <typename Task>
struct CudaDeviceTaskBuffer
{
    Task *ptr = nullptr;
    int64 count = 0;

    CudaDeviceTaskBuffer() = default;
    CudaDeviceTaskBuffer(const CudaDeviceTaskBuffer &) = delete;
    CudaDeviceTaskBuffer &operator=(const CudaDeviceTaskBuffer &) = delete;

    CudaDeviceTaskBuffer(CudaDeviceTaskBuffer &&other) noexcept
    {
        *this = std::move(other);
    }

    CudaDeviceTaskBuffer &operator=(CudaDeviceTaskBuffer &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    ~CudaDeviceTaskBuffer() { reset(); }

    void reset() noexcept
    {
        if (ptr)
        {
            // Destructors and noexcept reset paths must not throw/check; callers that need
            // error reporting should use clear_checked() explicitly.
            cudaFree(ptr);
            ptr = nullptr;
            count = 0;
        }
    }

    void reset(Task *new_ptr, int64 new_count) noexcept
    {
        reset();
        ptr = new_ptr;
        count = new_ptr ? new_count : 0;
    }

    void clear_checked()
    {
        if (ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            count = 0;
        }
    }

    Task *get() const noexcept { return ptr; }
    int64 size() const noexcept { return count; }
};

static inline bool should_use_compact_schedule(int64 total_tasks, int64 rectangular_tasks, bool grid_fits, double idle_threshold)
{
    if (!grid_fits || total_tasks <= 0 || rectangular_tasks <= 0 || total_tasks >= rectangular_tasks)
        return false;

    const double idle_rate = double(rectangular_tasks - total_tasks) / double(rectangular_tasks);
    return idle_rate > idle_threshold;
}

struct CudaSingleGroupSchedule
{
    int max_tasks = 0;
    int64 total_tasks = 0;
    int64 rectangular_tasks = 0;
    CudaDeviceTaskBuffer<CudaSingleGroupTask> dev_tasks;

    CudaSingleGroupSchedule() = default;
    CudaSingleGroupSchedule(const CudaSingleGroupSchedule &) = delete;
    CudaSingleGroupSchedule &operator=(const CudaSingleGroupSchedule &) = delete;
    CudaSingleGroupSchedule(CudaSingleGroupSchedule &&) noexcept = default;
    CudaSingleGroupSchedule &operator=(CudaSingleGroupSchedule &&) noexcept = default;

    int64 compact_task_count() const { return total_tasks; }
    int64 rectangular_task_count() const { return rectangular_tasks; }
    bool compact_enabled() const { return dev_tasks.get() != nullptr && compact_task_count() > 0; }

    double early_return_rate() const
    {
        return rectangular_task_count() == 0 ? 0.0 : double(rectangular_task_count() - compact_task_count()) / double(rectangular_task_count());
    }
};

struct CudaMultiGroupSchedule
{
    int64 total_tiles = 0;
    int64 rectangular_tasks = 0;
    CudaDeviceTaskBuffer<CudaMultiGroupTask> dev_tasks;

    CudaMultiGroupSchedule() = default;
    CudaMultiGroupSchedule(const CudaMultiGroupSchedule &) = delete;
    CudaMultiGroupSchedule &operator=(const CudaMultiGroupSchedule &) = delete;
    CudaMultiGroupSchedule(CudaMultiGroupSchedule &&) noexcept = default;
    CudaMultiGroupSchedule &operator=(CudaMultiGroupSchedule &&) noexcept = default;

    int64 compact_task_count() const { return total_tiles; }
    int64 rectangular_task_count() const { return rectangular_tasks; }
    bool compact_enabled() const { return dev_tasks.get() != nullptr && compact_task_count() > 0; }
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
    schedule.rectangular_tasks = static_cast<int64>(basis.num_blocks) * num_sms * kBlocksPerSmForRectangularGrid;

    constexpr double kCompactIdleThreshold = 0.5;
    const bool grid_fits = schedule.compact_task_count() <= std::numeric_limits<unsigned int>::max();
    if (should_use_compact_schedule(schedule.compact_task_count(), schedule.rectangular_task_count(), grid_fits, kCompactIdleThreshold))
    {
        CudaMultiGroupTask *dev_tasks = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_tasks, host_tasks.size() * sizeof(CudaMultiGroupTask)));
        schedule.dev_tasks.reset(dev_tasks, static_cast<int64>(host_tasks.size()));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks.get(), host_tasks.data(), host_tasks.size() * sizeof(CudaMultiGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
}

template <typename Ti>
static inline CudaSingleGroupSchedule cuda_single_group_schedule(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = kCudaBlockSize;
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

    constexpr double kCompactIdleThreshold = 0.0;
    const bool grid_fits = schedule.compact_task_count() <= std::numeric_limits<unsigned int>::max();
    if (should_use_compact_schedule(schedule.compact_task_count(), schedule.rectangular_task_count(), grid_fits, kCompactIdleThreshold))
    {
        CudaSingleGroupTask *dev_tasks = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_tasks, host_tasks.size() * sizeof(CudaSingleGroupTask)));
        schedule.dev_tasks.reset(dev_tasks, static_cast<int64>(host_tasks.size()));
        CUDA_CHECK(cudaMemcpy(schedule.dev_tasks.get(), host_tasks.data(), host_tasks.size() * sizeof(CudaSingleGroupTask), cudaMemcpyHostToDevice));
    }

    return schedule;
}

template <typename Ti>
static inline int cuda_single_group_max_tasks(const BasisViewDev<Ti> &basis)
{
    constexpr int block_size = kCudaBlockSize;
    int max_tasks = 0;
    for (int bid = 0; bid < basis.num_blocks; ++bid)
    {
        const int num_a_tiles = (basis.host_block_num_a[bid] + block_size - 1) / block_size;
        const int num_b_tiles = (basis.host_block_num_b[bid] + TILE_B - 1) / TILE_B;
        max_tasks = std::max(max_tasks, num_a_tiles * num_b_tiles);
    }
    return max_tasks;
}