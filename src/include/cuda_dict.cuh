#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>

// Header-only CUDA open-addressing dictionary.
//
// Key type Ti:
//   - must be uint32_t or uint64_t, or an equivalent unsigned 32/64-bit type.
//   - empty-slot sentinel is all-bits-one for Ti.
//     For Ti = uint32_t, this is 0xFFFFFFFFu.
//     For Ti = uint64_t, this is 0xFFFFFFFFFFFFFFFFull.
//
// Value type Tv:
//   - stored and loaded directly. No atomic operation is performed on Tv.
//   - lookup needs an explicit not_found_value from the caller.
//
// Important usage contract:
//   - capacity must be a non-zero power of two.
//   - real keys must not equal cuda_dict::empty_key<Ti>().
//   - insert kernels must finish before lookup kernels begin.
//     This implementation does not support concurrent insert + lookup.
//   - duplicate keys use first-writer-wins semantics: after a key is published,
//     later duplicate inserts do not overwrite its value.

#ifndef CUDA_DICT_CHECK
#define CUDA_DICT_CHECK(call)                                                     \
    do                                                                            \
    {                                                                             \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess)                                                 \
        {                                                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err__) << std::endl;                  \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

namespace cuda_dict
{

    // Device-side status flags. The host should allocate one uint32_t status flag,
    // initialize it to 0, and inspect it after insert kernels complete.
    enum Status : uint32_t
    {
        STATUS_OK = 0x0u,
        STATUS_INVALID_KEY = 0x1u,
        STATUS_INSERT_FAIL = 0x2u
    };

    template <typename Ti>
    struct is_supported_uint_key
    {
        static constexpr bool value =
            std::is_integral<Ti>::value &&
            std::is_unsigned<Ti>::value &&
            (sizeof(Ti) == 4 || sizeof(Ti) == 8);
    };

    template <typename Ti>
    __host__ __device__ __forceinline__ constexpr Ti empty_key()
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");
        return static_cast<Ti>(~static_cast<Ti>(0));
    }

    __host__ __device__ __forceinline__ constexpr bool is_power_of_two(size_t x)
    {
        return x != 0 && ((x & (x - 1)) == 0);
    }

    template <typename Ti>
    __host__ __forceinline__ size_t key_storage_bytes(size_t capacity)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");
        return capacity * sizeof(Ti);
    }

    // Initialize table_keys to empty_key<Ti>(), i.e. all bits set to 1.
    // For Ti = uint32_t, this fills keys with 0xFFFFFFFFu.
    template <typename Ti>
    __host__ __forceinline__ cudaError_t memset_empty_keys_async(
        Ti *d_table_keys,
        size_t capacity,
        cudaStream_t stream = 0)
    {
        return cudaMemsetAsync(d_table_keys, 0xFF, key_storage_bytes<Ti>(capacity), stream);
    }

    template <typename Tv>
    __host__ __forceinline__ cudaError_t memset_values_zero_async(
        Tv *d_table_values,
        size_t capacity,
        cudaStream_t stream = 0)
    {
        return cudaMemsetAsync(d_table_values, 0, capacity * sizeof(Tv), stream);
    }

    // SplitMix64 finalizer. Works well for both 32-bit and 64-bit unsigned keys.
    template <typename Ti>
    __device__ __forceinline__ size_t hash_slot(Ti key, size_t capacity)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        uint64_t x = static_cast<uint64_t>(key);
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        x = x ^ (x >> 31);

        // capacity must be a power of two.
        return static_cast<size_t>(x) & (capacity - 1u);
    }

    // Native 32-bit atomicCAS path.
    template <typename Ti>
    __device__ __forceinline__ typename std::enable_if<sizeof(Ti) == 4, Ti>::type
    atomic_cas_key(Ti *addr, Ti compare, Ti value)
    {
        unsigned int old = atomicCAS(
            reinterpret_cast<unsigned int *>(addr),
            static_cast<unsigned int>(compare),
            static_cast<unsigned int>(value));
        return static_cast<Ti>(old);
    }

    // Native 64-bit atomicCAS path.
    template <typename Ti>
    __device__ __forceinline__ typename std::enable_if<sizeof(Ti) == 8, Ti>::type
    atomic_cas_key(Ti *addr, Ti compare, Ti value)
    {
        unsigned long long int old = atomicCAS(
            reinterpret_cast<unsigned long long int *>(addr),
            static_cast<unsigned long long int>(compare),
            static_cast<unsigned long long int>(value));
        return static_cast<Ti>(old);
    }

    // Insert one key/value pair into SoA table arrays.
    // Returns false for invalid key or full/probe-failed table.
    template <typename Ti, typename Tv>
    __device__ __forceinline__ bool insert_device(
        Ti *__restrict__ table_keys,
        Tv *__restrict__ table_values,
        Ti key,
        Tv value,
        size_t capacity)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        if (key == empty_key<Ti>())
        {
            return false;
        }

        size_t slot = hash_slot(key, capacity);

        for (size_t probe = 0; probe < capacity; ++probe)
        {
            Ti prev_key = atomic_cas_key(&table_keys[slot], empty_key<Ti>(), key);

            if (prev_key == empty_key<Ti>())
            {
                // This thread published the key and owns this value write.
                // Lookups must happen only after the insert kernel has completed.
                table_values[slot] = value;
                return true;
            }

            if (prev_key == key)
            {
                // Duplicate key: keep the first writer's value.
                return true;
            }

            slot = (slot + 1u) & (capacity - 1u);
        }

        return false;
    }

    // Lookup one key. Returns not_found_value if missing or invalid.
    template <typename Ti, typename Tv>
    __device__ __forceinline__ Tv lookup_device(
        const Ti *__restrict__ table_keys,
        const Tv *__restrict__ table_values,
        Ti key,
        size_t capacity,
        Tv not_found_value)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        if (key == empty_key<Ti>())
        {
            return not_found_value;
        }

        size_t slot = hash_slot(key, capacity);

        for (size_t probe = 0; probe < capacity; ++probe)
        {
            Ti current_key = table_keys[slot];

            if (current_key == key)
            {
                return table_values[slot];
            }

            if (current_key == empty_key<Ti>())
            {
                return not_found_value;
            }

            slot = (slot + 1u) & (capacity - 1u);
        }

        return not_found_value;
    }

    // Batch insert kernel: input_values[idx] is inserted as value.
    template <typename Ti, typename Tv>
    __global__ void batch_insert_kernel(
        Ti *__restrict__ table_keys,
        Tv *__restrict__ table_values,
        const Ti *__restrict__ input_keys,
        const Tv *__restrict__ input_values,
        size_t num_elements,
        size_t capacity,
        uint32_t *__restrict__ status)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (; idx < num_elements; idx += stride)
        {
            const Ti key = input_keys[idx];
            const bool ok = insert_device(table_keys, table_values, key, input_values[idx], capacity);

            if (!ok && status != nullptr)
            {
                const uint32_t flag = (key == empty_key<Ti>()) ? STATUS_INVALID_KEY : STATUS_INSERT_FAIL;
                atomicOr(status, flag);
            }
        }
    }

    // Initial key -> index kernel: table[input_keys[idx]] = static_cast<Tv>(idx).
    // Useful for basis-state-to-index maps.
    template <typename Ti, typename Tv>
    __global__ void initial_key2idx_kernel(
        Ti *__restrict__ table_keys,
        Tv *__restrict__ table_values,
        const Ti *__restrict__ input_keys,
        size_t num_elements,
        size_t capacity,
        uint32_t *__restrict__ status)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (; idx < num_elements; idx += stride)
        {
            const Ti key = input_keys[idx];
            const bool ok = insert_device(
                table_keys,
                table_values,
                key,
                static_cast<Tv>(idx),
                capacity);

            if (!ok && status != nullptr)
            {
                const uint32_t flag = (key == empty_key<Ti>()) ? STATUS_INVALID_KEY : STATUS_INSERT_FAIL;
                atomicOr(status, flag);
            }
        }
    }

    // Batch lookup kernel.
    template <typename Ti, typename Tv>
    __global__ void batch_lookup_kernel(
        const Ti *__restrict__ table_keys,
        const Tv *__restrict__ table_values,
        const Ti *__restrict__ input_keys,
        Tv *__restrict__ results,
        size_t num_elements,
        size_t capacity,
        Tv not_found_value)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (; idx < num_elements; idx += stride)
        {
            results[idx] = lookup_device(
                table_keys,
                table_values,
                input_keys[idx],
                capacity,
                not_found_value);
        }
    }

    // Convenience launchers. They check host-side capacity and kernel-launch errors,
    // but they do not synchronize unless the caller calls cudaDeviceSynchronize() or
    // uses stream/event synchronization.
    template <typename Ti, typename Tv>
    __host__ inline void launch_batch_insert(
        Ti *d_table_keys,
        Tv *d_table_values,
        const Ti *d_input_keys,
        const Tv *d_input_values,
        size_t num_elements,
        size_t capacity,
        uint32_t *d_status,
        int block_size = 256,
        cudaStream_t stream = 0)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        if (num_elements == 0)
        {
            return;
        }
        if (!is_power_of_two(capacity))
        {
            std::cerr << "cuda_dict error: capacity must be a non-zero power of two." << std::endl;
            std::abort();
        }
        if (block_size <= 0)
        {
            std::cerr << "cuda_dict error: block_size must be positive." << std::endl;
            std::abort();
        }

        const int grid_size = static_cast<int>((num_elements + static_cast<size_t>(block_size) - 1u) /
                                               static_cast<size_t>(block_size));
        batch_insert_kernel<Ti, Tv><<<grid_size, block_size, 0, stream>>>(
            d_table_keys,
            d_table_values,
            d_input_keys,
            d_input_values,
            num_elements,
            capacity,
            d_status);
        CUDA_DICT_CHECK(cudaGetLastError());
    }

    template <typename Ti, typename Tv>
    __host__ inline void launch_initial_key2idx(
        Ti *d_table_keys,
        Tv *d_table_values,
        const Ti *d_input_keys,
        size_t num_elements,
        size_t capacity,
        uint32_t *d_status,
        int block_size = 256,
        cudaStream_t stream = 0)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        if (num_elements == 0)
        {
            return;
        }
        if (!is_power_of_two(capacity))
        {
            std::cerr << "cuda_dict error: capacity must be a non-zero power of two." << std::endl;
            std::abort();
        }
        if (block_size <= 0)
        {
            std::cerr << "cuda_dict error: block_size must be positive." << std::endl;
            std::abort();
        }

        const int grid_size = static_cast<int>((num_elements + static_cast<size_t>(block_size) - 1u) /
                                               static_cast<size_t>(block_size));
        initial_key2idx_kernel<Ti, Tv><<<grid_size, block_size, 0, stream>>>(
            d_table_keys,
            d_table_values,
            d_input_keys,
            num_elements,
            capacity,
            d_status);
        CUDA_DICT_CHECK(cudaGetLastError());
    }

    template <typename Ti, typename Tv>
    __host__ inline void launch_batch_lookup(
        const Ti *d_table_keys,
        const Tv *d_table_values,
        const Ti *d_input_keys,
        Tv *d_results,
        size_t num_elements,
        size_t capacity,
        Tv not_found_value,
        int block_size = 256,
        cudaStream_t stream = 0)
    {
        static_assert(is_supported_uint_key<Ti>::value,
                      "Ti must be an unsigned integer type with width 32 or 64 bits");

        if (num_elements == 0)
        {
            return;
        }
        if (!is_power_of_two(capacity))
        {
            std::cerr << "cuda_dict error: capacity must be a non-zero power of two." << std::endl;
            std::abort();
        }
        if (block_size <= 0)
        {
            std::cerr << "cuda_dict error: block_size must be positive." << std::endl;
            std::abort();
        }

        const int grid_size = static_cast<int>((num_elements + static_cast<size_t>(block_size) - 1u) /
                                               static_cast<size_t>(block_size));
        batch_lookup_kernel<Ti, Tv><<<grid_size, block_size, 0, stream>>>(
            d_table_keys,
            d_table_values,
            d_input_keys,
            d_results,
            num_elements,
            capacity,
            not_found_value);
        CUDA_DICT_CHECK(cudaGetLastError());
    }

} // namespace cuda_dict
