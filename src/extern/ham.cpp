#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <bit>
#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <parallel/algorithm>

#include <ankerl/unordered_dense.h>
#include "bitintegers.hpp"

template <typename Ti>
struct Pauli
{
    Ti z;
    Ti x;

    bool operator<(const Pauli &o) const
    {
        if (x != o.x)
            return x < o.x;
        return z < o.z;
    }

    bool operator==(const Pauli &o) const { return x == o.x && z == o.z; }

    bool operator!=(const Pauli &o) const { return !(*this == o); }
};

template <typename Ti>
struct PauliHash
{
    size_t operator()(const Pauli<Ti> &p) const noexcept
    {
        return mix_hash(fold_for_hash(p.x) ^ fold_for_hash(p.z));
    }
};

template <typename Tv, typename Tg>
struct HamResult
{
    void *axs;  // xs 的 α 部分, half_width(Ti) 指针 (对应 Julia 的 UInt64/UInt128)
    void *azs;  // zs 的 α 部分, half_width(Ti) 指针 (对应 Julia 的 UInt64/UInt128)
    void *bxs;  // xs 的 β 部分, half_width(Ti) 指针 (对应 Julia 的 UInt64/UInt128)
    void *bzs;  // zs 的 β 部分, half_width(Ti) 指针 (对应 Julia 的 UInt64/UInt128)
    Tv *cs;     // Float64/ComplexF64
    Tg *gs;     // xs(axs.|bxs)的分组边界 (1-based), [gs[i], gs[i+1]-1]中是同一个 x
    size_t ncs; // cs 长度
    size_t ngs; // gs 长度
};

template <typename Tv>
struct SingleTerm
{
    int p, q;
    Tv val;
};

template <typename Tv>
struct DoubleTerm
{
    int p, q, r, s;
    Tv val;
};

template <typename Ti, typename Tv>
struct PauliTerm
{
    Pauli<Ti> q;
    Tv c;
    bool operator<(const PauliTerm &o) const { return q < o.q; }
};

template <typename Ti, typename Tv>
using FastDict = ankerl::unordered_dense::map<Pauli<Ti>, Tv, PauliHash<Ti>>;

template <typename Tv>
using LocalSingleArray = std::vector<std::vector<std::vector<SingleTerm<Tv>>>>;

template <typename Tv>
using LocalDoubleArray = std::vector<std::vector<std::vector<DoubleTerm<Tv>>>>;

template <typename Ti, typename Tv>
using DictArray = std::vector<FastDict<Ti, Tv>>;

template <typename Ti, typename Tv>
using TermArray = std::vector<PauliTerm<Ti, Tv>>;

template <typename Ti, typename Tv>
FORCE_INLINE void insert_1body(
    FastDict<Ti, Tv> *__restrict dict,
    int p, int q,
    Tv coeff)
{
    Ti o = get_one<Ti>();
    Ti x1 = o << p, z11 = x1 - o, z12 = (x1 << 1) - o;
    Ti x2 = o << q, z21 = x2 - o, z22 = (x2 << 1) - o;
    Ti x12 = x1 ^ x2;
    Tv cc = 0.25 * coeff;

    Tv c1 = cc * (1.0 - 2.0 * (popcnt(z11 & x2) & 1));
    Tv c2 = cc * (1.0 - 2.0 * (popcnt(z12 & x2) & 1));

    Pauli<Ti> k;
    k.x = x12;
    k.z = z11 ^ z21;
    (*dict)[k] += c1;
    k.z = z11 ^ z22;
    (*dict)[k] -= c1;
    k.z = z12 ^ z21;
    (*dict)[k] += c2;
    k.z = z12 ^ z22;
    (*dict)[k] -= c2;
}

template <typename Ti, typename Tv>
FORCE_INLINE void insert_2body(
    FastDict<Ti, Tv> *__restrict dict,
    int p, int q, int r, int s,
    Tv coeff)
{
    Ti o = get_one<Ti>();
    Ti x1 = o << p, z11 = x1 - o, z12 = (x1 << 1) - o;
    Ti x2 = o << q, z21 = x2 - o, z22 = (x2 << 1) - o;
    Ti x3 = o << r, z31 = x3 - o, z32 = (x3 << 1) - o;
    Ti x4 = o << s, z41 = x4 - o, z42 = (x4 << 1) - o;
    Ti x34 = x3 ^ x4, x234 = x2 ^ x3 ^ x4, x1234 = x1 ^ x2 ^ x3 ^ x4;

    int p11_x234 = popcnt(z11 & x234) & 1;
    int p12_x234 = popcnt(z12 & x234) & 1;
    int p21_x34 = popcnt(z21 & x34) & 1;
    int p22_x34 = popcnt(z22 & x34) & 1;
    int p31_x4 = popcnt(z31 & x4) & 1;
    int p32_x4 = popcnt(z32 & x4) & 1;

    Tv cc = 0.0625 * coeff;

    int p1 = p11_x234 ^ p21_x34;
    int p2 = p11_x234 ^ p22_x34;
    int p3 = p12_x234 ^ p21_x34;
    int p4 = p12_x234 ^ p22_x34;

    Tv c11 = cc * (1.0 - 2.0 * (p1 ^ p31_x4)), c12 = cc * (1.0 - 2.0 * (p1 ^ p32_x4));
    Tv c21 = cc * (1.0 - 2.0 * (p2 ^ p31_x4)), c22 = cc * (1.0 - 2.0 * (p2 ^ p32_x4));
    Tv c31 = cc * (1.0 - 2.0 * (p3 ^ p31_x4)), c32 = cc * (1.0 - 2.0 * (p3 ^ p32_x4));
    Tv c41 = cc * (1.0 - 2.0 * (p4 ^ p31_x4)), c42 = cc * (1.0 - 2.0 * (p4 ^ p32_x4));

    Pauli<Ti> k;
    k.x = x1234;
    k.z = z11 ^ z21 ^ z31 ^ z41;
    (*dict)[k] += c11;
    k.z = z11 ^ z21 ^ z31 ^ z42;
    (*dict)[k] -= c11;
    k.z = z11 ^ z21 ^ z32 ^ z41;
    (*dict)[k] -= c12;
    k.z = z11 ^ z21 ^ z32 ^ z42;
    (*dict)[k] += c12;
    k.z = z11 ^ z22 ^ z31 ^ z41;
    (*dict)[k] += c21;
    k.z = z11 ^ z22 ^ z31 ^ z42;
    (*dict)[k] -= c21;
    k.z = z11 ^ z22 ^ z32 ^ z41;
    (*dict)[k] -= c22;
    k.z = z11 ^ z22 ^ z32 ^ z42;
    (*dict)[k] += c22;
    k.z = z12 ^ z21 ^ z31 ^ z41;
    (*dict)[k] += c31;
    k.z = z12 ^ z21 ^ z31 ^ z42;
    (*dict)[k] -= c31;
    k.z = z12 ^ z21 ^ z32 ^ z41;
    (*dict)[k] -= c32;
    k.z = z12 ^ z21 ^ z32 ^ z42;
    (*dict)[k] += c32;
    k.z = z12 ^ z22 ^ z31 ^ z41;
    (*dict)[k] += c41;
    k.z = z12 ^ z22 ^ z31 ^ z42;
    (*dict)[k] -= c41;
    k.z = z12 ^ z22 ^ z32 ^ z41;
    (*dict)[k] -= c42;
    k.z = z12 ^ z22 ^ z32 ^ z42;
    (*dict)[k] += c42;
}

template <typename Ti, typename Tv>
void stage1_allocate_and_scan(
    int nthreads, int nblocks, int norbs, double tol,
    const Tv *one_body_mo, const Tv *two_body_mo,
    LocalSingleArray<Tv> *local_single,
    LocalDoubleArray<Tv> *local_double)
{
    local_single->assign(nthreads, std::vector<std::vector<SingleTerm<Tv>>>(nblocks));
    local_double->assign(nthreads, std::vector<std::vector<DoubleTerm<Tv>>>(nblocks));

    size_t N1 = static_cast<size_t>(norbs);
    size_t N2 = N1 * N1;
    size_t N3 = N2 * N1;
    size_t N4 = N3 * N1;

    size_t revsize = (N4 / (nthreads * nblocks)) * 0.1 + 10;

    for (int t = 0; t < nthreads; ++t)
    {
        for (int b = 0; b < nblocks; ++b)
        {
            (*local_double)[t][b].reserve(revsize);
        }
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Ti ONE = get_one<Ti>();
#pragma omp for schedule(static) collapse(2)
        for (int q = 0; q < norbs; ++q)
        {
            for (int p = 0; p < norbs; ++p)
            {
                size_t idx = static_cast<size_t>(p) +
                             static_cast<size_t>(q) * N1;

                Tv val = one_body_mo[idx];

                if (std::abs(val) > tol)
                {
                    Ti mask = (ONE << p) ^ (ONE << q);
                    uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                    (*local_single)[tid][bucket_idx].push_back({p, q, val});
                }
            }
        }
#pragma omp for schedule(static) collapse(4)
        for (int s = 0; s < norbs; ++s)
        {
            for (int r = 0; r < norbs; ++r)
            {
                for (int q = 0; q < norbs; ++q)
                {
                    for (int p = 0; p < norbs; ++p)
                    {
                        size_t idx = static_cast<size_t>(p) +
                                     static_cast<size_t>(q) * N1 +
                                     static_cast<size_t>(r) * N2 +
                                     static_cast<size_t>(s) * N3;

                        Tv val = two_body_mo[idx];

                        if (std::abs(val) > tol)
                        {
                            Ti mask = (ONE << p) ^ (ONE << q) ^
                                      (ONE << r) ^ (ONE << s);
                            uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                            (*local_double)[tid][bucket_idx].push_back({p, q, r, s, val});
                        }
                    }
                }
            }
        }
    }
}

template <typename Ti, typename Tv>
void stage1_allocate_and_scan_twopass(
    int nthreads, int nblocks, int norbs, double tol,
    const Tv *one_body_mo, const Tv *two_body_mo,
    LocalSingleArray<Tv> *local_single,
    LocalDoubleArray<Tv> *local_double)
{
    size_t N1 = static_cast<size_t>(norbs);
    size_t N2 = N1 * N1;
    size_t N3 = N2 * N1;
    size_t N4 = N3 * N1;

    std::vector<std::vector<size_t>> count_single(nthreads, std::vector<size_t>(nblocks, 0));
    std::vector<std::vector<size_t>> count_double(nthreads, std::vector<size_t>(nblocks, 0));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Ti ONE = get_one<Ti>();
#pragma omp for schedule(static) collapse(2)
        for (int q = 0; q < norbs; ++q)
        {
            for (int p = 0; p < norbs; ++p)
            {
                size_t idx = static_cast<size_t>(p) +
                             static_cast<size_t>(q) * N1;
                if (std::abs(one_body_mo[idx]) > tol)
                {
                    Ti mask = (ONE << p) ^ (ONE << q);
                    uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                    count_single[tid][bucket_idx]++;
                }
            }
        }
#pragma omp for schedule(static) collapse(4)
        for (int s = 0; s < norbs; ++s)
        {
            for (int r = 0; r < norbs; ++r)
            {
                for (int q = 0; q < norbs; ++q)
                {
                    for (int p = 0; p < norbs; ++p)
                    {
                        size_t idx = static_cast<size_t>(p) +
                                     static_cast<size_t>(q) * N1 +
                                     static_cast<size_t>(r) * N2 + static_cast<size_t>(s) * N3;
                        if (std::abs(two_body_mo[idx]) > tol)
                        {
                            Ti mask = (ONE << p) ^ (ONE << q) ^
                                      (ONE << r) ^ (ONE << s);
                            uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                            count_double[tid][bucket_idx]++;
                        }
                    }
                }
            }
        }
    }

    local_single->assign(nthreads, std::vector<std::vector<SingleTerm<Tv>>>(nblocks));
    local_double->assign(nthreads, std::vector<std::vector<DoubleTerm<Tv>>>(nblocks));

    for (int t = 0; t < nthreads; ++t)
    {
        for (int b = 0; b < nblocks; ++b)
        {
            (*local_single)[t][b].reserve(count_single[t][b]);
            (*local_double)[t][b].reserve(count_double[t][b]);
        }
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Ti ONE = get_one<Ti>();
#pragma omp for schedule(static) collapse(2)
        for (int q = 0; q < norbs; ++q)
        {
            for (int p = 0; p < norbs; ++p)
            {
                size_t idx = static_cast<size_t>(p) +
                             static_cast<size_t>(q) * N1;

                Tv val = one_body_mo[idx];

                if (std::abs(val) > tol)
                {
                    Ti mask = (ONE << p) ^ (ONE << q);
                    uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                    (*local_single)[tid][bucket_idx].push_back({p, q, val});
                }
            }
        }
#pragma omp for schedule(static) collapse(4)
        for (int s = 0; s < norbs; ++s)
        {
            for (int r = 0; r < norbs; ++r)
            {
                for (int q = 0; q < norbs; ++q)
                {
                    for (int p = 0; p < norbs; ++p)
                    {
                        size_t idx = static_cast<size_t>(p) +
                                     static_cast<size_t>(q) * N1 +
                                     static_cast<size_t>(r) * N2 + static_cast<size_t>(s) * N3;

                        Tv val = two_body_mo[idx];

                        if (std::abs(val) > tol)
                        {
                            Ti mask = (ONE << p) ^ (ONE << q) ^
                                      (ONE << r) ^ (ONE << s);
                            uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                            (*local_double)[tid][bucket_idx].push_back(
                                {p, q, r, s, val});
                        }
                    }
                }
            }
        }
    }
}

template <typename Ti, typename Tv>
void stage2_reduce_dictionaries(
    int nthreads, int nblocks, Tv energy_nuc, double tol,
    const LocalSingleArray<Tv> *local_single,
    const LocalDoubleArray<Tv> *local_double,
    DictArray<Ti, Tv> *dicts)
{
    dicts->resize(nblocks);

#pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nblocks; ++b)
    {
        size_t raw_inserts = 0;
        for (int t = 0; t < nthreads; ++t)
        {
            raw_inserts += (*local_single)[t][b].size() * 4 +
                           (*local_double)[t][b].size() * 16;
        }

        (*dicts)[b].reserve(raw_inserts / 2 + 100);
        FastDict<Ti, Tv> *__restrict dict_ptr = &(*dicts)[b];

        for (int t = 0; t < nthreads; ++t)
        {
            const SingleTerm<Tv> *s_ptr = (*local_single)[t][b].data();
            for (size_t i = 0, s_len = (*local_single)[t][b].size(); i < s_len; ++i)
            {
                Tv ci = s_ptr[i].val;
                insert_1body<Ti, Tv>(
                    dict_ptr,
                    2 * s_ptr[i].p,
                    2 * s_ptr[i].q,
                    ci);
                insert_1body<Ti, Tv>(
                    dict_ptr,
                    2 * s_ptr[i].p + 1,
                    2 * s_ptr[i].q + 1,
                    ci);
            }
            const DoubleTerm<Tv> *d_ptr = (*local_double)[t][b].data();
            for (size_t i = 0, d_len = (*local_double)[t][b].size(); i < d_len; ++i)
            {
                Tv ci = d_ptr[i].val * 0.5;
                insert_2body<Ti, Tv>(
                    dict_ptr,
                    2 * d_ptr[i].p,
                    2 * d_ptr[i].q,
                    2 * d_ptr[i].r,
                    2 * d_ptr[i].s,
                    ci);
                insert_2body<Ti, Tv>(
                    dict_ptr,
                    2 * d_ptr[i].p + 1,
                    2 * d_ptr[i].q + 1,
                    2 * d_ptr[i].r + 1,
                    2 * d_ptr[i].s + 1,
                    ci);
                insert_2body<Ti, Tv>(
                    dict_ptr,
                    2 * d_ptr[i].p,
                    2 * d_ptr[i].q + 1,
                    2 * d_ptr[i].r + 1,
                    2 * d_ptr[i].s,
                    ci);
                insert_2body<Ti, Tv>(
                    dict_ptr,
                    2 * d_ptr[i].p + 1,
                    2 * d_ptr[i].q,
                    2 * d_ptr[i].r,
                    2 * d_ptr[i].s + 1,
                    ci);
            }
        }
    }

    if (std::abs(energy_nuc) > tol)
    {
        Pauli<Ti> k_zero = {get_zero<Ti>(), get_zero<Ti>()};
        (*dicts)[0][k_zero] += energy_nuc;
    }
}

template <typename Ti, typename Tv>
void stage3_count_and_write(
    int nblocks, double tol,
    const DictArray<Ti, Tv> *dicts,
    TermArray<Ti, Tv> *all_terms,
    size_t *out_nqs)
{
    std::vector<size_t> bucket_counts(nblocks, 0);

#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblocks; ++b)
    {
        for (const auto &kv : (*dicts)[b])
        {
            if (std::abs(kv.second) > tol)
            {
                bucket_counts[b]++;
            }
        }
    }

    std::vector<size_t> offsets(nblocks + 1, 0);
    for (int b = 0; b < nblocks; ++b)
    {
        offsets[b + 1] = offsets[b] + bucket_counts[b];
    }

    size_t nqs = offsets.back();
    *out_nqs = nqs;
    all_terms->resize(nqs);

#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblocks; ++b)
    {
        size_t idx = offsets[b];
        for (const auto &kv : (*dicts)[b])
        {
            if (std::abs(kv.second) > tol)
            {
                (*all_terms)[idx++] = {kv.first, kv.second};
            }
        }
    }
}

template <typename Ti,
          typename Tv,
          typename Tg>
HamResult<Tv, Tg> stage5_prepare_output(
    size_t nqs,
    const TermArray<Ti, Tv> *all_terms,
    int based)
{
    size_t alloc_nnz = std::max<size_t>(1, nqs);

    using Th = half_width_t<Ti>;

    Tv *final_cs = (Tv *)malloc(alloc_nnz * sizeof(Tv));
    Th *final_axs = (Th *)malloc(alloc_nnz * sizeof(Th));
    Th *final_azs = (Th *)malloc(alloc_nnz * sizeof(Th));
    Th *final_bxs = (Th *)malloc(alloc_nnz * sizeof(Th));
    Th *final_bzs = (Th *)malloc(alloc_nnz * sizeof(Th));

    std::vector<Tg> bounds;
    bounds.push_back(static_cast<Tg>(based));

    if (nqs > 0)
    {
        Ti last_x = (*all_terms)[0].q.x;
        for (size_t i = 0; i < nqs; ++i)
        {
            Pauli<Ti> q = (*all_terms)[i].q;
            if (i > 0 && q.x != last_x)
            {
                bounds.push_back(static_cast<Tg>(i + based));
                last_x = q.x;
            }
        }
        bounds.push_back(static_cast<Tg>(nqs + based));
    }
    else
    {
        bounds.push_back(static_cast<Tg>(based));
    }

    size_t ngs = bounds.size();
    Tg *final_gs = (Tg *)malloc(std::max<size_t>(1, ngs) * sizeof(Tg));
    std::copy(bounds.begin(), bounds.end(), final_gs);

    if (nqs > 0)
    {
#pragma omp parallel for
        for (size_t i = 0; i < nqs; ++i)
        {
            Pauli<Ti> q = (*all_terms)[i].q;

            final_cs[i] = (*all_terms)[i].c;
            final_axs[i] = zip_even_bit_bmi2(q.x);
            final_azs[i] = zip_even_bit_bmi2(q.z);
            final_bxs[i] = zip_odd_bit_bmi2(q.x);
            final_bzs[i] = zip_odd_bit_bmi2(q.z);
        }
    }

    return {
        (void *)final_axs,
        (void *)final_azs,
        (void *)final_bxs,
        (void *)final_bzs,
        final_cs,
        final_gs,
        nqs,
        ngs};
}

template <typename Ti, typename Tv, typename Tg>
HamResult<Tv, Tg> generate_hamiltonian_tmpl(
    Tv energy_nuc,
    const Tv *one_body_mo,
    const Tv *two_body_mo,
    int norbs,
    double tol,
    int based,
    bool verbose)
{
    int nthreads = omp_get_max_threads();
    int nblocks = nthreads * 64;

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Stage 1 ---
    LocalSingleArray<Tv> local_single;
    LocalDoubleArray<Tv> local_double;
    stage1_allocate_and_scan<Ti, Tv>(
        nthreads, nblocks, norbs, tol, one_body_mo, two_body_mo,
        &local_single, &local_double);

    auto t1 = std::chrono::high_resolution_clock::now();

    // --- Stage 2 ---
    DictArray<Ti, Tv> dicts;
    stage2_reduce_dictionaries<Ti, Tv>(
        nthreads, nblocks, energy_nuc, tol,
        &local_single, &local_double, &dicts);

    local_single.clear();
    local_single.shrink_to_fit();
    local_double.clear();
    local_double.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();

    // --- Stage 3 ---
    TermArray<Ti, Tv> all_terms;
    size_t nqs = 0;
    stage3_count_and_write<Ti, Tv>(nblocks, tol, &dicts, &all_terms, &nqs);

    dicts.clear();
    dicts.shrink_to_fit();

    auto t3 = std::chrono::high_resolution_clock::now();

    // --- Stage 4 ---
    __gnu_parallel::sort(all_terms.begin(), all_terms.end());

    auto t4 = std::chrono::high_resolution_clock::now();

    // --- Stage 5 ---
    HamResult<Tv, Tg> res = stage5_prepare_output<Ti, Tv, Tg>(nqs, &all_terms, based);

    auto t5 = std::chrono::high_resolution_clock::now();

    if (verbose)
    {
        std::chrono::duration<double> t_scan = t1 - t0;
        std::chrono::duration<double> t_reduce = t2 - t1;
        std::chrono::duration<double> t_write = t3 - t2;
        std::chrono::duration<double> t_sort = t4 - t3;
        std::chrono::duration<double> t_output = t5 - t4;
        std::chrono::duration<double> t_total = t5 - t0;

        printf("\n");
        printf("[Summary] Orbital number:       %d\n", norbs);
        printf("[Summary] Number of threads:    %d\n", nthreads);
        printf("[Summary] Bucket size:          %d\n", nblocks);
        printf("[Summary] Number of gs:         %zu\n", res.ngs);
        printf("[Summary] Number of cs:         %zu\n", res.ncs);

        printf("[Time] Total:         %.4f seconds\n", t_total.count());
        printf("[Time] Scanning:      %.4f seconds\n", t_scan.count());
        printf("[Time] Reduction:     %.4f seconds\n", t_reduce.count());
        printf("[Time] Writing:       %.4f seconds\n", t_write.count());
        printf("[Time] Sorting:       %.4f seconds\n", t_sort.count());
        printf("[Time] Output:        %.4f seconds\n", t_output.count());
    }

    return res;
}

extern "C"
{
    using complex64 = std::complex<double>;

    // 处理最高 (31o, 62q) 的分子体系
    HamResult<double, int64_t> generate_hamiltonian_64_128_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 31)
        {
            std::cerr << "Error: norbs > 31 not supported for uint64 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint64, double, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }

    // 处理最高 (31o, 62q) 的周期性体系
    HamResult<complex64, int64_t> generate_hamiltonian_64_128_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 31)
        {
            std::cerr << "Error: norbs > 31 not supported for uint64 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint64, complex64, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }

    // 处理最高 (63o, 126q) 的分子体系
    HamResult<double, int64_t> generate_hamiltonian_128_256_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 63)
        {
            std::cerr << "Error: norbs > 63 not supported for uint128 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint128, double, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }

    // 处理最高 (63o, 126q) 的周期性体系
    HamResult<complex64, int64_t> generate_hamiltonian_128_256_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 63)
        {
            std::cerr << "Error: norbs > 63 not supported for uint128 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint128, complex64, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }

    // 处理最高 (127o, 254q) 的分子体系
    HamResult<double, int64_t> generate_hamiltonian_256_512_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 127)
        {
            std::cerr << "Error: norbs > 127 not supported for uint256 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint256, double, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }

    // 处理最高 (127o, 254q) 的周期性体系
    HamResult<complex64, int64_t> generate_hamiltonian_256_512_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        int based,
        bool verbose)
    {
        if (norbs > 127)
        {
            std::cerr << "Error: norbs > 127 not supported for uint256 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
        }
        return generate_hamiltonian_tmpl<uint256, complex64, int64_t>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, based, verbose);
    }
}
