#pragma once

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
#include <limits>
#include <parallel/algorithm>

#include <ankerl/unordered_dense.h>
#include "bitintegers.hpp"

namespace binsim::ham
{

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



    template <typename Th>
    struct LocalPauli
    {
        Th x;
        Th z;

        bool operator<(const LocalPauli &o) const
        {
            if (x != o.x)
                return x < o.x;
            return z < o.z;
        }

        bool operator==(const LocalPauli &o) const { return x == o.x && z == o.z; }

        bool operator!=(const LocalPauli &o) const { return !(*this == o); }
    };

    template <typename Th>
    struct LocalPauliHash
    {
        size_t operator()(const LocalPauli<Th> &p) const noexcept
        {
            return mix_hash(fold_for_hash(p.x) ^ fold_for_hash(p.z));
        }
    };

    template <typename Th, typename Tv>
    using LocalPauliDict = ankerl::unordered_dense::map<LocalPauli<Th>, Tv, LocalPauliHash<Th>>;

    template <typename Th, typename Tv>
    struct LocalPairTerm
    {
        LocalPauli<Th> alpha;
        LocalPauli<Th> beta;
        Tv c;
    };

    template <typename Th>
    struct LocalPauliPair
    {
        LocalPauli<Th> alpha;
        LocalPauli<Th> beta;

        bool operator==(const LocalPauliPair &o) const
        {
            return alpha == o.alpha && beta == o.beta;
        }
    };

    template <typename Th>
    struct LocalPauliPairHash
    {
        size_t operator()(const LocalPauliPair<Th> &p) const noexcept
        {
            size_t ah = LocalPauliHash<Th>{}(p.alpha);
            size_t bh = LocalPauliHash<Th>{}(p.beta);
            return mix_hash(ah ^ (bh + 0x9e3779b97f4a7c15ULL + (ah << 6) + (ah >> 2)));
        }
    };

    template <typename Th, typename Tv>
    using LocalPairDict = ankerl::unordered_dense::map<LocalPauliPair<Th>, Tv, LocalPauliPairHash<Th>>;

    template <typename Th, typename Tv>
    struct DenseLocalBlock
    {
        std::vector<LocalPauli<Th>> alpha_terms;
        std::vector<LocalPauli<Th>> beta_terms;
        ankerl::unordered_dense::map<LocalPauli<Th>, uint32_t, LocalPauliHash<Th>> alpha_ids;
        ankerl::unordered_dense::map<LocalPauli<Th>, uint32_t, LocalPauliHash<Th>> beta_ids;
        std::vector<Tv> coeff_matrix;
        LocalPairDict<Th, Tv> sparse_pairs;
        size_t nbeta = 0;
        bool dense = true;
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

    template <typename Tv>
    FORCE_INLINE Tv parity_coeff(Tv cc, int parity)
    {
        return parity ? -cc : cc;
    }

    template <typename Ti, typename Tv>
    FORCE_INLINE void insert_1body(FastDict<Ti, Tv> *__restrict dict, int p, int q, Tv coeff)
    {
        Ti o = get_one<Ti>();
        Ti x1 = o << p, z11 = x1 - o, z12 = (x1 << 1) - o;
        Ti x2 = o << q, z21 = x2 - o, z22 = (x2 << 1) - o;
        Ti x12 = x1 ^ x2;
        Tv cc = 0.25 * coeff;

        Tv c1 = parity_coeff(cc, popcnt(z11 & x2) & 1);
        Tv c2 = parity_coeff(cc, popcnt(z12 & x2) & 1);

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
    FORCE_INLINE void insert_2body(FastDict<Ti, Tv> *__restrict dict, int p, int q, int r, int s, Tv coeff)
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

        Tv c11 = parity_coeff(cc, p1 ^ p31_x4);
        Tv c21 = parity_coeff(cc, p2 ^ p31_x4);
        Tv c31 = parity_coeff(cc, p3 ^ p31_x4);
        Tv c41 = parity_coeff(cc, p4 ^ p31_x4);
        Tv c12 = parity_coeff(cc, p1 ^ p32_x4);
        Tv c22 = parity_coeff(cc, p2 ^ p32_x4);
        Tv c32 = parity_coeff(cc, p3 ^ p32_x4);
        Tv c42 = parity_coeff(cc, p4 ^ p32_x4);

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


    template <typename Ti, typename Fn>
    FORCE_INLINE void for_each_1body_pauli(int p, int q, Fn &&fn)
    {
        Ti o = get_one<Ti>();
        Ti x1 = o << p, z11 = x1 - o, z12 = (x1 << 1) - o;
        Ti x2 = o << q, z21 = x2 - o, z22 = (x2 << 1) - o;
        Ti x12 = x1 ^ x2;
        int c1 = parity_coeff(1, popcnt(z11 & x2) & 1);
        int c2 = parity_coeff(1, popcnt(z12 & x2) & 1);
        Pauli<Ti> k;
        k.x = x12;
        k.z = z11 ^ z21;
        fn(k, c1);
        k.z = z11 ^ z22;
        fn(k, -c1);
        k.z = z12 ^ z21;
        fn(k, c2);
        k.z = z12 ^ z22;
        fn(k, -c2);
    }

    template <typename Ti, typename Fn>
    FORCE_INLINE void for_each_2body_pauli(int p, int q, int r, int s, Fn &&fn)
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

        int p1 = p11_x234 ^ p21_x34;
        int p2 = p11_x234 ^ p22_x34;
        int p3 = p12_x234 ^ p21_x34;
        int p4 = p12_x234 ^ p22_x34;
        int signs[16] = {
            parity_coeff(1, p1 ^ p31_x4), -parity_coeff(1, p1 ^ p31_x4),
            -parity_coeff(1, p1 ^ p32_x4), parity_coeff(1, p1 ^ p32_x4),
            parity_coeff(1, p2 ^ p31_x4), -parity_coeff(1, p2 ^ p31_x4),
            -parity_coeff(1, p2 ^ p32_x4), parity_coeff(1, p2 ^ p32_x4),
            parity_coeff(1, p3 ^ p31_x4), -parity_coeff(1, p3 ^ p31_x4),
            -parity_coeff(1, p3 ^ p32_x4), parity_coeff(1, p3 ^ p32_x4),
            parity_coeff(1, p4 ^ p31_x4), -parity_coeff(1, p4 ^ p31_x4),
            -parity_coeff(1, p4 ^ p32_x4), parity_coeff(1, p4 ^ p32_x4)};
        Ti zs[16] = {
            z11 ^ z21 ^ z31 ^ z41, z11 ^ z21 ^ z31 ^ z42,
            z11 ^ z21 ^ z32 ^ z41, z11 ^ z21 ^ z32 ^ z42,
            z11 ^ z22 ^ z31 ^ z41, z11 ^ z22 ^ z31 ^ z42,
            z11 ^ z22 ^ z32 ^ z41, z11 ^ z22 ^ z32 ^ z42,
            z12 ^ z21 ^ z31 ^ z41, z12 ^ z21 ^ z31 ^ z42,
            z12 ^ z21 ^ z32 ^ z41, z12 ^ z21 ^ z32 ^ z42,
            z12 ^ z22 ^ z31 ^ z41, z12 ^ z22 ^ z31 ^ z42,
            z12 ^ z22 ^ z32 ^ z41, z12 ^ z22 ^ z32 ^ z42};
        Pauli<Ti> k;
        k.x = x1234;
        for (int i = 0; i < 16; ++i)
        {
            k.z = zs[i];
            fn(k, signs[i]);
        }
    }

    template <typename Ti>
    FORCE_INLINE LocalPauli<half_width_t<Ti>> local_alpha_pauli(const Pauli<Ti> &q)
    {
        using Th = half_width_t<Ti>;
        return LocalPauli<Th>{static_cast<Th>(zip_even_bit_bmi2(q.x)), static_cast<Th>(zip_even_bit_bmi2(q.z))};
    }

    template <typename Ti>
    FORCE_INLINE LocalPauli<half_width_t<Ti>> local_beta_pauli(const Pauli<Ti> &q)
    {
        using Th = half_width_t<Ti>;
        return LocalPauli<Th>{static_cast<Th>(zip_odd_bit_bmi2(q.x)), static_cast<Th>(zip_odd_bit_bmi2(q.z))};
    }

    template <typename Th>
    uint32_t dense_local_id(ankerl::unordered_dense::map<LocalPauli<Th>, uint32_t, LocalPauliHash<Th>> *ids,
                            std::vector<LocalPauli<Th>> *terms,
                            const LocalPauli<Th> &p)
    {
        auto [it, inserted] = ids->try_emplace(p, static_cast<uint32_t>(terms->size()));
        if (inserted)
            terms->push_back(p);
        return it->second;
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
        size_t *out_ncs)
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

        size_t ncs = offsets.back();
        *out_ncs = ncs;
        all_terms->resize(ncs);

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


    template <typename Tv>
    struct HamResult;

    inline size_t dense_local_bucket_limit_bytes()
    {
        const char *env = std::getenv("BINSIM_DENSE_LOCAL_BUCKET_BYTES");
        if (env != nullptr)
        {
            char *end = nullptr;
            unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && parsed > 0)
                return static_cast<size_t>(parsed);
        }
        return 64ULL * 1024ULL * 1024ULL;
    }

    template <typename Ti, typename Tv, typename Fn>
    void scan_bucket_pauli_terms(int nthreads, int b,
                                 const LocalSingleArray<Tv> *local_single,
                                 const LocalDoubleArray<Tv> *local_double,
                                 Fn &&fn)
    {
        for (int t = 0; t < nthreads; ++t)
        {
            const SingleTerm<Tv> *s_ptr = (*local_single)[t][b].data();
            for (size_t i = 0, s_len = (*local_single)[t][b].size(); i < s_len; ++i)
            {
                Tv ci = s_ptr[i].val;
                auto emit = [&](const Pauli<Ti> &q, int sign)
                { fn(q, static_cast<Tv>(0.25 * ci * sign)); };
                for_each_1body_pauli<Ti>(2 * s_ptr[i].p, 2 * s_ptr[i].q, emit);
                for_each_1body_pauli<Ti>(2 * s_ptr[i].p + 1, 2 * s_ptr[i].q + 1, emit);
            }
            const DoubleTerm<Tv> *d_ptr = (*local_double)[t][b].data();
            for (size_t i = 0, d_len = (*local_double)[t][b].size(); i < d_len; ++i)
            {
                Tv ci = d_ptr[i].val * 0.5;
                auto emit = [&](const Pauli<Ti> &q, int sign)
                { fn(q, static_cast<Tv>(0.0625 * ci * sign)); };
                for_each_2body_pauli<Ti>(2 * d_ptr[i].p, 2 * d_ptr[i].q, 2 * d_ptr[i].r, 2 * d_ptr[i].s, emit);
                for_each_2body_pauli<Ti>(2 * d_ptr[i].p + 1, 2 * d_ptr[i].q + 1, 2 * d_ptr[i].r + 1, 2 * d_ptr[i].s + 1, emit);
                for_each_2body_pauli<Ti>(2 * d_ptr[i].p, 2 * d_ptr[i].q + 1, 2 * d_ptr[i].r + 1, 2 * d_ptr[i].s, emit);
                for_each_2body_pauli<Ti>(2 * d_ptr[i].p + 1, 2 * d_ptr[i].q, 2 * d_ptr[i].r, 2 * d_ptr[i].s + 1, emit);
            }
        }
    }

    template <typename Ti, typename Tv>
    void stage2_reduce_dense_local(int nthreads, int nblocks, Tv energy_nuc, double tol,
                                   const LocalSingleArray<Tv> *local_single,
                                   const LocalDoubleArray<Tv> *local_double,
                                   std::vector<DenseLocalBlock<half_width_t<Ti>, Tv>> *blocks)
    {
        using Th = half_width_t<Ti>;
        blocks->resize(nblocks);
        const size_t bucket_limit = dense_local_bucket_limit_bytes();

#pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < nblocks; ++b)
        {
            auto &block = (*blocks)[b];
            LocalPauli<Th> zero{get_zero<Th>(), get_zero<Th>()};
            dense_local_id(&block.alpha_ids, &block.alpha_terms, zero);
            dense_local_id(&block.beta_ids, &block.beta_terms, zero);

            scan_bucket_pauli_terms<Ti, Tv>(nthreads, b, local_single, local_double,
                                            [&](const Pauli<Ti> &q, Tv) {
                                                dense_local_id(&block.alpha_ids, &block.alpha_terms, local_alpha_pauli(q));
                                                dense_local_id(&block.beta_ids, &block.beta_terms, local_beta_pauli(q));
                                            });

            size_t nalpha = block.alpha_terms.size();
            block.nbeta = block.beta_terms.size();
            bool overflow = block.nbeta != 0 && nalpha > std::numeric_limits<size_t>::max() / block.nbeta;
            size_t entries = overflow ? std::numeric_limits<size_t>::max() : nalpha * block.nbeta;
            overflow = overflow || entries > std::numeric_limits<size_t>::max() / sizeof(Tv);
            size_t dense_bytes = overflow ? std::numeric_limits<size_t>::max() : entries * sizeof(Tv);
            block.dense = !overflow && dense_bytes <= bucket_limit;

            if (block.dense)
                block.coeff_matrix.assign(entries, Tv{});
            else
                block.sparse_pairs.reserve(nalpha + block.nbeta + 100);

            scan_bucket_pauli_terms<Ti, Tv>(nthreads, b, local_single, local_double,
                                            [&](const Pauli<Ti> &q, Tv coeff) {
                                                LocalPauli<Th> alpha = local_alpha_pauli(q);
                                                LocalPauli<Th> beta = local_beta_pauli(q);
                                                uint32_t aid = block.alpha_ids[alpha];
                                                uint32_t bid = block.beta_ids[beta];
                                                if (block.dense)
                                                    block.coeff_matrix[static_cast<size_t>(aid) * block.nbeta + bid] += coeff;
                                                else
                                                    block.sparse_pairs[LocalPauliPair<Th>{alpha, beta}] += coeff;
                                            });
        }

        if (std::abs(energy_nuc) > tol)
        {
            auto &block = (*blocks)[0];
            LocalPauli<Th> zero{get_zero<Th>(), get_zero<Th>()};
            uint32_t aid = dense_local_id(&block.alpha_ids, &block.alpha_terms, zero);
            uint32_t bid = dense_local_id(&block.beta_ids, &block.beta_terms, zero);
            if (block.dense && !block.coeff_matrix.empty())
                block.coeff_matrix[static_cast<size_t>(aid) * block.nbeta + bid] += energy_nuc;
            else
                block.sparse_pairs[LocalPauliPair<Th>{zero, zero}] += energy_nuc;
        }
    }

    template <typename Th, typename Tv>
    void stage3_dense_local_count_and_write(int nblocks, double tol,
                                            const std::vector<DenseLocalBlock<Th, Tv>> *blocks,
                                            std::vector<LocalPairTerm<Th, Tv>> *all_terms,
                                            size_t *out_ncs)
    {
        std::vector<size_t> bucket_counts(nblocks, 0);
#pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; ++b)
        {
            const auto &block = (*blocks)[b];
            if (block.dense)
            {
                for (const Tv &coeff : block.coeff_matrix)
                    if (std::abs(coeff) > tol)
                        bucket_counts[b]++;
            }
            else
            {
                for (const auto &kv : block.sparse_pairs)
                    if (std::abs(kv.second) > tol)
                        bucket_counts[b]++;
            }
        }
        std::vector<size_t> offsets(nblocks + 1, 0);
        for (int b = 0; b < nblocks; ++b)
            offsets[b + 1] = offsets[b] + bucket_counts[b];
        *out_ncs = offsets.back();
        all_terms->resize(*out_ncs);
#pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; ++b)
        {
            const auto &block = (*blocks)[b];
            size_t idx = offsets[b];
            if (block.dense)
            {
                for (size_t aid = 0; aid < block.alpha_terms.size(); ++aid)
                    for (size_t bid = 0; bid < block.nbeta; ++bid)
                    {
                        Tv coeff = block.coeff_matrix[aid * block.nbeta + bid];
                        if (std::abs(coeff) > tol)
                            (*all_terms)[idx++] = {block.alpha_terms[aid], block.beta_terms[bid], coeff};
                    }
            }
            else
            {
                for (const auto &kv : block.sparse_pairs)
                    if (std::abs(kv.second) > tol)
                        (*all_terms)[idx++] = {kv.first.alpha, kv.first.beta, kv.second};
            }
        }
    }

    template <typename Th, typename Tv>
    HamResult<Tv> stage5_prepare_dense_local_output(size_t ncs, const std::vector<LocalPairTerm<Th, Tv>> *all_terms)
    {
        size_t alloc_nnz = std::max<size_t>(1, ncs);
        Th *final_axs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_azs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_bxs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_bzs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Tv *final_cs = (Tv *)malloc(alloc_nnz * sizeof(Tv));
#pragma omp parallel for
        for (size_t i = 0; i < ncs; ++i)
        {
            final_axs[i] = (*all_terms)[i].alpha.x;
            final_azs[i] = (*all_terms)[i].alpha.z;
            final_bxs[i] = (*all_terms)[i].beta.x;
            final_bzs[i] = (*all_terms)[i].beta.z;
            final_cs[i] = (*all_terms)[i].c;
        }
        return {(void *)final_axs, (void *)final_azs, (void *)final_bxs, (void *)final_bzs, final_cs, ncs};
    }

    template <typename Tv>
    struct HamResult
    {
        void *axs;  // xs 的 α 部分
        void *azs;  // zs 的 α 部分
        void *bxs;  // xs 的 β 部分
        void *bzs;  // zs 的 β 部分
        Tv *cs;     // Float64/ComplexF64
        size_t ncs; // cs 长度
    };

    template <typename Ti,
              typename Tv>
    HamResult<Tv> stage5_prepare_output(size_t ncs, const TermArray<Ti, Tv> *all_terms)
    {
        using Th = half_width_t<Ti>;
        size_t alloc_nnz = std::max<size_t>(1, ncs);
        Th *final_axs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_azs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_bxs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Th *final_bzs = (Th *)malloc(alloc_nnz * sizeof(Th));
        Tv *final_cs = (Tv *)malloc(alloc_nnz * sizeof(Tv));

        if (ncs > 0)
        {
#pragma omp parallel for
            for (size_t i = 0; i < ncs; ++i)
            {
                Pauli<Ti> q = (*all_terms)[i].q;
                final_axs[i] = zip_even_bit_bmi2(q.x);
                final_azs[i] = zip_even_bit_bmi2(q.z);
                final_bxs[i] = zip_odd_bit_bmi2(q.x);
                final_bzs[i] = zip_odd_bit_bmi2(q.z);
                final_cs[i] = (*all_terms)[i].c;
            }
        }

        return {
            (void *)final_axs,
            (void *)final_azs,
            (void *)final_bxs,
            (void *)final_bzs,
            final_cs,
            ncs};
    }

    template <typename Ti, typename Tv>
    HamResult<Tv> generate_hamiltonian_tmpl(
        Tv energy_nuc,
        const Tv *one_body_mo,
        const Tv *two_body_mo,
        int norbs,
        double tol,
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
        size_t ncs = 0;
        stage3_count_and_write<Ti, Tv>(nblocks, tol, &dicts, &all_terms, &ncs);

        dicts.clear();
        dicts.shrink_to_fit();

        auto t3 = std::chrono::high_resolution_clock::now();

        // --- Stage 4 ---
        __gnu_parallel::sort(all_terms.begin(), all_terms.end());

        auto t4 = std::chrono::high_resolution_clock::now();

        // --- Stage 5 ---
        HamResult<Tv> res = stage5_prepare_output<Ti, Tv>(ncs, &all_terms);

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


    template <typename Ti, typename Tv>
    HamResult<Tv> generate_hamiltonian_dense_local_tmpl(
        Tv energy_nuc,
        const Tv *one_body_mo,
        const Tv *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        using Th = half_width_t<Ti>;
        int nthreads = omp_get_max_threads();
        int nblocks = nthreads * 64;

        auto t0 = std::chrono::high_resolution_clock::now();

        LocalSingleArray<Tv> local_single;
        LocalDoubleArray<Tv> local_double;
        stage1_allocate_and_scan<Ti, Tv>(
            nthreads, nblocks, norbs, tol, one_body_mo, two_body_mo,
            &local_single, &local_double);

        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<DenseLocalBlock<Th, Tv>> blocks;
        stage2_reduce_dense_local<Ti, Tv>(
            nthreads, nblocks, energy_nuc, tol,
            &local_single, &local_double, &blocks);

        local_single.clear();
        local_single.shrink_to_fit();
        local_double.clear();
        local_double.shrink_to_fit();

        auto t2 = std::chrono::high_resolution_clock::now();

        std::vector<LocalPairTerm<Th, Tv>> all_terms;
        size_t ncs = 0;
        stage3_dense_local_count_and_write<Th, Tv>(nblocks, tol, &blocks, &all_terms, &ncs);

        auto t3 = std::chrono::high_resolution_clock::now();

        __gnu_parallel::sort(all_terms.begin(), all_terms.end(), [](const auto &a, const auto &b) {
            Ti ax = uzip_even_bit_bmi2(a.alpha.x) ^ uzip_odd_bit_bmi2(a.beta.x);
            Ti bx = uzip_even_bit_bmi2(b.alpha.x) ^ uzip_odd_bit_bmi2(b.beta.x);
            if (ax != bx)
                return ax < bx;
            Ti az = uzip_even_bit_bmi2(a.alpha.z) ^ uzip_odd_bit_bmi2(a.beta.z);
            Ti bz = uzip_even_bit_bmi2(b.alpha.z) ^ uzip_odd_bit_bmi2(b.beta.z);
            return az < bz;
        });

        auto t4 = std::chrono::high_resolution_clock::now();

        HamResult<Tv> res = stage5_prepare_dense_local_output<Th, Tv>(ncs, &all_terms);

        auto t5 = std::chrono::high_resolution_clock::now();

        if (verbose)
        {
            std::chrono::duration<double> t_scan = t1 - t0;
            std::chrono::duration<double> t_reduce = t2 - t1;
            std::chrono::duration<double> t_write = t3 - t2;
            std::chrono::duration<double> t_sort = t4 - t3;
            std::chrono::duration<double> t_output = t5 - t4;
            std::chrono::duration<double> t_total = t5 - t0;

            size_t dense_buckets = 0, sparse_buckets = 0, dense_entries = 0;
            for (const auto &block : blocks)
            {
                if (block.dense)
                {
                    dense_buckets++;
                    dense_entries += block.coeff_matrix.size();
                }
                else
                    sparse_buckets++;
            }

            printf("\n");
            printf("[Summary] (Experimental Dense-Local Accumulator)\n");
            printf("[Summary] Orbital number:       %d\n", norbs);
            printf("[Summary] Number of threads:    %d\n", nthreads);
            printf("[Summary] Bucket size:          %d\n", nblocks);
            printf("[Summary] Number of cs:         %zu\n", res.ncs);
            printf("[Summary] Dense buckets:        %zu\n", dense_buckets);
            printf("[Summary] Sparse fallback:      %zu\n", sparse_buckets);
            printf("[Summary] Dense matrix entries: %zu\n", dense_entries);

            printf("[Time] Total:         %.4f seconds\n", t_total.count());
            printf("[Time] Scanning:      %.4f seconds\n", t_scan.count());
            printf("[Time] Dense reduce:  %.4f seconds\n", t_reduce.count());
            printf("[Time] Writing:       %.4f seconds\n", t_write.count());
            printf("[Time] Sorting:       %.4f seconds\n", t_sort.count());
            printf("[Time] Output:        %.4f seconds\n", t_output.count());
        }

        blocks.clear();
        blocks.shrink_to_fit();

        return res;
    }

    template <typename Ti, typename Tv>
    FORCE_INLINE void insert_1body_real(FastDict<Ti, Tv> *__restrict dict, int p, int q, Tv coeff)
    {
        if (p == q)
        {
            Tv cc = 0.5 * coeff;
            Pauli<Ti> k;
            k.x = get_zero<Ti>();
            k.z = get_zero<Ti>();
            (*dict)[k] += cc;
            k.z = get_one<Ti>() << p;
            (*dict)[k] -= cc;
            return;
        }
        
        Ti o = get_one<Ti>();
        Ti x1 = o << p, z11 = x1 - o, z12 = (x1 << 1) - o;
        Ti x2 = o << q, z21 = x2 - o, z22 = (x2 << 1) - o;
        Ti x12 = x1 ^ x2;
        Tv cc = 0.25 * coeff;

        Tv c1 = parity_coeff(cc, popcnt(z11 & x2) & 1);
        Tv c2 = parity_coeff(cc, popcnt(z12 & x2) & 1);

        Pauli<Ti> k;
        k.x = x12;
        k.z = z11 ^ z21;
        (*dict)[k] += c1; // XX
        k.z = z12 ^ z22;
        (*dict)[k] -= c2; // YY
    }

    template <typename Ti, typename Tv>
    FORCE_INLINE void insert_2body_real(FastDict<Ti, Tv> *__restrict dict, int p, int q, int r, int s, Tv coeff)
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

        Tv c11 = parity_coeff(cc, p1 ^ p31_x4);
        Tv c12 = parity_coeff(cc, p1 ^ p32_x4);
        Tv c21 = parity_coeff(cc, p2 ^ p31_x4);
        Tv c22 = parity_coeff(cc, p2 ^ p32_x4);
        Tv c31 = parity_coeff(cc, p3 ^ p31_x4);
        Tv c32 = parity_coeff(cc, p3 ^ p32_x4);
        Tv c41 = parity_coeff(cc, p4 ^ p31_x4);
        Tv c42 = parity_coeff(cc, p4 ^ p32_x4);

        Pauli<Ti> k;
        k.x = x1234;
        k.z = z11 ^ z21 ^ z31 ^ z41;
        (*dict)[k] += c11; // XXXX
        k.z = z11 ^ z21 ^ z32 ^ z42;
        (*dict)[k] += c12; // XXYY
        k.z = z11 ^ z22 ^ z31 ^ z42;
        (*dict)[k] -= c21; // XYXY
        k.z = z11 ^ z22 ^ z32 ^ z41;
        (*dict)[k] -= c22; // XYYX
        k.z = z12 ^ z21 ^ z31 ^ z42;
        (*dict)[k] -= c31; // YXXY
        k.z = z12 ^ z21 ^ z32 ^ z41;
        (*dict)[k] -= c32; // YXYX
        k.z = z12 ^ z22 ^ z31 ^ z41;
        (*dict)[k] += c41; // YYXX
        k.z = z12 ^ z22 ^ z32 ^ z42;
        (*dict)[k] += c42; // YYYY
    }

    template <typename Ti, typename Tv>
    void stage1_allocate_and_scan_real(
        int nthreads, int nblocks, int norbs, Tv tol,
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

        // 因为扫描空间降到了 1/4，预分配相应下调
        size_t revsize = (N4 / (4 * nthreads * nblocks)) * 0.1 + 10;
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

// --- 1-body 扫描 ---
#pragma omp for schedule(static) collapse(2)
            for (int p = 0; p < norbs; ++p)
            {
                for (int q = 0; q < norbs; ++q)
                {
                    // 1-body 是厄米矩阵，只提取上三角作为代表元
                    if (p > q)
                        continue;

                    size_t flat_idx = static_cast<size_t>(p) +
                                      static_cast<size_t>(q) * N1;

                    Tv val = one_body_mo[flat_idx];
                    if (std::abs(val) > tol)
                    {
                        Tv weight = (p == q) ? 1.0 : 2.0;
                        Ti mask = (ONE << p) ^ (ONE << q);
                        uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                        (*local_single)[tid][bucket_idx].push_back({p, q, val * weight});
                    }
                }
            }

// --- 2-body 扫描 ---
#pragma omp for schedule(dynamic, 64) collapse(4)
            for (size_t s = 0; s < N1; ++s)
            {
                for (size_t r = 0; r < N1; ++r)
                {
                    for (size_t q = 0; q < N1; ++q)
                    {
                        for (size_t p = 0; p < N1; ++p)
                        {
                            // 计算当前排列及关联排列的绝对唯一 ID
                            size_t i1 = s * N3 + r * N2 + q * N1 + p; // (p,q,r,s)
                            size_t i2 = r * N3 + s * N2 + p * N1 + q; // (q,p,s,r) 算符等价
                            size_t i3 = p * N3 + q * N2 + r * N1 + s; // (s,r,q,p) 厄米共轭
                            size_t i4 = q * N3 + p * N2 + s * N1 + r; // (r,s,p,q) 厄米共轭的算符等价

                            // 仅当当前 i1 是这 4 个等价元中的最大值时，才进行处理（选作代表元）
                            size_t max_idx = std::max({i1, i2, i3, i4});
                            if (i1 == max_idx)
                            {
                                size_t flat_idx = p + q * N1 + r * N2 + s * N3;
                                Tv val = two_body_mo[flat_idx];

                                if (std::abs(val) > tol)
                                {
                                    // 动态计算该代表元所代表的独特排列数量 (1, 2, 或 4)
                                    int duplicates = 0;
                                    if (i1 == i1)
                                        duplicates++;
                                    if (i1 == i2)
                                        duplicates++;
                                    if (i1 == i3)
                                        duplicates++;
                                    if (i1 == i4)
                                        duplicates++;
                                    Tv weight = 4.0 / static_cast<Tv>(duplicates);

                                    Ti mask = (ONE << p) ^ (ONE << q) ^
                                              (ONE << r) ^ (ONE << s);

                                    uint32_t bucket_idx = mix_hash(fold_for_hash(mask)) % nblocks;
                                    (*local_double)[tid][bucket_idx].push_back({(int)p, (int)q, (int)r, (int)s, val * weight});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename Ti, typename Tv>
    void stage2_reduce_dictionaries_real(
        int nthreads, int nblocks, Tv energy_nuc, Tv tol,
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
                // 容量预估大幅下调：1-body产生2项，2-body产生8项
                raw_inserts += (*local_single)[t][b].size() * 2 +
                               (*local_double)[t][b].size() * 8;
            }

            (*dicts)[b].reserve(raw_inserts / 2 + 100);
            FastDict<Ti, Tv> *__restrict dict_ptr = &(*dicts)[b];

            for (int t = 0; t < nthreads; ++t)
            {
                const SingleTerm<Tv> *s_ptr = (*local_single)[t][b].data();
                for (size_t i = 0, s_len = (*local_single)[t][b].size(); i < s_len; ++i)
                {
                    Tv ci = s_ptr[i].val;
                    insert_1body_real<Ti, Tv>(
                        dict_ptr,
                        2 * s_ptr[i].p,
                        2 * s_ptr[i].q,
                        ci);
                    insert_1body_real<Ti, Tv>(
                        dict_ptr,
                        2 * s_ptr[i].p + 1,
                        2 * s_ptr[i].q + 1,
                        ci);
                }

                const DoubleTerm<Tv> *d_ptr = (*local_double)[t][b].data();
                for (size_t i = 0, d_len = (*local_double)[t][b].size(); i < d_len; ++i)
                {
                    Tv ci = d_ptr[i].val * 0.5;
                    insert_2body_real<Ti, Tv>(
                        dict_ptr,
                        2 * d_ptr[i].p,
                        2 * d_ptr[i].q,
                        2 * d_ptr[i].r,
                        2 * d_ptr[i].s,
                        ci);
                    insert_2body_real<Ti, Tv>(
                        dict_ptr,
                        2 * d_ptr[i].p + 1,
                        2 * d_ptr[i].q + 1,
                        2 * d_ptr[i].r + 1,
                        2 * d_ptr[i].s + 1,
                        ci);
                    insert_2body_real<Ti, Tv>(
                        dict_ptr,
                        2 * d_ptr[i].p,
                        2 * d_ptr[i].q + 1,
                        2 * d_ptr[i].r + 1,
                        2 * d_ptr[i].s,
                        ci);
                    insert_2body_real<Ti, Tv>(
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

    template <typename Ti>
    HamResult<double> generate_hamiltonian_real_tmpl(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        int nthreads = omp_get_max_threads();
        int nblocks = nthreads * 64;

        auto t0 = std::chrono::high_resolution_clock::now();

        LocalSingleArray<double> local_single;
        LocalDoubleArray<double> local_double;
        stage1_allocate_and_scan_real<Ti, double>(nthreads, nblocks, norbs, tol, one_body_mo, two_body_mo, &local_single, &local_double);

        auto t1 = std::chrono::high_resolution_clock::now();

        DictArray<Ti, double> dicts;
        stage2_reduce_dictionaries_real<Ti, double>(nthreads, nblocks, energy_nuc, tol,
                                                    &local_single, &local_double, &dicts);

        local_single.clear();
        local_single.shrink_to_fit();
        local_double.clear();
        local_double.shrink_to_fit();

        auto t2 = std::chrono::high_resolution_clock::now();

        TermArray<Ti, double> all_terms;
        size_t ncs = 0;
        stage3_count_and_write<Ti, double>(nblocks, tol, &dicts, &all_terms, &ncs);
        dicts.clear();
        dicts.shrink_to_fit();

        auto t3 = std::chrono::high_resolution_clock::now();

        __gnu_parallel::sort(all_terms.begin(), all_terms.end());

        auto t4 = std::chrono::high_resolution_clock::now();

        HamResult<double> res = stage5_prepare_output<Ti, double>(ncs, &all_terms);

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
            printf("[Summary] (Real-Integral Specialized Engine)\n");
            printf("[Summary] Orbital number:       %d\n", norbs);
            printf("[Summary] Number of threads:    %d\n", nthreads);
            printf("[Summary] Bucket size:          %d\n", nblocks);
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
} // namespace binsim::ham
