#pragma once

// Experimental real-integral JW engine based on pair-space contraction.
//
// Four stages:
//   1. Build directed alpha/beta E_pq = a^+_p a_q pair tables (O(N^2)).
//   2. Scan the real integral symmetry representatives and build local dense
//      alpha/beta Pauli coordinate maps for diag/pure_a/pure_b/mixed blocks.
//   3. Repeat the same contraction and accumulate into dense arrays/matrices.
//   4. Flatten the four blocks, remove numerical zeros, sort and return the
//      same HamResult<double> layout as ham.hpp.
//
// This is intentionally an independent correctness/prototyping path. It uses
// no global Pauli->coefficient dictionary during accumulation. The mixed block
// is a full dense matrix, so this test implementation is not intended for very
// large active spaces without replacing it by tiled dense storage.

#include "ham/ham.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace binsim::ham::pair_test
{
    template <typename Ti>
    struct ComplexPauliTerm
    {
        Pauli<Ti> q{};
        std::complex<double> c{};
    };

    template <typename Ti>
    struct DirectedPair
    {
        std::vector<ComplexPauliTerm<Ti>> terms;
    };

    template <typename Ti>
    struct PairTables
    {
        int norbs = 0;
        std::vector<DirectedPair<Ti>> alpha;
        std::vector<DirectedPair<Ti>> beta;

        const DirectedPair<Ti> &a(int p, int q) const
        {
            return alpha[static_cast<size_t>(p) * norbs + q];
        }

        const DirectedPair<Ti> &b(int p, int q) const
        {
            return beta[static_cast<size_t>(p) * norbs + q];
        }
    };

    template <typename Ti>
    FORCE_INLINE std::complex<double> pauli_phase(const Pauli<Ti> &lhs,
                                                  const Pauli<Ti> &rhs)
    {
        // Physical-Pauli convention: P(x,z)=i^(popcnt(x&z)) X^x Z^z,
        // so (x,z)=(1,1) denotes Y.  Multiplication gives
        // i^[n1+n2-nout+2 popcnt(z1&x2)] P(x1^x2,z1^z2).
        const Ti out_x = lhs.x ^ rhs.x;
        const Ti out_z = lhs.z ^ rhs.z;
        int e = static_cast<int>(popcnt(lhs.x & lhs.z)) +
                static_cast<int>(popcnt(rhs.x & rhs.z)) -
                static_cast<int>(popcnt(out_x & out_z)) +
                2 * static_cast<int>(popcnt(lhs.z & rhs.x));
        e %= 4;
        if (e < 0)
            e += 4;
        switch (e)
        {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
        }
    }

    template <typename Ti>
    FORCE_INLINE ComplexPauliTerm<Ti> multiply_pauli(const ComplexPauliTerm<Ti> &lhs,
                                                     const ComplexPauliTerm<Ti> &rhs)
    {
        ComplexPauliTerm<Ti> out;
        out.q.x = lhs.q.x ^ rhs.q.x;
        out.q.z = lhs.q.z ^ rhs.q.z;
        out.c = lhs.c * rhs.c * pauli_phase(lhs.q, rhs.q);
        return out;
    }

    template <typename Ti>
    inline void append_merged(std::vector<ComplexPauliTerm<Ti>> &dst,
                              const ComplexPauliTerm<Ti> &term,
                              double eps = 1e-15)
    {
        for (auto &old : dst)
        {
            if (old.q == term.q)
            {
                old.c += term.c;
                return;
            }
        }
        if (std::abs(term.c) > eps)
            dst.push_back(term);
    }

    template <typename Ti>
    inline std::array<ComplexPauliTerm<Ti>, 2> ladder_terms(int mode,
                                                            bool creation)
    {
        const Ti one = get_one<Ti>();
        const Ti bit = one << mode;
        const Ti prefix = bit - one;

        ComplexPauliTerm<Ti> xterm;
        xterm.q.x = bit;
        xterm.q.z = prefix;
        xterm.c = {0.5, 0.0};

        ComplexPauliTerm<Ti> yterm;
        yterm.q.x = bit;
        yterm.q.z = prefix ^ bit;
        yterm.c = creation ? std::complex<double>(0.0, -0.5)
                           : std::complex<double>(0.0, 0.5);

        return {xterm, yterm};
    }

    template <typename Ti>
    inline DirectedPair<Ti> build_directed_pair(int creation_mode,
                                                int annihilation_mode)
    {
        DirectedPair<Ti> pair;
        pair.terms.reserve(4);
        const auto lhs = ladder_terms<Ti>(creation_mode, true);
        const auto rhs = ladder_terms<Ti>(annihilation_mode, false);
        for (const auto &a : lhs)
            for (const auto &b : rhs)
                append_merged(pair.terms, multiply_pauli(a, b));
        return pair;
    }

    // Stage 1: two independent O(N^2) local processes.
    template <typename Ti>
    PairTables<Ti> stage1_build_pair_tables(int norbs)
    {
        PairTables<Ti> tables;
        tables.norbs = norbs;
        tables.alpha.resize(static_cast<size_t>(norbs) * norbs);
        tables.beta.resize(static_cast<size_t>(norbs) * norbs);

#pragma omp parallel for schedule(static) collapse(2)
        for (int p = 0; p < norbs; ++p)
        {
            for (int q = 0; q < norbs; ++q)
            {
                const size_t idx = static_cast<size_t>(p) * norbs + q;
                tables.alpha[idx] = build_directed_pair<Ti>(2 * p, 2 * q);
                tables.beta[idx] = build_directed_pair<Ti>(2 * p + 1, 2 * q + 1);
            }
        }
        return tables;
    }

    template <typename Th>
    struct LocalKeyHash
    {
        size_t operator()(const Pauli<Th> &q) const noexcept
        {
            return PauliHash<Th>{}(q);
        }
    };

    template <typename Th>
    struct DenseAxis
    {
        ankerl::unordered_dense::map<Pauli<Th>, uint32_t, LocalKeyHash<Th>> ids;
        std::vector<Pauli<Th>> keys;

        uint32_t intern(const Pauli<Th> &q)
        {
            auto [it, inserted] = ids.try_emplace(q, static_cast<uint32_t>(keys.size()));
            if (inserted)
                keys.push_back(q);
            return it->second;
        }

        uint32_t find_id(const Pauli<Th> &q) const
        {
            auto it = ids.find(q);
            if (it == ids.end())
                throw std::logic_error("pair_test dense axis was not discovered in stage 2");
            return it->second;
        }
    };

    enum class BlockKind : uint8_t
    {
        diag,
        pure_a,
        pure_b,
        mixed
    };

    template <typename Th>
    struct SplitKey
    {
        Pauli<Th> a{};
        Pauli<Th> b{};
    };

    template <typename Ti>
    FORCE_INLINE SplitKey<half_width_t<Ti>> split_key(const Pauli<Ti> &q)
    {
        using Th = half_width_t<Ti>;
        SplitKey<Th> out;
        out.a.x = zip_even_bit_bmi2(q.x);
        out.a.z = zip_even_bit_bmi2(q.z);
        out.b.x = zip_odd_bit_bmi2(q.x);
        out.b.z = zip_odd_bit_bmi2(q.z);
        return out;
    }

    template <typename Th>
    FORCE_INLINE bool is_identity(const Pauli<Th> &q)
    {
        return q.x == get_zero<Th>() && q.z == get_zero<Th>();
    }

    template <typename Th>
    FORCE_INLINE BlockKind classify(const SplitKey<Th> &q)
    {
        // diag is reserved for fully Z-diagonal strings, including I/Z-only
        // pure-spin strings. The remaining pure blocks contain off-diagonal
        // X/Y strings on one spin sector only.
        if (q.a.x == get_zero<Th>() && q.b.x == get_zero<Th>())
            return BlockKind::diag;
        if (is_identity(q.b))
            return BlockKind::pure_a;
        if (is_identity(q.a))
            return BlockKind::pure_b;
        return BlockKind::mixed;
    }

    template <typename Ti, typename Emit>
    inline void emit_pair_product(const DirectedPair<Ti> &lhs,
                                  const DirectedPair<Ti> &rhs,
                                  std::complex<double> scale,
                                  Emit &&emit)
    {
        for (const auto &a : lhs.terms)
        {
            for (const auto &b : rhs.terms)
            {
                auto term = multiply_pauli(a, b);
                term.c *= scale;
                emit(term);
            }
        }
    }

    template <typename Ti, typename Emit>
    inline void emit_same_spin_two_body(const PairTables<Ti> &tables,
                                        bool alpha,
                                        int p, int q, int r, int s,
                                        double scale,
                                        Emit &&emit)
    {
        const auto &ps = alpha ? tables.a(p, s) : tables.b(p, s);
        const auto &qr = alpha ? tables.a(q, r) : tables.b(q, r);
        emit_pair_product(ps, qr, {scale, 0.0}, emit);

        // a^+_p a^+_q a_r a_s = E_ps E_qr - delta_sq E_pr.
        if (s == q)
        {
            const auto &pr = alpha ? tables.a(p, r) : tables.b(p, r);
            for (const auto &term0 : pr.terms)
            {
                auto term = term0;
                term.c *= -scale;
                emit(term);
            }
        }
    }

    template <typename Ti, typename Emit>
    inline void emit_mixed_two_body(const PairTables<Ti> &tables,
                                    bool first_alpha,
                                    int p, int q, int r, int s,
                                    double scale,
                                    Emit &&emit)
    {
        const auto &ps = first_alpha ? tables.a(p, s) : tables.b(p, s);
        const auto &qr = first_alpha ? tables.b(q, r) : tables.a(q, r);
        emit_pair_product(ps, qr, {scale, 0.0}, emit);
    }

    template <typename F>
    inline void for_each_unique_real_eri_orbit(int p, int q, int r, int s, F &&f)
    {
        const std::array<std::array<int, 4>, 4> orbit{{
            {{p, q, r, s}},
            {{q, p, s, r}},
            {{s, r, q, p}},
            {{r, s, p, q}},
        }};
        for (size_t i = 0; i < orbit.size(); ++i)
        {
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j)
                duplicate = duplicate || (orbit[i] == orbit[j]);
            if (!duplicate)
                f(orbit[i][0], orbit[i][1], orbit[i][2], orbit[i][3]);
        }
    }

    template <typename Ti, typename Emit>
    void enumerate_contributions(const PairTables<Ti> &tables,
                                 double energy_nuc,
                                 const double *one_body_mo,
                                 const double *two_body_mo,
                                 int norbs,
                                 double integral_tol,
                                 Emit &&emit)
    {
        const size_t N1 = static_cast<size_t>(norbs);
        const size_t N2 = N1 * N1;
        const size_t N3 = N2 * N1;

        if (std::abs(energy_nuc) > integral_tol)
        {
            ComplexPauliTerm<Ti> term;
            term.q.x = get_zero<Ti>();
            term.q.z = get_zero<Ti>();
            term.c = {energy_nuc, 0.0};
            emit(term);
        }

        // Real Hermitian one-body matrix: explicitly emit both orientations.
        for (int p = 0; p < norbs; ++p)
        {
            for (int q = p; q < norbs; ++q)
            {
                const double value = one_body_mo[static_cast<size_t>(p) +
                                                 static_cast<size_t>(q) * N1];
                if (std::abs(value) <= integral_tol)
                    continue;

                auto emit_pair = [&](const DirectedPair<Ti> &pair)
                {
                    for (const auto &t0 : pair.terms)
                    {
                        auto t = t0;
                        t.c *= value;
                        emit(t);
                    }
                };

                emit_pair(tables.a(p, q));
                emit_pair(tables.b(p, q));
                if (p != q)
                {
                    emit_pair(tables.a(q, p));
                    emit_pair(tables.b(q, p));
                }
            }
        }

        // Select one representative from each fourfold real-integral orbit,
        // then explicitly expand the unique members of that orbit. This keeps
        // the pair algebra exact and lets imaginary terms cancel naturally.
        for (int s = 0; s < norbs; ++s)
        {
            for (int r = 0; r < norbs; ++r)
            {
                for (int q = 0; q < norbs; ++q)
                {
                    for (int p = 0; p < norbs; ++p)
                    {
                        const size_t i1 = static_cast<size_t>(s) * N3 + static_cast<size_t>(r) * N2 + static_cast<size_t>(q) * N1 + p;
                        const size_t i2 = static_cast<size_t>(r) * N3 + static_cast<size_t>(s) * N2 + static_cast<size_t>(p) * N1 + q;
                        const size_t i3 = static_cast<size_t>(p) * N3 + static_cast<size_t>(q) * N2 + static_cast<size_t>(r) * N1 + s;
                        const size_t i4 = static_cast<size_t>(q) * N3 + static_cast<size_t>(p) * N2 + static_cast<size_t>(s) * N1 + r;
                        if (i1 != std::max({i1, i2, i3, i4}))
                            continue;

                        const double value = two_body_mo[static_cast<size_t>(p) +
                                                         static_cast<size_t>(q) * N1 +
                                                         static_cast<size_t>(r) * N2 +
                                                         static_cast<size_t>(s) * N3];
                        if (std::abs(value) <= integral_tol)
                            continue;

                        // Hamiltonian carries the conventional 1/2 prefactor.
                        const double scale = 0.5 * value;
                        for_each_unique_real_eri_orbit(p, q, r, s,
                                                       [&](int pp, int qq, int rr, int ss)
                                                       {
                                                           emit_same_spin_two_body(tables, true, pp, qq, rr, ss, scale, emit);
                                                           emit_same_spin_two_body(tables, false, pp, qq, rr, ss, scale, emit);
                                                           emit_mixed_two_body(tables, true, pp, qq, rr, ss, scale, emit);
                                                           emit_mixed_two_body(tables, false, pp, qq, rr, ss, scale, emit);
                                                       });
                    }
                }
            }
        }
    }

    template <typename Th>
    struct DenseLayout
    {
        DenseAxis<Th> diag_a, diag_b;
        DenseAxis<Th> pure_a;
        DenseAxis<Th> pure_b;
        DenseAxis<Th> mixed_a, mixed_b;
    };

    // Stage 2: discover local dense coordinates only. No coefficient dictionary.
    template <typename Ti>
    DenseLayout<half_width_t<Ti>> stage2_build_dense_layout(
        const PairTables<Ti> &tables,
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double integral_tol)
    {
        using Th = half_width_t<Ti>;
        DenseLayout<Th> layout;

        enumerate_contributions(tables, energy_nuc, one_body_mo, two_body_mo,
                                norbs, integral_tol,
                                [&](const ComplexPauliTerm<Ti> &term)
                                {
                                    if (std::abs(term.c) <= integral_tol)
                                        return;
                                    const auto key = split_key(term.q);
                                    switch (classify(key))
                                    {
                                    case BlockKind::diag:
                                        layout.diag_a.intern(key.a);
                                        layout.diag_b.intern(key.b);
                                        break;
                                    case BlockKind::pure_a:
                                        layout.pure_a.intern(key.a);
                                        break;
                                    case BlockKind::pure_b:
                                        layout.pure_b.intern(key.b);
                                        break;
                                    case BlockKind::mixed:
                                        layout.mixed_a.intern(key.a);
                                        layout.mixed_b.intern(key.b);
                                        break;
                                    }
                                });
        return layout;
    }

    struct DenseBlocks
    {
        std::vector<std::complex<double>> diag;
        std::vector<std::complex<double>> pure_a;
        std::vector<std::complex<double>> pure_b;
        std::vector<std::complex<double>> mixed;
    };

    inline size_t checked_product(size_t a, size_t b, const char *name)
    {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
            throw std::overflow_error(name);
        return a * b;
    }

    // Stage 3: pure dense accumulation using local dense IDs.
    template <typename Ti>
    DenseBlocks stage3_dense_contract(
        const PairTables<Ti> &tables,
        const DenseLayout<half_width_t<Ti>> &layout,
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double integral_tol)
    {
        DenseBlocks blocks;
        const size_t ndiag = checked_product(layout.diag_a.keys.size(), layout.diag_b.keys.size(), "diag dense block overflow");
        const size_t nmixed = checked_product(layout.mixed_a.keys.size(), layout.mixed_b.keys.size(), "mixed dense block overflow");
        blocks.diag.assign(ndiag, {});
        blocks.pure_a.assign(layout.pure_a.keys.size(), {});
        blocks.pure_b.assign(layout.pure_b.keys.size(), {});
        blocks.mixed.assign(nmixed, {});

        enumerate_contributions(tables, energy_nuc, one_body_mo, two_body_mo,
                                norbs, integral_tol,
                                [&](const ComplexPauliTerm<Ti> &term)
                                {
                                    if (std::abs(term.c) <= integral_tol)
                                        return;
                                    const auto key = split_key(term.q);
                                    switch (classify(key))
                                    {
                                    case BlockKind::diag:
                                    {
                                        const size_t ia = layout.diag_a.find_id(key.a);
                                        const size_t ib = layout.diag_b.find_id(key.b);
                                        blocks.diag[ia * layout.diag_b.keys.size() + ib] += term.c;
                                        break;
                                    }
                                    case BlockKind::pure_a:
                                        blocks.pure_a[layout.pure_a.find_id(key.a)] += term.c;
                                        break;
                                    case BlockKind::pure_b:
                                        blocks.pure_b[layout.pure_b.find_id(key.b)] += term.c;
                                        break;
                                    case BlockKind::mixed:
                                    {
                                        const size_t ia = layout.mixed_a.find_id(key.a);
                                        const size_t ib = layout.mixed_b.find_id(key.b);
                                        blocks.mixed[ia * layout.mixed_b.keys.size() + ib] += term.c;
                                        break;
                                    }
                                    }
                                });
        return blocks;
    }

    template <typename Ti>
    FORCE_INLINE Pauli<Ti> join_key(const Pauli<half_width_t<Ti>> &a,
                                    const Pauli<half_width_t<Ti>> &b)
    {
        // Portable inverse of zip_even/zip_odd. This is used only once per
        // surviving output term in the experimental path.
        Pauli<Ti> out{get_zero<Ti>(), get_zero<Ti>()};
        const Ti one = get_one<Ti>();
        const int nbits = static_cast<int>(sizeof(half_width_t<Ti>) * 8);
        for (int i = 0; i < nbits; ++i)
        {
            const auto hbit = get_one<half_width_t<Ti>>() << i;
            if ((a.x & hbit) != get_zero<half_width_t<Ti>>())
                out.x = out.x ^ (one << (2 * i));
            if ((a.z & hbit) != get_zero<half_width_t<Ti>>())
                out.z = out.z ^ (one << (2 * i));
            if ((b.x & hbit) != get_zero<half_width_t<Ti>>())
                out.x = out.x ^ (one << (2 * i + 1));
            if ((b.z & hbit) != get_zero<half_width_t<Ti>>())
                out.z = out.z ^ (one << (2 * i + 1));
        }
        return out;
    }

    // Stage 4: flatten blocks, validate reality, sort, and build HamResult.
    template <typename Ti>
    HamResult<double> stage4_finalize(
        const DenseLayout<half_width_t<Ti>> &layout,
        const DenseBlocks &blocks,
        double output_tol,
        double imag_tol = 1e-10)
    {
        using Th = half_width_t<Ti>;
        TermArray<Ti, double> terms;

        auto append = [&](const Pauli<Th> &a, const Pauli<Th> &b,
                          const std::complex<double> &c)
        {
            if (std::abs(c.imag()) > imag_tol * std::max(1.0, std::abs(c.real())))
                throw std::runtime_error("pair_test real contraction left a non-negligible imaginary coefficient");
            if (std::abs(c.real()) > output_tol)
                terms.push_back({join_key<Ti>(a, b), c.real()});
        };

        for (size_t ia = 0; ia < layout.diag_a.keys.size(); ++ia)
            for (size_t ib = 0; ib < layout.diag_b.keys.size(); ++ib)
                append(layout.diag_a.keys[ia], layout.diag_b.keys[ib],
                       blocks.diag[ia * layout.diag_b.keys.size() + ib]);

        const Pauli<Th> identity{get_zero<Th>(), get_zero<Th>()};
        for (size_t i = 0; i < layout.pure_a.keys.size(); ++i)
            append(layout.pure_a.keys[i], identity, blocks.pure_a[i]);
        for (size_t i = 0; i < layout.pure_b.keys.size(); ++i)
            append(identity, layout.pure_b.keys[i], blocks.pure_b[i]);

        for (size_t ia = 0; ia < layout.mixed_a.keys.size(); ++ia)
            for (size_t ib = 0; ib < layout.mixed_b.keys.size(); ++ib)
                append(layout.mixed_a.keys[ia], layout.mixed_b.keys[ib],
                       blocks.mixed[ia * layout.mixed_b.keys.size() + ib]);

        __gnu_parallel::sort(terms.begin(), terms.end());
        return stage5_prepare_output<Ti, double>(terms.size(), &terms);
    }

    template <typename Ti>
    HamResult<double> generate_hamiltonian_real_pair_tmpl(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto tables = stage1_build_pair_tables<Ti>(norbs);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> pair_time = t1 - t0;
        std::printf("[Time] Stage 1 pair tables:     %.4f seconds\n", pair_time.count());

        auto layout = stage2_build_dense_layout(
            tables, energy_nuc, one_body_mo, two_body_mo, norbs, tol);
        const auto t2 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> layout_time = t2 - t1;
        std::printf("[Time] Stage 2 dense layout:    %.4f seconds\n", layout_time.count());

        auto blocks = stage3_dense_contract(
            tables, layout, energy_nuc, one_body_mo, two_body_mo, norbs, tol);
        const auto t3 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> contract_time = t3 - t2;
        std::printf("[Time] Stage 3 contraction:     %.4f seconds\n", contract_time.count());

        auto result = stage4_finalize<Ti>(layout, blocks, tol);
        const auto t4 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> final_time = t4 - t3;
        std::printf("[Time] Stage 4 finalization:    %.4f seconds\n", final_time.count());

        return result;
    }

} // namespace binsim::ham::pair_test
