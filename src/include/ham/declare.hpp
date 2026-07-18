#pragma once
#include <complex>
#include <iostream>
#include "core/bit.hpp"
#include "ham/ham.hpp"

#define DECLARE_HAM_REAL_INTERFACES(Ti, MAX_NORB, SUFFIX)                        \
    extern "C"                                                                   \
    {                                                                            \
        binsim::ham::HamResult<double> generate_hamiltonian_real##SUFFIX(        \
            double energy_nuc,                                                   \
            const double *one_body_mo,                                           \
            const double *two_body_mo,                                           \
            int norbs,                                                           \
            double tol,                                                          \
            bool verbose)                                                        \
        {                                                                        \
            if (norbs > MAX_NORB)                                                \
            {                                                                    \
                std::cerr << "Error: norbs > " << MAX_NORB                       \
                          << " not supported for " #Ti " backend." << std::endl; \
                return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};         \
            }                                                                    \
            return binsim::ham::generate_hamiltonian_real_tmpl<Ti>(              \
                energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);      \
        }                                                                        \
    }

#define DECLARE_HAM_INTERFACES(Ti, MAX_NORB, F64_SUFFIX, C64_SUFFIX)             \
    extern "C"                                                                   \
    {                                                                            \
        using complex64 = std::complex<double>;                                  \
                                                                                 \
        binsim::ham::HamResult<double> generate_hamiltonian##F64_SUFFIX(         \
            double energy_nuc,                                                   \
            const double *one_body_mo,                                           \
            const double *two_body_mo,                                           \
            int norbs,                                                           \
            double tol,                                                          \
            bool verbose)                                                        \
        {                                                                        \
            if (norbs > MAX_NORB)                                                \
            {                                                                    \
                std::cerr << "Error: norbs > " << MAX_NORB                       \
                          << " not supported for " #Ti " backend." << std::endl; \
                return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};         \
            }                                                                    \
            return binsim::ham::generate_hamiltonian_tmpl<Ti, double>(           \
                energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);      \
        }                                                                        \
                                                                                 \
        binsim::ham::HamResult<complex64> generate_hamiltonian##C64_SUFFIX(      \
            complex64 energy_nuc,                                                \
            const complex64 *one_body_mo,                                        \
            const complex64 *two_body_mo,                                        \
            int norbs,                                                           \
            double tol,                                                          \
            bool verbose)                                                        \
        {                                                                        \
            if (norbs > MAX_NORB)                                                \
            {                                                                    \
                std::cerr << "Error: norbs > " << MAX_NORB                       \
                          << " not supported for " #Ti " backend." << std::endl; \
                return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};         \
            }                                                                    \
            return binsim::ham::generate_hamiltonian_tmpl<Ti, complex64>(        \
                energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);      \
        }                                                                        \
    }
