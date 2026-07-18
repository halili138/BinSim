#include <complex>
#include <iostream>

#include "core/bit.hpp"
#include "ham/ham.hpp"

extern "C"
{
    using complex64 = std::complex<double>;

    // 处理最高 (31o, 62q) 的分子体系
    binsim::ham::HamResult<double> generate_hamiltonian_64_128_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 31)
        {
            std::cerr << "Error: norbs > 31 not supported for uint64 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint64, double>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    // 处理最高 (31o, 62q) 的周期性体系
    binsim::ham::HamResult<complex64> generate_hamiltonian_64_128_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 31)
        {
            std::cerr << "Error: norbs > 31 not supported for uint64 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint64, complex64>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    // 处理最高 (63o, 126q) 的分子体系
    binsim::ham::HamResult<double> generate_hamiltonian_128_256_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 63)
        {
            std::cerr << "Error: norbs > 63 not supported for uint128 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint128, double>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    // 处理最高 (63o, 126q) 的周期性体系
    binsim::ham::HamResult<complex64> generate_hamiltonian_128_256_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 63)
        {
            std::cerr << "Error: norbs > 63 not supported for uint128 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint128, complex64>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    // 处理最高 (127o, 254q) 的分子体系
    binsim::ham::HamResult<double> generate_hamiltonian_256_512_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 127)
        {
            std::cerr << "Error: norbs > 127 not supported for uint256 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint256, double>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    // 处理最高 (127o, 254q) 的周期性体系
    binsim::ham::HamResult<complex64> generate_hamiltonian_256_512_c64(
        complex64 energy_nuc,
        const complex64 *one_body_mo,
        const complex64 *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 127)
        {
            std::cerr << "Error: norbs > 127 not supported for uint256 backend."
                      << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_tmpl<uint256, complex64>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }
}
