#include <iostream>

#include "core/bit.hpp"
#include "ham/ham.hpp"

extern "C"
{
    binsim::ham::HamResult<double> generate_hamiltonian_real_64_128_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 31)
        {
            std::cerr << "Error: norbs > 31 not supported for uint64 backend." << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_real_tmpl<uint64>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    binsim::ham::HamResult<double> generate_hamiltonian_real_128_256_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 63)
        {
            std::cerr << "Error: norbs > 63 not supported for uint128 backend." << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_real_tmpl<uint128>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }

    binsim::ham::HamResult<double> generate_hamiltonian_real_256_512_f64(
        double energy_nuc,
        const double *one_body_mo,
        const double *two_body_mo,
        int norbs,
        double tol,
        bool verbose)
    {
        if (norbs > 127)
        {
            std::cerr << "Error: norbs > 127 not supported for uint256 backend." << std::endl;
            return {nullptr, nullptr, nullptr, nullptr, nullptr, 0};
        }
        return binsim::ham::generate_hamiltonian_real_tmpl<uint256>(
            energy_nuc, one_body_mo, two_body_mo, norbs, tol, verbose);
    }
}
