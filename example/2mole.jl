include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name = ARGS[1]
#     mole.ratio = parse(Float64, ARGS[2])
#     mole.basis = ARGS[3]

#     build(mole)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham = JW_hamiltonian(mole)
#     ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, ComplexF64.(ham.cs))
#     mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=ComplexF64))

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=false)
#     pool = FEB(orbs, Tv=ComplexF64, complete=true)

#     run_vqrte_tfim_adjoint(
#         basis, ham, pool, get_hf(basis, mole.nelec, Tv=ComplexF64), mole.e_scale, 
#         max_step=1000, per_print=40)
# end



if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    # run_vqite_tfim_forward(
    #     basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     max_step=1000, per_print=40)

    run_adapt_vqite_tfim_adjoint(
        basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
        dt=0.02, max_adapt_step=100, max_inner_step=10000, Gtol=1e-3, xtol=1e-6, verbose=2, inner_per_print=100)

    # run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     adapt_options=ADAPT_OPTIONS(
    #         Gtol=1e-3,
    #         gtol=1e-4,
    #         htol=1e-3,
    #         Δtol=1e-8,
    #         verbose=1,
    #     ),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=10000,
    #         verbose=1,
    #     )
    # )
end
