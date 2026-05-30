include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, ComplexF64.(ham.cs))
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=ComplexF64))

    orbs = Orbitals()
    kernel(mole, orbs, generalize=true)
    pool = FEB(orbs, Tv=ComplexF64, complete=true)

    # run_exact_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-8,
    #         maxiter=100000,
    #         verbose=2,
    #     )
    # )

    run_vqrte_tfim_native(basis, ham, pool, get_hf(basis, mole.nelec, Tv=ComplexF64), mole.e_scale, max_step=2000, per_print=40)
end


