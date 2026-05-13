include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    k = 5
    e_scales, _ = run_fci(basis, ham; k=k)

    orbs = Orbitals()
    kernel(mole, orbs, generalize=true)
    pool = FEB(orbs)

    v0s, weights, _ = generate_ssvqe_inputs(basis, ham, k_states=k)

    run_ssvqe(basis, ham, pool, v0s, weights, e_scales,
        options=VQE_OPTIONS(
            ftol=1e-8,
            gtol=1e-6,
            maxiter=100000,
            verbose=2)
    )

    # run_adapt_ssvqe(basis, ham, pool, v0s, weights, e_scales,
    #     adapt_options=ADAPT_OPTIONS(
    #         Gtol=1e-3,
    #         gtol=1e-4,
    #         htol=1e-3,
    #         Δtol=1e-8,
    #     ),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-8,
    #         gtol=1e-6,
    #         maxiter=1000,
    #         verbose=2,
    #     ),
    # )
end
