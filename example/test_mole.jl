include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    v0  = get_hf(basis, mole.nelec)
    mole.e_scale, _ = run_fci(basis, ham, v0, net="otf")
    
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    v0  = get_hf(basis, mole.nelec)

    # e_opt, v_opt, x_opt = run_vqe(basis, ham, pool, v0, mole.e_scale, net="otf",
    #     options=VQE_OPTIONS(
    #         ftol=1e-8,
    #         gtol=1e-6,
    #         maxiter=100000,
    #         verbose=1),
    # )

    run_adapt_vqe(basis, ham, pool, v0, mole.e_scale, net="agg",
        adapt_options=ADAPT_OPTIONS(
            Gtol=1e-3,
            gtol=1e-4,
            htol=1e-3,
            Δtol=1e-8,
        ),
        vqe_options=VQE_OPTIONS(
            ftol=1e-8,
            gtol=1e-6,
            maxiter=1000,
            verbose=0,
        ),
    )
end
