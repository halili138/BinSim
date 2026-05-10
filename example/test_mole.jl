include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    v0 = get_hf(basis, mole.nelec)

    mole.e_scale, _ = run_fci(basis, ham, v0, net="otf")

    v0 = get_hf(basis, mole.nelec)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    e_opt, x_opt, v_opt = run_vqe(basis, ham, pool, v0, mole.e_scale, net="otf",
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-6,
            maxiter=100000,
            verbose=1),
    )

    kernel(mole, orbs, excited_order=4)
    pool = FEB(orbs)

    # run_enpt2(basis, ham, v_opt, mole.e_scale, net="otf")
    # run_qse(basis, ham, pool, v_opt, mole.e_scale)

    # run_adapt_vqe(basis, ham, pool, v0, mole.e_scale, net="agg",
    #     adapt_options=ADAPT_OPTIONS(
    #         Gtol=1e-3,
    #         gtol=1e-4,
    #         htol=1e-3,
    #         Δtol=1e-8,
    #     ),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=1000,
    #         verbose=1,
    #     ),
    # )
end
