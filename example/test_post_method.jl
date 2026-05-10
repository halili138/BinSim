include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    
    e_scales, _ = run_krylovkit_diag(basis, ham; k=5)

    v0 = get_hf(basis, mole.nelec)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    e_opt, x_opt, v_opt = run_vqe(basis, ham, pool, v0, e_scales[1], net="otf",
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-6,
            maxiter=100000,
            verbose=1),
    )

    # run_enpt2(basis, ham, v_opt, mole.e_scale, net="otf")

    kernel(mole, orbs, excited_order=3)
    pool = FEB(orbs)

    run_qse(basis, ham, pool, v_opt, e_scales=e_scales, n_states=length(e_scales))
    run_qeom(basis, ham, pool, v_opt, e_scales=e_scales, n_states=length(e_scales))
end
