ENV["OMP_NUM_THREADS"] = 8
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../jl/binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    
    e_fcis, _ = run_fci(basis, ham; k=5)

    v0 = get_hf(basis, mole.nelec)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs) # Using UCCSD operator 

    e_opt, v_opt, x_opt = run_vqe(basis, ham, pool, v0, e_fcis[1],
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-6,
            maxiter=100000,
            verbose=1),
    )

    run_enpt2(basis, ham, v_opt, e_fcis[1])

    kernel(mole, orbs, excited_order=3, generalize=false)
    pool = FEB(orbs) # Using UCCSDT operator 

    run_qse(basis, ham, pool, v_opt, e_scales=e_fcis, n_states=length(e_fcis))
    run_qeom(basis, ham, pool, v_opt, e_scales=e_fcis, n_states=length(e_fcis))
end
