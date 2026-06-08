include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    e_fci, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)

    run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci,
        options = VQE_OPTIONS(
            ftol      = 1e-8, 
            gtol      = 1e-6, 
            maxiter   = 1000, 
            verbose   = 2, 
            save_path = joinpath(@__DIR__, "callback/vqe_uccsd_amplitudes_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
        )
    )
end
