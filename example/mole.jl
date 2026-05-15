include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec), net="otf")

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    run_exact_vqe_adaptive(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-8,
            maxiter=100000,
            verbose=2,
            save_path=joinpath(@__DIR__, "callback/exact_vqe_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
        ),
    )
end

