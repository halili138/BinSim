include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    Tv::DataType = Float64

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs, Tv=Tv, complete = Tv <: Complex ? true : false)

    run_vqte(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
        options = TimeEvolOptions(
            dt          = 1e-2, 
            maxiter     = 9999, 
            xtol        = 1e-6, 
            mtol        = 1e-4, 
            per_print   = 100, 
            verbose     = 1, 
            run_rk4     = true, 
            method      = "ite", 
            mode        = "forward", 
            M_order     = "exact",
        )
    )

    run_adapt_vqte(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
        adapt_options = ADAPT_OPTIONS(
            maxiter     = 100, 
            Gtol        = 1e-3, 
            gtol        = 1e-4, 
            htol        = 1e-2, 
            Δtol        = 1e-8, 
            verbose     = 2, 
            save_path   = "",
        ),
        vqte_options = TimeEvolOptions(
            dt          = 1e-2, 
            maxiter     = 9999, 
            xtol        = 1e-6, 
            mtol        = 1e-4, 
            per_print   = 1000, 
            verbose     = 2, 
            run_rk4     = false, 
            method      = "ite", 
            mode        = "adjoint", 
            M_order     = "diag",
        ),
    )
end
