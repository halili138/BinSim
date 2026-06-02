include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    Tv::DataType = ComplexF64

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs, Tv=Tv, complete = Tv <: Complex ? true : false)

    run_vqte(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
        options=TimeEvolOptions(
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

    # run_adapt_vqite_tfim_adjoint(
    #     basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     dt=0.01, max_adapt_step=100, max_inner_step=99999, Gtol=1e-3, xtol=1e-6, verbose=2, inner_per_print=10000,
    #     is_diag=true, run_rk4=true)

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
