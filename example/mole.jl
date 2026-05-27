include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-8,
            maxiter=100000,
            verbose=2,
        )
    )
    # run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     adapt_options=ADAPT_OPTIONS(verbose=1),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=1000,
    #         verbose=0,
    #     )
    # )

    # run_exact_vqe_adaptive(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-8,
    #         maxiter=100000,
    #         verbose=2,
    #         # save_path=joinpath(@__DIR__, "callback/exact_vqe_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
    #     ),
    # )
end


# include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()

#     filepath  = "$(ARGS[1])-1.0-$(ARGS[2])-sto-3g.jld2"
#     name      = splitext(basename(filepath))[1]
#     save_path = joinpath(@__DIR__, "callback/exact_vqe_$(name).jld2")

#     build(mole, filepath)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham   = JW_hamiltonian(mole)
#     v0    = get_hf(basis, mole.nelec)

#     e_scales::Dict{String,Float64} = Dict(
#         "0.0"   => -78.44726910513450,
#         "10.0"  => -78.44761142358699,
#         "20.0"  => -78.44854306751958,
#         "30.0"  => -78.44980728814951,
#         "40.0"  => -78.45106186866273,
#         "50.0"  => -78.45197423148180,
#         "60.0"  => -78.45230690893402,
#         "70.0"  => -78.45197423148208,
#         "80.0"  => -78.45106186866256,
#         "90.0"  => -78.44980728814940,
#         "100.0" => -78.44854306751976,
#         "110.0" => -78.44761142358695,
#         "120.0" => -78.44726910513467,
#     )

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=false)
#     pool = FEB(orbs)

#     run_exact_vqe_adaptive(basis, ham, pool, v0, e_scales[ARGS[2]],
#         options=VQE_OPTIONS(
#             ftol=1e-10,
#             gtol=1e-8,
#             maxiter=100000,
#             verbose=2,
#             save_path=save_path,
#         ),
#     )
# end



