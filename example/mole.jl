include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()

#     filepath  = "$(ARGS[1])-1.0-$(ARGS[2])-sto-3g.jld2"
#     name      = splitext(basename(filepath))[1]
#     save_path = joinpath(@__DIR__, "callback/adapt_idxs_$(name).jld2")

#     build(mole, filepath)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham   = JW_hamiltonian(mole)
#     v0    = get_hf(basis, mole.nelec)

#     # e_scales::Dict{String,Float64} = Dict(
#     #     "0.0"   => -78.44726910513450,
#     #     "10.0"  => -78.44761142358699,
#     #     "20.0"  => -78.44854306751958,
#     #     "30.0"  => -78.44980728814951,
#     #     "40.0"  => -78.45106186866273,
#     #     "50.0"  => -78.45197423148180,
#     #     "60.0"  => -78.45230690893402,
#     #     "70.0"  => -78.45197423148208,
#     #     "80.0"  => -78.45106186866256,
#     #     "90.0"  => -78.44980728814940,
#     #     "100.0" => -78.44854306751976,
#     #     "110.0" => -78.44761142358695,
#     #     "120.0" => -78.44726910513467,
#     # )

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=true)
#     pool = FEB(orbs)

#     run_adapt_vqe(basis, ham, pool, v0, mole.e_scale, net=ARGS[3],
#         adapt_options=ADAPT_OPTIONS(
#             Gtol=1e-3,
#             gtol=1e-4,
#             htol=1e-3,
#             Δtol=1e-12,
#             save_path=save_path,
#         ),
#         vqe_options=VQE_OPTIONS(
#             ftol=1e-10,
#             gtol=1e-8,
#             maxiter=1000,
#             verbose=3,
#         ),
#     )
# end



if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec), net="otf")

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    # run_exact_vqe_adaptive(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale)

    e_opt, v_opt, x_opt = run_vqe(
        basis, 
        ham, pool, 
        get_hf(basis, mole.nelec), 
        mole.e_scale,
        net="otf",
        options=VQE_OPTIONS(
            ftol=1e-10,
            gtol=1e-6,
            maxiter=100000,
            verbose=2),
    )
end

