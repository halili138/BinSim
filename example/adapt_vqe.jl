include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    # amplitudes, selec_idxs = load_idxs("save_temp.jld2")

    run_adapt_vqe(mole;
                #   amplitudes = amplitudes,  
                #   selec_idxs = selec_idxs,
                  adapt_options=ADAPT_OPTIONS(
                    Gtol=1e-3,
                    gtol=1e-4,
                    htol=1e-3,
                    Δtol=1e-8,
                    # save_path="save_temp.jld2",
                    ),
                  vqe_options=VQE_OPTIONS(
                    ftol=1e-8,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )
end

