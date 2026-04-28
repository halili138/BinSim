include("../binsim.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    # x0 = load_x("save_temp.jld2")

    run_vqe(mole,
            # x0=x0,
            options=VQE_OPTIONS(
                ftol=1e-8,
                gtol=1e-6,
                maxiter=1000,
                verbose=1,
                # save_path="save_temp.jld2",
            ),
        )
end

