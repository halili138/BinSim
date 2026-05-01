include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    pbc = Pbc()
    pbc.name   = "1d-h"
    pbc.ratio  = 1.0
    pbc.basis  = "gth-szv"
    pbc.pseudo = "gth-pade"
    pbc.mesh   = [4,1,1]
    pbc.scaled_center = [0,0,0]

    build(pbc)

    pbc.e_scale = -0.90257763392980
    pbc.orbsym  = ones(Int64, pbc.norb)
    
    # amplitudes, selec_idxs = load_idxs("save_temp.jld2")

    run_adapt_vqe(pbc;
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
                    ftol=1e-10,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )
end

