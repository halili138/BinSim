include("../binsim.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    pbc = Pbc()
    pbc.name   = "1d-h"
    pbc.ratio  = 1.0
    pbc.basis  = "gth-szv"
    pbc.pseudo = "gth-pade"
    pbc.mesh   = [3,1,1]
    pbc.scaled_center = [0,0,0]

    build(pbc)

    pbc.e_scale = -1.0601859104945792
    pbc.orbsym = ones(Int64, pbc.norb)

    run_vqe(pbc,
            options=VQE_OPTIONS(
                ftol=1e-10,
                gtol=1e-6,
                maxiter=100000,
                verbose=1,
            ),
        )
end

