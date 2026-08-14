ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 1)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/cubinsim.jl")


function test_adapt_vqe(mole)
    basis       = BasisManager(mole)
    ham         = JW_hamiltonian(mole)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=true)
    pool        = FEB(orbs)

    h_v0        = get_hf(basis)
    h_v0_idxs   = findall(x -> x != 0, h_v0) 
    h_v0_vals   = h_v0[h_v0_idxs]
    d_v0_idxs   = CuArray{Int64,1,CUDA.DeviceMemory}(h_v0_idxs)
    d_v0_vals   = CuArray{Float64,1,CUDA.DeviceMemory}(h_v0_vals)

    d_lv        = CUDA.zeros(Float64, basis.dim)
    d_rv        = CUDA.zeros(Float64, basis.dim)

    h_funcs     = OTF_Functions(basis, ham, pool)
    d_funcs     = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool)
    nparams     = length(pool)
    x0          = Float64[]
    idxs        = Int64[]

    run_adapt_vqe(d_funcs, d_lv, d_rv, d_v0_idxs, d_v0_vals, mole.e_scale, nparams, x0, idxs, 
        ADAPT_OPTIONS(
            maxiter   = 99999, 
            Gtol      = 1.0, 
            gtol      = 1.0, 
            htol      = 1e-3, 
            Δtol      = 1e-10, 
            verbose   = 3, 
            save_path = joinpath(@__DIR__, "callback/adapt_uccgsd_amplitudes_selec_idxs_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2"),
        ), 
        VQE_OPTIONS(
            ftol      = 1e-10, 
            gtol      = 1e-6, 
            maxiter   = 9999, 
            verbose   = 3, 
            save_path = "",
        )
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    test_adapt_vqe(mole)
end


