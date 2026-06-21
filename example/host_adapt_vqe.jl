ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")


function test_adapt_vqe(mole)
    basis   = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham     = JW_hamiltonian(mole)
    orbs    = Orbitals(); kernel(mole, orbs, generalize=false)
    pool    = FEB(orbs)

    lv      = get_hf(basis, mole.nelec)
    rv      = zeros(Float64, basis.dim)
    v0_idxs = findall(x -> x != 0, lv) 
    v0_vals = lv[v0_idxs]

    mole.e_scale, _ = run_fci(basis, ham, lv)

    funcs   = OTF_Functions(basis, ham, pool)
    nparams = length(pool)
    x0      = Float64[]
    idxs    = Int64[]

    run_adapt_vqe2(funcs, lv, rv, v0_idxs, v0_vals, mole.e_scale, nparams, x0, idxs, 
        ADAPT_OPTIONS(
            maxiter   = 99999, 
            Gtol      = 1e-3, 
            gtol      = 1e-4, 
            htol      = 1e-3, 
            Δtol      = 1e-8, 
            verbose   = 3, 
            save_path = "",
        ), 
        VQE_OPTIONS(
            ftol      = 1e-8, 
            gtol      = 1e-6, 
            maxiter   = 9999, 
            verbose   = 1, 
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

    mole.orbsym = Int64.(mole.orbsym .% 10)

    test_adapt_vqe(mole)
end

