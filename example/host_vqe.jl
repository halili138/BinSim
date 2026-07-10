ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")


function test_vqe(mole)
    basis   = BasisManager(mole)
    ham     = JW_hamiltonian(mole)
    orbs    = Orbitals(); kernel(mole, orbs, generalize=false)
    pool    = FEB(orbs)

    lv      = get_hf(basis)
    rv      = zeros(Float64, basis.dim)
    v0_idxs = findall(x -> x != 0, lv) 
    v0_vals = lv[v0_idxs]

    # mole.e_scale, _ = run_fci(basis, ham, lv)

    funcs   = OTF_Functions(basis, ham, pool)
    x0      = zeros(Float64, length(pool))
    idxs    = [i for i in eachindex(pool)]

    return run_vqe2(funcs, lv, rv, v0_idxs, v0_vals, mole.e_scale, x0, idxs, 
        VQE_OPTIONS(
            ftol      = 1e-8, 
            gtol      = 1e-6, 
            maxiter   = 999999, 
            verbose   = 1, 
            # save_path = joinpath(@__DIR__, "callback/vqe_uccsd_amplitudes_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
        )
    )
end


# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name  = ARGS[1]
#     mole.ratio = parse(Float64, ARGS[2])
#     mole.basis = ARGS[3]

#     build(mole)

#     mole.orbsym = Int64.(mole.orbsym .% 10)

#     test_vqe(mole)
# end

if abspath(PROGRAM_FILE) == @__FILE__
    es_fci   = []
    es_uccsd = []
    ds = 0.5:0.1:5
    for i in ds
        mole = Mole()
        mole.name  = "n2"
        mole.ratio = i
        mole.basis = "sto-3g"

        build(mole)

        # mole.orbsym = Int64.(mole.orbsym .% 10)
        mole.orbsym = zeros(Int64, mole.norb)

        e_opt, _, _ = test_vqe(mole)
        push!(es_uccsd, e_opt)
        push!(es_fci, mole.e_scale)
    end

    for (d, e1, e2) in zip(collect(ds), es_uccsd, es_fci)
        @printf("d: %-10.2f    uccsd: %-18.12f    fci: %-18.12f    error: %.3e\n", d, e1, e2, abs(e1-e2))
    end
end

