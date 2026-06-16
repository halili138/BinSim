ENV["OMP_NUM_THREADS"] = 8
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../jl/binsim.jl")

function pvqd_fig3_test(nq::Int, nL::Int)
    println("============================================================================")
    println("--- Fig from: doi: https://doi.org/10.22331/q-2021-07-28-512 ---")
    println("============================================================================")

    Ti = UInt32
    Tv = ComplexF64
    
    norb = nq ÷ 2 
    N = 1 << norb - 1
    astrs = [UInt32(i) for i in 0:N]
    bstrs = [UInt32(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))

    ref_astrs::Vector{UInt32} = [0]
    ref_bstrs::Vector{UInt32} = [0]
    ref_vals::Vector{Tv}      = [1]
    v0 = get_reference_state(basis, ref_astrs, ref_bstrs, ref_vals)
    normalize!(v0)

    J  = 0.25
    h  = 1.0
    ham = ising_module(nq, J, h, Tv=Tv, is_pbc=false)

    pool = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    for l in 1:nL
        alpha = (l % 2 != 0) ? "X" : "Y"
        for i in 0:nq-1
            push!(pool, QubitOperatorAABB([(i, alpha)], -im, Ti, Tv))
        end
        for i in 0:nq-2
            push!(pool, QubitOperatorAABB([(i, "Z"), (i+1, "Z")], -im, Ti, Tv))
        end
    end

    ops_mx = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    ops_mz = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    for i in 0:nq-1
        push!(ops_mx, QubitOperatorAABB([(i, "X")], 1.0 / nq, Ti, Tv))
        push!(ops_mz, QubitOperatorAABB([(i, "Z")], 1.0 / nq, Ti, Tv))
    end
    obs_X = linearcombine(ops_mx, ones(Tv, nq), 0.0, 1e-12)
    obs_Z = linearcombine(ops_mz, ones(Tv, nq), 0.0, 1e-12)

    run_vqrte_tdva(basis, ham, pool, v0, obs_X, obs_Z,
        dt=0.05,
        max_step=40,
        per_print=1
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    pvqd_fig3_test(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
end
