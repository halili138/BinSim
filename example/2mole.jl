include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name = ARGS[1]
#     mole.ratio = parse(Float64, ARGS[2])
#     mole.basis = ARGS[3]

#     build(mole)

#     Tv::DataType = ComplexF64

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham = JW_hamiltonian(mole)
#     ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, Tv.(ham.cs))
#     mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec, Tv=Tv))

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=false)
#     pool = FEB(orbs, Tv=Tv, complete = Tv <: Complex ? true : false)

#     # run_vqrte_tfim_forward(
#     #     basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
#     #     max_step = 1000,
#     #     per_print = 100
#     # )

#     run_vqrte_tfim_forward(
#         basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
#         max_step = 1000,
#         per_print = 100
#     )


#     # run_vqte(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
#     #     options = TimeEvolOptions(
#     #         dt          = 1e-2, 
#     #         maxiter     = 10, 
#     #         xtol        = 1e-6, 
#     #         mtol        = 1e-4, 
#     #         per_print   = 100, 
#     #         verbose     = 1, 
#     #         run_rk4     = true, 
#     #         method      = "rte", 
#     #         mode        = "forward", 
#     #         M_order     = "exact",
#     #     )
#     # )

#     # run_adapt_vqte(basis, ham, pool, get_hf(basis, mole.nelec, Tv=Tv), mole.e_scale,
#     #     adapt_options = ADAPT_OPTIONS(
#     #         maxiter     = 100, 
#     #         Gtol        = 1e-3, 
#     #         gtol        = 1e-4, 
#     #         htol        = 1e-2, 
#     #         Δtol        = 1e-8, 
#     #         verbose     = 2, 
#     #         save_path   = "",
#     #     ),
#     #     vqte_options = TimeEvolOptions(
#     #         dt          = 1e-2, 
#     #         maxiter     = 9999, 
#     #         xtol        = 1e-6, 
#     #         mtol        = 1e-4, 
#     #         per_print   = 1000, 
#     #         verbose     = 2, 
#     #         run_rk4     = false, 
#     #         method      = "ite", 
#     #         mode        = "adjoint", 
#     #         M_order     = "diag",
#     #     ),
#     # )

# end

# ham = ising_module(10, Ti=UInt32, Tv=ComplexF64, is_pbc=true)
#     ham_otf = OTF(basis, ham)
    
function ising_pool(nq::Int64; 
    Ti::DataType=UInt32, Tv::DataType=Float64, 
    is_pbc::Bool=false, pool_type::String="local")
    
    # 算符池是一个由单一 Pauli 字符串构成的数组，不需要 linearcombine
    pool = BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}}[]
    
    # ==========================================
    # 1. 单体算符 (Weight-1)
    # 必须包含 1 个 Y。由 [ZZ, X] 原始对易子产生。
    # ==========================================
    for i in 0:nq-1
        push!(pool, QubitOperatorAABB([(i, "Y")], 1.0, Ti, Tv))
    end

    # ==========================================
    # 2. 双体算符 (Weight-2)
    # 必须包含 1 个 Y 和 1 个非 Y (Z 或 X)，保证总 Y 数量为奇数
    # ==========================================
    
    # 根据用户选择，决定是只用近邻(local)还是全连接(all2all)
    pairs = Tuple{Int, Int}[]
    if pool_type == "local"
        for i in 0:nq-2
            push!(pairs, (i, i+1))
        end
        if is_pbc
            push!(pairs, (nq-1, 0))
        end
    elseif pool_type == "all2all"
        for i in 0:nq-1
            for j in i+1:nq-1
                push!(pairs, (i, j))
            end
        end
    end

    for (i, j) in pairs
        # ZY 和 YZ 组合：通常在 TFIM 中贡献最大的双体梯度
        push!(pool, QubitOperatorAABB([(i, "Z"), (j, "Y")], 1.0, Ti, Tv))
        push!(pool, QubitOperatorAABB([(i, "Y"), (j, "Z")], 1.0, Ti, Tv))
        
        # XY 和 YX 组合：为了进一步增加算符池的表达能力 (过完备性补充)
        push!(pool, QubitOperatorAABB([(i, "X"), (j, "Y")], 1.0, Ti, Tv))
        push!(pool, QubitOperatorAABB([(i, "Y"), (j, "X")], 1.0, Ti, Tv))
    end

    println("Size of $(pool_type) operator pool: $(length(pool))")

    return pool
end

if abspath(PROGRAM_FILE) == @__FILE__
    nq    = parse(Int64, ARGS[1])
    J     = 1.0
    h     = 0.5
    Tv    = ComplexF64
    norb  = nq ÷ 2
    astrs = [UInt32(i) for i in 0:(1 << norb - 1)]
    bstrs = [UInt32(i) for i in 0:(1 << norb - 1)]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))
    ham   = ising_module(nq, J, h, Tv=Tv, is_pbc=true)
    e_fci, v_fci = run_fci(basis, ham, normalize!(ones(Tv, basis.dim)))
    pool  = ising_pool(nq, Tv=Tv, is_pbc=true, pool_type="local")

    ham_quench = ising_module(nq, J, 0.6, Tv=Tv, is_pbc=true)
    run_fci(basis, ham_quench, normalize!(ones(Tv, basis.dim)))

    run_vqrte_forward(
        basis, ham_quench, pool, v_fci, e_fci,
        max_step = 10,
        per_print = 1
    )
    # run_adapt_vqrte_tfim_forward(
    #     basis, ham_quench, pool, v_fci, e_fci,
    #     per_print = 100
    # )
end

