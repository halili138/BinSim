include("../binsim.jl")

using PyCall
pushfirst!(pyimport("sys")."path", @__DIR__)
nesite = pyimport("nesite")

function QubitOperator(of_qubit, Ti::Type, Tv::Type)
    qubit_dict = of_qubit.terms
    nop = length(qubit_dict)
    As = Array{BinaryQubitAABB{Ti,Tv,Vector{Ti},Vector{Tv}},1}(undef, nop)
    Cs = Array{Tv,1}(undef, nop)

    for (i, (k, v)) in enumerate(qubit_dict)
        terms = Tuple{Int,String}[k...]
        As[i] = QubitOperatorAABB(terms, 1.0, Ti, Tv)
        Cs[i] = v
    end

    return linearcombine(As, Cs, 0, eps2)
end

function supf_ham_and_pool(NS, NL, U_val, T_val, ESMAX, EKMAX, uc)
    nqp = nesite.SiteQuasiPartcle(NS, NL, U_val, T_val, ESMAX, EKMAX, uc)
    ham = QubitOperator(nqp.fun, UInt32, ComplexF64)
    op_pool = [QubitOperator(op, UInt32, ComplexF64) for op in nqp.op_pool]
    r0_pool = [QubitOperator(op, UInt32, ComplexF64) for op in nqp.R0]

    return ham, op_pool, r0_pool
end

function supf_basis(N::Int)
    if 2 * N > 64
        throw(ArgumentError("N 不能超过 32。"))
    end

    dim = 1 << N
    basis = UInt64[]

    # 临时存储当前权重 k 的所有状态值 (m 和 n 的候选池)
    current_states = UInt64[]

    # 1. 遍历所有权重 k
    for k in 0:N
        empty!(current_states)

        # --- 步骤 A: 利用“对角线逻辑”收集状态 ---
        for s::UInt64 in 0:dim-1
            if count_ones(s) == k
                push!(current_states, s)
            end
        end

        length(current_states) == 0 && continue

        # --- 步骤 B: 利用收集好的状态生成块 (笛卡尔积) ---
        for m in current_states
            base_idx = m << N
            for n in current_states
                # 生成非对角块的所有元素
                idx = base_idx | n
                push!(basis, idx)
            end
        end
    end

    sort!(basis)

    astrs = unique(zip_even_bit.(basis))
    bstrs = unique(zip_odd_bit.(basis))

    norb = N
    orbsym = ones(Int64, norb)

    basis_manager = BasisManager(norb, astrs, bstrs, orbsym)

    return basis_manager
end

function get_v0_supf(basis::BasisManager, r0_pool::Vector{<:BinaryQubitAABB})
    v = zeros(ComplexF64, basis.dim)
    Hv = zeros(ComplexF64, basis.dim)
    v[1] = 1

    for r0 in r0_pool
        r0_otf = OTF(basis, r0)
        hvec_otf!(basis, r0_otf, v, Hv)
        v .= Hv
    end

    normalize!(Hv)

    return Hv
end

if abspath(PROGRAM_FILE) == @__FILE__
    NS = 2
    NL = 2
    nq = NS + 2 * NL
    ham, pool, r0_pool = supf_ham_and_pool(NS, NL, 10, 10, 1, 5, 10)
    ham = ham' * ham
    basis = supf_basis(nq)

    v0 = get_v0_supf(basis, r0_pool)

    # run_fci(basis, ham, v0, net="otf")

    # run_vqe(basis, ham, pool, v0, 0.0, net="agg",
    #     options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-6,
    #         maxiter=100000,
    #         verbose=1),
    # )

    # x0, idxs0 = load_idxs("1.jld2")

    run_adapt_vqe(basis, ham, pool, v0, 0.0, net="agg",
        # amplitudes = x0,
        # selec_idxs = idxs0,
        adapt_options=ADAPT_OPTIONS(
            Gtol=1e-3,
            gtol=1e-4,
            htol=1e-3,
            Δtol=1e-18,
            save_path="1.jld2"
        ),
        vqe_options=VQE_OPTIONS(
            ftol=1e-14,
            gtol=1e-10,
            maxiter=1000,
            verbose=1,
        ),
    )
end

