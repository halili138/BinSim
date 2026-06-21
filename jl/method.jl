function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,K,V}[], time_print=true)
    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, funcs.ham, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    println("Solving FCI with davidson ... ")
    @time e_fci, v_fci = davidson(funcs.hvec, v0, diags, tol=1e-5)
    println("")

    return e_fci, v_fci
end


function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}; k::Int=1) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,K,V}[], time_print=false)
    hvec_map = LinearMap{Tv}(
        (dst, src) -> funcs.hvec(src, dst),
        basis.dim,
        ismutating=true,
        ishermitian=true
    )

    print("Running Arpack directly on C++ OTF Network...")
    time_ops = @elapsed λ_aggs, ϕ_aggs = eigs(hvec_map, nev=k, which=:SR)
    @printf("Done in %.4f seconds\n", time_ops)

    λ_aggs = real.(λ_aggs)
    df_states = [i == 1 ? "000 (GS)" : @sprintf("%03d", i - 1) for i in 1:k]
    df_energies = [@sprintf("%.14f", e) for e in λ_aggs]

    df_step = DataFrame(
        "State" => df_states,
        "f (Energy)" => df_energies,
    )

    println("-------------------------------")
    show(stdout, df_step, summary=false, eltypes=false, show_row_number=false)
    println("\n")

    return λ_aggs, ϕ_aggs
end


function run_vqe(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv}, e_scale::Float64;
    x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS(),
) where {Ti,Tv,K,V}
    funcs   = OTF_Functions(basis, ham, pool)
    lv      = zeros(Tv, basis.dim)
    rv      = zeros(Tv, basis.dim)
    idxs    = [i for i in eachindex(pool)]

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    energy  = Ref(0.0)
    gnorm   = Ref(0.0)
    δ²H     = Ref(0.0)
    error   = Ref(0.0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        lv .= v0
        result = @timed energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, idxs, x, lv, rv)
        energy[], grads, δ²H[] = result.value
        gnorm[] = norm(grads)
        error[] = abs(energy[] - e_scale)
        options.verbose > 1 && show_optimze(energy[], gnorm[], δ²H[], error[])
        options.verbose > 2 && show_time(result)

        return energy[], grads
    end

    println("Performing VQE optimization ... ")
    time_ops = @elapsed e_opt, x_opt = optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
    @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n",
            time_ops, energy[], gnorm[], δ²H[], error[])
    println("\n")

    lv .= v0

    for i in eachindex(idxs)
        funcs.expm(idxs[i], x_opt[i], lv)
    end

    return e_opt, lv, x_opt
end


function run_vqe2(funcs, lv, rv, v0_idxs, v0_vals, e_scale::Float64, x0::Vector{Float64}, idxs::Vector{Int64}, options::VQE_OPTIONS)
    energy  = Ref(0.0)
    gnorm   = Ref(0.0)
    δ²H     = Ref(0.0)
    error   = Ref(0.0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals

        result = @timed energy_objective(
            funcs.hvec, 
            funcs.expm, 
            funcs.backgrad, 
            idxs, x, lv, rv
            )
            
        energy[], grads, δ²H[] = result.value
        gnorm[] = norm(grads)
        error[] = abs(energy[] - e_scale)
        options.verbose > 1 && show_optimze(energy[], gnorm[], δ²H[], error[])
        options.verbose > 2 && show_time(result)

        return energy[], grads
    end

    println("Performing VQE optimization ... ")
    time_ops = @elapsed e_opt, x_opt = optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
    @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n",
            time_ops, energy[], gnorm[], δ²H[], error[])
    println("\n")

    fill!(lv, 0.0)
    lv[v0_idxs] .= v0_vals

    for (i, t) in zip(idxs, x_opt)
        funcs.expm(i, t, lv)
    end

    return e_opt, lv, x_opt
end


function run_adapt_vqe(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv}, e_scale::Float64;
    amplitudes::Vector{Float64}=Float64[], selec_idxs::Vector{Int64}=Int64[], adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(), 
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, pool)
    lv    = zeros(Tv, basis.dim)
    rv    = zeros(Tv, basis.dim)
    idxs  = [i for i in eachindex(pool)]

    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    println("Performing ADAPT-VQE ... ")

    time_ops = @elapsed amplitudes, selec_idxs = _adapt_vqe(
        funcs.hvec,
        funcs.expm,
        funcs.backgrad,
        funcs.batchgrad,
        idxs,
        v0,
        lv,
        rv,
        e_scale,
        amplitudes,
        selec_idxs,
        adapt_options,
        vqe_options,
    )

    @printf("Done in %.4f seconds\n\n", time_ops)

    return amplitudes, selec_idxs
end


function krylov_expmv!(f_matvec!::Function, dst::Vector{Tv}, src::Vector{Tv};
    t::Float64=1.0, krylov_dim::Int=30, tol::Float64=1e-12,
) where Tv
    """
    Arnoldi-Krylov approximation of dst = exp(t * A) * src, where A is
    represented only by the mutating matvec callback f_matvec!(v, Av).
    This is intended for non-Trotter exact VQE evolution generated by the
    linear-combined cluster operator, so it does not assume Hermiticity.
    """
    β0 = norm(src)
    if β0 < tol
        fill!(dst, zero(Tv))
        return dst
    end

    m_max = max(1, min(krylov_dim, length(src)))
    V = [zeros(Tv, length(src)) for _ in 1:m_max]
    w = zeros(Tv, length(src))
    H = zeros(Tv, m_max + 1, m_max)

    @. V[1] = src / β0
    m_actual = m_max

    for j in 1:m_max
        f_matvec!(V[j], w)

        for i in 1:j
            H[i, j] = dot(V[i], w)
            @. w -= H[i, j] * V[i]
        end

        # One re-orthogonalization pass improves stability for long VQE pools.
        for i in 1:j
            hij = dot(V[i], w)
            H[i, j] += hij
            @. w -= hij * V[i]
        end

        h_next = norm(w)
        if j < m_max
            H[j + 1, j] = h_next
            if h_next < tol
                m_actual = j
                break
            end
            @. V[j + 1] = w / h_next
        end
    end

    Hm = Matrix(H[1:m_actual, 1:m_actual])
    e1 = zeros(Tv, m_actual)
    e1[1] = one(Tv)
    coeffs = β0 .* (exp(t .* Hm) * e1)

    fill!(dst, zero(Tv))
    for j in 1:m_actual
        @. dst += coeffs[j] * V[j]
    end

    return dst
end


function run_exact_vqe_krylov(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv}, e_scale::Float64;
    x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS(ftol=1e-10),
    krylov_dim::Int=30, krylov_tol::Float64=1e-12, fd_step::Float64=1e-6,
) where {Ti,Tv,K,V}
    println("============================================================================")
    println("--- Adaptive Exact UCC VQE (Krylov expm action + finite-difference gradients) ---")

    Hfuncs = OTF_Functions(basis, ham, pool, time_print=false)
    if !isempty(x0)
        @assert length(x0) == length(pool)
    else
        x0 = zeros(Float64, length(pool))
    end

    v_work = zeros(Tv, basis.dim)
    Hv = zeros(Tv, basis.dim)

    function build_state!(dst::Vector{Tv}, x::Vector{Float64})
        if norm(x) < 1e-12
            copyto!(dst, v0)
            normalize!(dst)
            return dst
        end

        T = linearcombine(pool, x, 0.0, 1e-12)
        Tfuncs = OTF_Functions(basis, T, BinaryQubitAABB{Ti,Tv,K,V}[], info_print=false, time_print=false)
        krylov_expmv!(Tfuncs.hvec, dst, v0, t=1.0, krylov_dim=krylov_dim, tol=krylov_tol)
        normalize!(dst)
        return dst
    end

    function energy_at(x::Vector{Float64})
        build_state!(v_work, x)
        Hfuncs.hvec(v_work, Hv)
        E = real(dot(v_work, Hv))
        δ²H = max(0.0, norm(Hv) ^ 2 / norm(v_work) ^ 2 - E ^ 2)
        return E, δ²H
    end

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        E, δ²H = energy_at(x)
        grads = zeros(Float64, length(x))

        xp = copy(x)
        xm = copy(x)
        for i in eachindex(x)
            h = fd_step * max(1.0, abs(x[i]))
            xp[i] = x[i] + h
            xm[i] = x[i] - h
            Ep, _ = energy_at(xp)
            Em, _ = energy_at(xm)
            grads[i] = (Ep - Em) / (2h)
            xp[i] = x[i]
            xm[i] = x[i]
        end

        err = abs(E - e_scale)
        options.verbose > 0 && show_optimze(E, norm(grads), δ²H, err)
        return E, grads
    end

    time_ops = @elapsed e_opt, x_opt = optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
    build_state!(v_work, x_opt)
    @printf("Converged in %.4f seconds with: f = %.14f  err = %.3e\n\n", time_ops, e_opt, abs(e_opt - e_scale))

    return e_opt, copy(v_work), x_opt
end


function run_exact_vqe(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv}, e_scale::Float64;
    x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS(ftol=1e-10), ode_tol::Float64=1e-8, # ODE 积分精度
) where {Ti,Tv,K,V}
    println("============================================================================")
    println("--- Adaptive Exact UCC VQE (Augmented ODE Adjoint Method) ---")

    Hfuncs = OTF_Functions(basis, ham, pool, time_print=false)

    if !isempty(x0)
        @assert length(x0) == length(pool)
    else
        x0 = zeros(Float64, length(pool))
    end

    Hv = zeros(Tv, basis.dim)
    a  = zeros(Tv, basis.dim)
    gt = zeros(Tv, length(pool))

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        if norm(x) < 1e-12
            Hfuncs.hvec(v0, Hv)
            E = real(dot(v0, Hv))
            err = abs(E - e_scale)
            δ²H = max(0.0, norm(Hv) ^ 2 / norm(v0) ^ 2 - E ^ 2)

            @. a = 2.0 * Hv
            Hfuncs.batchtran(v0, a, gt)

            options.verbose > 0 && show_optimze(E, norm(gt), δ²H, err)

            return E, real.(gt)
        end

        T = linearcombine(pool, x, 0.0, 1e-12)
        Tfuncs = OTF_Functions(basis, T, BinaryQubitAABB{Ti,Tv,K,V}[], info_print=false, time_print=false)
        f_forward! = (du, u, p, s) -> Tfuncs.hvec(u, du)
        prob_fwd = ODEProblem(f_forward!, v0, (0.0, 1.0))
        sol_fwd = solve(prob_fwd, Tsit5(), abstol=ode_tol, reltol=ode_tol, save_everystep=false)

        v_final = sol_fwd.u[end]
        normalize!(v_final)

        Hfuncs.hvec(v_final, Hv)
        E   = real(dot(v_final, Hv))
        err = abs(E - e_scale)
        δ²H = max(0.0, norm(Hv) ^ 2 / norm(v_final) ^ 2 - E ^ 2)

        @. a = 2.0 * Hv
        g_init = zeros(Float64, length(pool))
        u_back_init = ArrayPartition(v_final, a, g_init)
        f_backward! = (du, u, p, s) -> begin
            ψ_curr = u.x[1]
            a_curr = u.x[2]
            dψ = du.x[1]
            da = du.x[2]
            dg = du.x[3]

            Tfuncs.hvec(ψ_curr, dψ)
            Tfuncs.hvec(a_curr, da)

            Hfuncs.batchtran(ψ_curr, a_curr, gt)

            @. dg = -real(gt)
        end

        prob_bwd = ODEProblem(f_backward!, u_back_init, (1.0, 0.0))
        sol_bwd = solve(prob_bwd, Tsit5(), abstol=ode_tol, reltol=ode_tol, save_everystep=false)
        g_tot = sol_bwd.u[end].x[3]

        options.verbose > 0 && show_optimze(E, norm(g_tot), δ²H, err)

        return E, g_tot
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end


function run_enpt2(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}, e_scale::Float64;
    ref_tol::Float64=1e-4, level_shift::Float64=0.0
) where {Ti,Tv,K,V}
    """
    Epstein-Nesbet 二阶微扰理论 (ENPT2) 后处理校正
    利用已收敛的近似波函数 v0, 计算残差并估计动态相关能。

    公式: E^(2) = sum_{i ∉ ref} |<i|H - E0|v0>|^2 / (E0 - H_ii - shift)
    """
    println("\n--- Starting ENPT2 Post-Processing ---")

    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,K,V}[], time_print=false)
    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, funcs.ham, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    # 确保参考态已经归一化
    v = copy(v0)
    normalize!(v)

    w = zeros(Tv, basis.dim)

    # 1. 计算 H|v0> 
    funcs.hvec(v, w)

    # 2. 计算零阶能量 E0 = <v0|H|v0>
    E0 = real(dot(v, w))

    # 3. 计算残差向量 |r> = (H - E0)|v0>
    # 注意此时 w 存储的是 H|v0>
    r = zeros(Tv, basis.dim)
    @. r = w - E0 * v

    # 4. 计算二阶能量校正 E2
    E2 = 0.0
    diverge_count = 0
    ref_size = 0

    for i in 1:basis.dim
        # 判断当前行列式是否在参考空间外（权重系数极小）
        if abs(v[i]) < ref_tol
            # 计算分母：E0 - H_ii - shift
            denominator = E0 - diags[i] - level_shift

            # 防止闯入态问题 (Intruder state problem), 分母必须为负且有一定大小
            if denominator < -1e-6
                E2 += abs2(r[i]) / denominator
            else
                diverge_count += 1
            end
        else
            ref_size += 1
        end
    end

    @printf("\n  Reference space size : %d / %d (tol = %.1e)\n", ref_size, basis.dim, ref_tol)
    @printf("  Zero-order E0        : %.14f\n", E0)
    @printf("  PT2 Correction E2    : %.14f\n", E2)
    @printf("  Total Energy (E0+E2) : %.14f\n", E0 + E2)
    @printf("  Error                : %.4e\n", abs(E0 + E2 - e_scale))

    if diverge_count > 0
        @printf("  Warning: %d states ignored due to denominator > -1e-6 (Intruder states)\n", diverge_count)
    end
    println("--------------------------------------\n")

    return E0, E2
end


function run_qse(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv};
    e_scales::Vector{Float64}=[], S_tol::Float64=1e-8, n_states::Int=5,
) where {Ti,Tv,K,V}
    println("\n--- Starting Quantum Subspace Expansion (QSE) ---\n")

    funcs = OTF_Functions(basis, ham, pool, time_print=false)

    get_V! = (i, dst) -> begin
        if i == 1
            copyto!(dst, v0)
        else
            funcs.tvec(i - 1, v0, dst)
        end
    end

    N_sub = length(pool) + 1
    print("Building S and H Matrices ... ")
    vi = zeros(Tv, basis.dim)
    vj = zeros(Tv, basis.dim)
    Hv = zeros(Tv, basis.dim)
    S_mat = zeros(Tv, N_sub, N_sub)
    H_mat = zeros(Tv, N_sub, N_sub)

    time_ops = @elapsed begin
        for j in 1:N_sub
            get_V!(j, vj)
            funcs.hvec(vj, Hv)
            for i in 1:j
                if i == j
                    S_mat[j, j] = real(dot(vj, vj))
                    H_mat[j, j] = real(dot(vj, Hv))
                else
                    get_V!(i, vi)
                    s_val = real(dot(vi, vj))
                    h_val = real(dot(vi, Hv))
                    S_mat[i, j] = s_val
                    S_mat[j, i] = s_val
                    H_mat[i, j] = h_val
                    H_mat[j, i] = h_val
                end
            end
        end
    end

    S_mat = Hermitian(S_mat)
    H_mat = Hermitian(H_mat)
    @printf("Done in %.4f seconds\n", time_ops)

    # 5. Canonical Orthogonalization (正则正交化消除线性相关)
    λ_S, U_S = eigen(S_mat)

    valid_idx = findall(x -> x > S_tol, λ_S)
    N_valid = length(valid_idx)
    @printf("Conditioning S matrix: %d / %d basis vectors kept (S_tol = %.1e)\n\n", N_valid, N_sub, S_tol)

    if N_valid == 0
        error("No linearly independent basis vectors found. Try reducing S_tol.")
    end

    λ_S_valid = λ_S[valid_idx]
    U_S_valid = U_S[:, valid_idx]

    # 正交化转换矩阵
    X = U_S_valid * Diagonal(1.0 ./ sqrt.(λ_S_valid))

    # 6. 将哈密顿量转换到正交基底并对角化 H_orth = X^† * H * X
    H_orth = Hermitian(X' * H_mat * X)
    E_qse, C_orth = eigen(H_orth)

    # 7. 映射回原 QSE 基底的系数
    C_qse = X * C_orth

    n_print = min(n_states, N_valid)
    states = [i == 1 ? "000 (GS)" : @sprintf("%03d", i - 1) for i in 1:n_print]
    energies = [@sprintf("%.15f", E_qse[i]) for i in 1:n_print]
    delta_es = [i == 1 ? "-" : @sprintf("%.15f", E_qse[i] - E_qse[1]) for i in 1:n_print]
    errors = [i <= length(e_scales) ? @sprintf("%.3e", abs(e_scales[i] - E_qse[i])) : "-" for i in 1:n_print]
    df = DataFrame(
        "State" => states,
        "Energy" => energies,
        "ΔE (vs Ground)" => delta_es,
        "Error" => errors
    )

    show(stdout, df, summary=false, eltypes=false, show_row_number=false)
    println("\n")

    return E_qse, C_qse
end


function run_qeom(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0::Vector{Tv};
    e_scales::Vector{Float64}=[], S_tol::Float64=1e-8, n_states::Int=5,
) where {Ti,Tv,K,V}
    println("\n--- Starting Quantum Equation-of-Motion (qEOM) ---\n")

    funcs = OTF_Functions(basis, ham, pool, time_print=false)

    N_pool = length(pool)
    # 2. 准备 qEOM 核心的辅助态
    v0_tilde = zeros(Tv, basis.dim)
    funcs.hvec(v0, v0_tilde)            # |v0_tilde> = H |v0>

    E_ref = real(dot(v0, v0_tilde)) # VQE 参考态能量

    print("Building qEOM M and S Matrices ... ")
    time_ops = @elapsed begin
        v_i = zeros(Tv, basis.dim)
        v_i_tilde = zeros(Tv, basis.dim)
        v_j = zeros(Tv, basis.dim)
        v_j_tilde = zeros(Tv, basis.dim)
        hv_j = zeros(Tv, basis.dim)
        S_mat = zeros(Tv, N_pool, N_pool)
        M_mat = zeros(Tv, N_pool, N_pool)

        for j in 1:N_pool
            funcs.tvec(j, v0, v_j)               # |v_j> = O_j |v0>
            funcs.tvec(j, v0_tilde, v_j_tilde)   # |v_j_tilde> = O_j |v0_tilde> = O_j H |v0>
            funcs.hvec(v_j, hv_j)                     # H |v_j>
            for i in 1:j
                funcs.tvec(i, v0, v_i)
                funcs.tvec(i, v0_tilde, v_i_tilde)
                S_val = real(dot(v_i, v_j))
                term1 = real(dot(v_i, hv_j))       # <v_i | H | v_j>
                term2 = real(dot(v_i_tilde, v_j))  # <v_i_tilde | v_j>
                term3 = real(dot(v_i, v_j_tilde))  # <v_i | v_j_tilde>

                # qEOM 核心公式：M_ij = term1 - 0.5 * term2 - 0.5 * term3
                M_val = term1 - 0.5 * term2 - 0.5 * term3

                # 对称赋值
                S_mat[i, j] = S_val
                S_mat[j, i] = S_val
                M_mat[i, j] = M_val
                M_mat[j, i] = M_val
            end
        end
    end
    S_mat = Hermitian(S_mat)
    M_mat = Hermitian(M_mat)
    @printf("Done in %.4f seconds\n", time_ops)

    # 4. Canonical Orthogonalization (正则正交化处理)
    λ_S, U_S = eigen(S_mat)

    valid_idx = findall(x -> x > S_tol, λ_S)
    N_valid = length(valid_idx)
    @printf("Conditioning S matrix: %d / %d basis vectors kept (S_tol = %.1e)\n\n", N_valid, N_pool, S_tol)

    if N_valid == 0
        error("No linearly independent basis vectors found in the pool. Try reducing S_tol.")
    end

    λ_S_valid = λ_S[valid_idx]
    U_S_valid = U_S[:, valid_idx]
    X = U_S_valid * Diagonal(1.0 ./ sqrt.(λ_S_valid))

    # 5. 求解 qEOM 本征值问题: X^† M X C = ΔE C
    M_orth = Hermitian(X' * M_mat * X)
    ΔE_qeom, C_orth = eigen(M_orth)

    # 映射回算符池系数 (激发态展开系数)
    C_qeom = X * C_orth

    N = length(e_scales)
    n_exc = min(n_states, N_valid)
    states = [i == 0 ? "000 (Ref)" : @sprintf("%03d", i) for i in 0:n_exc]
    energies = [i == 0 ? @sprintf("%.15f", E_ref) : @sprintf("%.15f", E_ref + ΔE_qeom[i]) for i in 0:n_exc]
    delta_es = [i == 0 ? "-" : @sprintf("%.15f", ΔE_qeom[i]) for i in 0:n_exc]
    errors = [
        if i == 0
            N >= 1 ? @sprintf("%.3e", abs(E_ref - e_scales[1])) : "-"
        else
            (i + 1) <= N ? @sprintf("%.3e", abs((E_ref + ΔE_qeom[i]) - e_scales[i+1])) : "-"
        end
        for i in 0:n_exc
    ]
    df_qeom = DataFrame(
        "State" => states,
        "Total Energy" => energies,
        "ΔE (Excit)" => delta_es,
        "Error" => errors
    )
    show(stdout, df_qeom, summary=false, eltypes=false, show_row_number=false)
    println("\n")

    return ΔE_qeom, C_qeom
end


function run_qpe_ode(
    basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, v0::Vector{Tv}; 
    e_scales::Vector{Float64}=[], dt::Float64=0.05, max_step::Int64=4000, ode_tol::Float64=1e-8,
) where {Ti,Tv,TK,TV}
    @assert Tv <: Complex

    println("\n--- Starting Quantum Phase Estimation (QPE via ODE) ---")

    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,TK,TV}[], time_print=false)

    v = v0
    normalize!(v)
    w = zeros(Tv, basis.dim)

    # 1. 计算参考能量 E_ref，用于平移能谱避免高频相位混叠
    funcs.hvec(v, w)
    E_ref = real(dot(v, w))

    @printf("Reference Energy: %.6f Hartree\n", E_ref)
    @printf("Time step (dt)  : %.4f\n", dt)
    @printf("Total steps     : %d\n", max_step)
    resolution = 2 * pi / (max_step * dt)
    @printf("Energy Resol.   : %.4f Hartree\n", resolution)

    # 2. 定义薛定谔方程 ODE: dψ/dt = -i * (H - E_ref) * ψ
    f_schrodinger! = (du, u, p, t) -> begin
        funcs.hvec(u, w)
        @. du = -im * (w - E_ref * u)
    end

    # 3. 设定采样时间点
    # FFT 需要严格等距的时间采样点：0, dt, 2dt, ..., (max_step-1)*dt
    tspan = (0.0, (max_step - 1) * dt)
    save_times = range(0.0, step=dt, length=max_step)

    # 4. 执行自适应 ODE 演化
    prob = ODEProblem(f_schrodinger!, v, tspan)

    print("Running ODE Real-Time Evolution ... ")
    t_evo = @elapsed begin
        # saveat=save_times 是灵魂：求解器内部会自动按自适应步长积分，
        # 但只在我们指定的时间点通过高阶插值把波函数保存下来！
        sol = solve(prob, Tsit5(), saveat=save_times, abstol=ode_tol, reltol=ode_tol)
    end

    @printf("Done in %.4f seconds\n", t_evo)

    # 5. 提取自相关函数 C(t)
    C_t = zeros(Tv, max_step)
    for i in 1:max_step
        # sol.u[i] 就是精确在 t = (i-1)*dt 时刻的波函数
        C_t[i] = dot(v, sol.u[i])
    end

    # 6. 信号处理与 FFT 
    window = [0.5 * (1 - cos(2 * pi * i / (max_step - 1))) for i in 0:(max_step - 1)]
    C_t_windowed = C_t .* window
    S = fft(C_t_windowed)
    freqs = fftfreq(max_step, 2 * pi / dt)

    # 频率转换回能量并加回参考点
    energies = -freqs .+ E_ref
    powers = abs.(S)

    # 7. 寻峰算法
    peaks = []
    threshold = 0.01 * maximum(powers)

    for i in 2:(max_step-1)
        if powers[i] > powers[i-1] && powers[i] > powers[i+1] && powers[i] > threshold
            push!(peaks, (energies[i], powers[i]))
        end
    end

    sort!(peaks, by=x -> x[2], rev=true)

    println("\n--- QPE Extracted Energy Spectrum (Top Peaks) ---")
    if isempty(peaks)
        println("  No significant peaks found.")
    else
        @printf("  %-6s %-18s %-18s %-18s\n", "Peak", "Energy", "Error", "Relative Power")
        max_power = peaks[1][2]
        for (i, (E, P)) in enumerate(peaks[1:min(10, length(peaks))])
            if i <= length(e_scales)
                err = abs(E - e_scales[i])
            else
                err = NaN
            end
            @printf("  %03d    % 15.10f    % .3e      % 10.4f\n", i, E, err, P / max_power)
        end
    end
    println("=================================================================\n")

    return peaks
end
