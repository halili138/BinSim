function get_hvec(basis::BasisManager, ham::BinaryQubitAABB; is_time::Bool=false)
    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed ham_otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)

    if is_time
        f_hvec = (v, Hv) -> @printf("hvec time %.6f seconds", @elapsed hvec_otf!(basis, ham_otf, v, Hv))
    else
        f_hvec = (v, Hv) -> hvec_otf!(basis, ham_otf, v, Hv)
    end

    return f_hvec
end


function get_tvec(basis::BasisManager, pool::Vector{<:BinaryQubitAABB};
    expm::Bool=false, tvec::Bool=false, grad::Bool=false,
    backgrad::Bool=false, batchgrad::Bool=false, tran::Bool=false
)
    print("Pre-compiling Pool OTF ... ")
    time_ops = @elapsed pool_otf = OTF(basis, pool)
    @printf("Done in %.4f seconds\n", time_ops)

    f_expm = nothing
    f_tvec = nothing
    f_grad = nothing
    f_backgrad = nothing
    f_batchgrad = nothing
    f_tran = nothing

    expm && (f_expm = (idx, θ, v) -> expm_svd!(basis, pool_otf, idx, θ, v))
    tvec && (f_tvec = (idx, v, Tv) -> tvec_svd!(basis, pool_otf, idx, v, Tv))
    grad && (f_grad = (idx, θ, lv, rv) -> return grad_svd(basis, pool_otf, idx, θ, lv, rv))
    backgrad && (f_backgrad = (idx, θ, lv, rv) -> return backgrad_svd!(basis, pool_otf, idx, θ, lv, rv))
    batchgrad && (f_batchgrad = (lv, rv, g, x) -> return batch_grad_svd(basis, pool_otf, x, lv, rv, g))
    tran && (f_tran = (lv, rv, trans) -> return tran_svd(basis, pool_otf, lv, rv, trans))

    return f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran
end


function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}) where {Ti,Tv,K,V}
    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)

    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, otf, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    hvec! = (v, Hv) -> @printf("hvec time %.6f seconds", @elapsed hvec_otf!(basis, otf, v, Hv))

    return @time davidson(hvec!, v0, diags, tol=1e-5)
end


function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}; k::Int=1) where {Ti,Tv,K,V}
    hvec! = get_hvec(basis, ham, is_time=false)
    hvec_map = LinearMap{Tv}(
        (dst, src) -> hvec!(src, dst),
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


function run_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS()
) where {Ti,Tv,K,V}
    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, expm=true, backgrad=true)

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    energy = Ref(0.0)
    norm_g = Ref(0.0)
    δ²H = Ref(0.0)
    error = Ref(0.0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        lv .= v0
        result = @timed energy_objective(f_hvec, f_expm, f_backgrad, idxs, x, lv, rv)
        energy[], grad, δ²H[] = result.value

        norm_g[] = norm(grad)
        error[] = abs(energy[] - e_scale)
        options.verbose > 1 && show_optimze(energy[], norm_g[], δ²H[], error[])
        options.verbose > 2 && show_time(result)

        return energy[], grad
    end

    println("Performing VQE optimization ... ")
    time_ops = @elapsed e_opt, x_opt = optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
    @printf("Converged in %.4f seconds with: f = %.14f  |g| = %.3e  δ²H = %.3e  err = %.3e\n",
        time_ops, energy[], norm_g[], δ²H[], error[])
    println("\n")

    lv .= v0

    for i in eachindex(idxs)
        f_expm(idxs[i], x_opt[i], lv)
    end

    return e_opt, lv, x_opt
end


function run_adapt_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    amplitudes::Vector{Float64}=Float64[],
    selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}
    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, expm=true, backgrad=true, batchgrad=true)

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    amplitudes, selec_idxs = _adapt_vqe(
        f_hvec,
        f_expm,
        f_backgrad,
        f_batchgrad,
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

    return amplitudes, selec_idxs
end


function run_rk4_ite(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv},
    e_scale::Float64;
    dτ::Float64=0.02,
    max_step::Int64=1000,
    tol::Float64=1e-8,
) where {Ti,Tv,K,V}
    """
    四阶 Runge-Kutta 虚时演化 (4 次 hvec/步):
    dpsi/dtau = -H|psi>
    k1 = -H * psi
    k2 = -H * (psi + dtau/2 * k1)
    k3 = -H * (psi + dtau/2 * k2)
    k4 = -H * (psi + dtau * k3)
    psi_new = psi + dtau/6 * (k1 + 2*k2 + 2*k3 + k4)
    """

    hvec! = get_hvec(basis, ham, is_time=false)

    v = v0
    ws::Vector{Vector{Tv}} = [zeros(Tv, basis.dim) for _ in 1:5]

    E_hist = Float64[]
    dH_hist = Float64[]

    step = 0
    while step <= max_step
        step += 1

        hvec!(v, ws[1])

        ln = norm(v)^2
        rn = norm(ws[1])^2
        E = real(dot(v, ws[1])) / ln
        dH = max(0.0, rn / ln - E^2)
        push!(E_hist, E)
        push!(dH_hist, dH)

        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        @printf("  Step %03d     E %.14f    Err %.3e    dE %.3e    δ²H %.3e   τ %.2f\n",
            step, E, abs(E - e_scale), dE, dH, step * dτ)

        abs(dE) < tol && break

        vt = ws[5]

        ws[1] .*= -1.0

        @. vt = v + dτ / 2 * ws[1]
        hvec!(vt, ws[2])
        ws[2] .*= -1.0

        @. vt = v + dτ / 2 * ws[2]
        hvec!(vt, ws[3])
        ws[3] .*= -1.0

        @. vt = v + dτ * ws[3]
        hvec!(vt, ws[4])
        ws[4] .*= -1.0

        @. v += dτ / 6 * (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4])

        normalize!(v)
    end

    println("  Converged at step $step\n")

    return E_hist[end]
end


function estimate_max_step(hvec!::Function, dim::Int, E_ground_guess::Float64)
    v = randn(Float64, dim)
    w = zeros(Float64, dim)
    normalize!(v)

    λ_max = 0.0
    for _ in 1:40
        hvec!(v, w)

        # 【核心修正】：执行平移幂法 (H - E_guess * I)|v>
        # 这确保了正方向的能量绝对值被彻底放大
        @. w = w - E_ground_guess * v

        # 此时得到的本征值是平移后的, 需要加回来
        λ_shifted = real(dot(v, w))
        λ_max = λ_shifted + E_ground_guess

        nw = norm(w)
        @. v = w / nw
    end

    @printf("Estimated λ_max: %.6f\n", λ_max)

    N = λ_max + E_ground_guess
    dτ_limit = 2.0 / (λ_max + E_ground_guess)

    # 【无条件稳定判定】
    if N <= 0.0
        println("Spectrum sum is negative. Unconditionally stable!")
        # dτ_limit = dτ_limit * 1.05
        dτ_limit = 1e12
    else
        dτ_limit = dτ_limit * 0.95
    end

    println("Theoretical dτ limit: $(dτ_limit)\n")

    return dτ_limit
end


function run_euler_ite(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv},
    e_scale::Float64;
    dτ::Float64=0.0,
    max_step::Int64=5000,
    tol::Float64=1e-8,
    save_path::String="",
) where {Ti,Tv,K,V}
    """
    一阶 Euler 虚时演化 (1 次 hvec/步):
    dpsi/dtau = -H|psi>
    psi_new = psi - dtau * H * psi
    """

    hvec! = get_hvec(basis, ham, is_time=false)

    if iszero(dτ)
        dτ = estimate_max_step(hvec!, basis.dim, e_scale)
    end

    v = copy(v0)
    w = zeros(Tv, basis.dim)

    E_hist = Float64[]
    dH_hist = Float64[]

    step = 0
    while step <= max_step
        step += 1
        if !isempty(save_path) && (step % 5 == 0)
            jldopen(save_path, "w") do file
                file["v"] = v
            end
            println("Successifully save wave function to $(save_path) at step = $(step)")
        end

        @time hvec!(v, w)

        ln = norm(v)^2
        rn = norm(w)^2
        E = real(dot(v, w)) / ln
        dH = max(0.0, rn / ln - E^2)
        push!(E_hist, E)
        push!(dH_hist, dH)

        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        if dτ <= 10
            @printf("  Step %04d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e    τ %.2f\n",
                step, E, abs(E - e_scale), dE, dH, step * dτ)
        else
            @printf("  Step %04d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e\n",
                step, E, abs(E - e_scale), dE, dH)
        end

        abs(dE) < tol && break

        @. v -= dτ * w

        # @. v += dτ * (E * v - w)

        normalize!(v)
    end

    println("  \nConverged at step $step\n")

    return E_hist[end]
end


function run_krylov_ite(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    v0::Vector{Tv},
    e_scale::Float64;
    dτ::Float64=1.0,
    krylov_dim::Int=20,
    max_step::Int64=200,
    tol::Float64=1e-8,
) where {Ti,Tv,TK,TV}
    """
    Krylov 子空间指数法虚时演化 (m 次 hvec/步):
    利用 Lanczos 算法构建 m 维子空间, 投影哈密顿量为三对角矩阵 Tm
    psi(τ + dτ) ≈ V * exp(-dτ * Tm) * e1
    """

    hvec! = get_hvec(basis, ham, is_time=false)

    v = v0
    normalize!(v)

    V = [zeros(Tv, basis.dim) for _ in 1:krylov_dim]
    w = zeros(Tv, basis.dim)
    α = zeros(Float64, krylov_dim)
    β = zeros(Float64, krylov_dim)

    E_hist = Float64[]
    dH_hist = Float64[]

    step = 0
    while step <= max_step
        step += 1

        # 1. 初始基向量
        copyto!(V[1], v)
        m_actual = krylov_dim

        # 2. Lanczos 迭代构建子空间
        for j in 1:krylov_dim
            v_j = V[j]    # 极速获取引用, 类型为纯正的 Vector{Tv}
            hvec!(v_j, w) # 这里 hvec! 接收的将是完美的纯向量, 毫无阻碍

            if j == 1
                rn = norm(w)^2
                E = real(dot(v_j, w))
                dH = max(0.0, rn - E^2)
                push!(E_hist, E)
                push!(dH_hist, dH)
            end

            α[j] = real(dot(v_j, w))

            @. w = w - α[j] * v_j
            if j > 1
                v_prev = V[j-1]
                @. w = w - β[j-1] * v_prev
            end

            # 完全正交化
            for i in 1:j
                v_i = V[i]
                c = dot(v_i, w)
                @. w = w - c * v_i
            end

            norm_w = norm(w)

            if j < krylov_dim
                if norm_w < 1e-12
                    m_actual = j
                    break
                end
                β[j] = norm_w
                v_next = V[j+1]
                @. v_next = w / norm_w
            end
        end

        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        @printf("  Step %03d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e    τ %.2f\n",
            step, E_hist[end], abs(E_hist[end] - e_scale), dE, dH_hist[end], step * dτ)

        abs(dE) < tol && break

        # 3. 构造子空间投影的三对角矩阵 Tm 并求指数
        Tm = SymTridiagonal(α[1:m_actual], β[1:m_actual-1])
        U = exp(-dτ * Matrix(Tm))

        c = U[:, 1]

        # 4. 映射回全空间
        fill!(v, 0.0)
        for j in 1:m_actual
            v_j = V[j]
            @. v += c[j] * v_j
        end

        normalize!(v)
    end

    println("  \nConverged at step $step\n")

    return E_hist[end]
end


function run_enpt2(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv},
    e_scale::Float64;
    ref_tol::Float64=1e-4,
    level_shift::Float64=0.0
) where {Ti,Tv,K,V}
    """
    Epstein-Nesbet 二阶微扰理论 (ENPT2) 后处理校正
    利用已收敛的近似波函数 v0, 计算残差并估计动态相关能。

    公式: E^(2) = sum_{i ∉ ref} |<i|H - E0|v0>|^2 / (E0 - H_ii - shift)
    """
    println("\n--- Starting ENPT2 Post-Processing ---")

    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)
    hvec! = (v, Hv) -> hvec_otf!(basis, otf, v, Hv)

    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, otf, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    # 确保参考态已经归一化
    v = copy(v0)
    normalize!(v)

    w = zeros(Tv, basis.dim)

    # 1. 计算 H|v0> 
    hvec!(v, w)

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


function run_qse(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv};
    e_scales::Vector{Float64}=Float64[],
    S_tol::Float64=1e-8,
    n_states::Int=5,
) where {Ti,Tv,K,V}
    println("\n--- Starting Quantum Subspace Expansion (QSE) ---\n")

    hvec! = get_hvec(basis, ham, is_time=false)
    f_expm, tvec!, f_grad, f_backgrad.f_batchgrad, f_tran = get_tvec(basis, pool, tvec=true)

    get_V! = (k, dst) -> begin
        if k == 1
            copyto!(dst, v0)
        else
            tvec!(k - 1, v0, dst)
        end
    end

    N_sub = length(pool) + 1
    print("Building S and H Matrices ... ")
    v_i_buf = zeros(Tv, basis.dim)
    v_j_buf = zeros(Tv, basis.dim)
    hv_j_buf = zeros(Tv, basis.dim)
    S_mat = zeros(Tv, N_sub, N_sub)
    H_mat = zeros(Tv, N_sub, N_sub)

    time_ops = @elapsed begin
        for j in 1:N_sub
            get_V!(j, v_j_buf)
            hvec!(v_j_buf, hv_j_buf)
            for i in 1:j
                if i == j
                    S_mat[j, j] = real(dot(v_j_buf, v_j_buf))
                    H_mat[j, j] = real(dot(v_j_buf, hv_j_buf))
                else
                    get_V!(i, v_i_buf)
                    s_val = real(dot(v_i_buf, v_j_buf))
                    h_val = real(dot(v_i_buf, hv_j_buf))
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


function run_qeom(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv};
    e_scales::Vector{Float64}=Float64[],
    S_tol::Float64=1e-8,
    n_states::Int=5,
) where {Ti,Tv,K,V}

    println("\n--- Starting Quantum Equation-of-Motion (qEOM) ---\n")

    hvec! = get_hvec(basis, ham, is_time=false)
    f_expm, tvec!, f_grad, f_backgrad.f_batchgrad, f_tran = get_tvec(basis, pool, tvec=true)

    N_pool = length(pool)
    # 2. 准备 qEOM 核心的辅助态
    v0_tilde = zeros(Tv, basis.dim)
    hvec!(v0, v0_tilde)            # |v0_tilde> = H |v0>

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
            tvec!(j, v0, v_j)               # |v_j> = O_j |v0>
            tvec!(j, v0_tilde, v_j_tilde)   # |v_j_tilde> = O_j |v0_tilde> = O_j H |v0>
            hvec!(v_j, hv_j)                     # H |v_j>
            for i in 1:j
                tvec!(i, v0, v_i)
                tvec!(i, v0_tilde, v_i_tilde)
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


function run_ssvqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0s::Vector{Vector{Tv}},
    weights::Vector{Float64},
    e_scales::Vector{Float64};
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS()
) where {Ti,Tv,K,V}
    K_states = length(v0s)
    @assert length(weights) == K_states "Number of weights must match number of initial states"
    @assert length(e_scales) == K_states "Number of energy scales must match number of initial states"

    println("SSVQE Target States: $(K_states)\n")

    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, expm=true, backgrad=true)

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]


    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    step_counter = Ref(0)

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        total_L = 0.0
        total_grad = zeros(Float64, length(x))
        max_δ²H = 0.0
        state_metrics = []

        time_ops = @elapsed for k in 1:K_states
            lv .= v0s[k]
            e_k, g_k, δ²H_k = energy_objective(f_hvec, f_expm, f_backgrad, idxs, x, lv, rv)

            total_L += weights[k] * e_k
            total_grad .+= weights[k] .* g_k
            max_δ²H = max(max_δ²H, δ²H_k)

            # 计算并记录当前态的指标
            norm_gk = norm(g_k)
            err_k = abs(e_k - e_scales[k])
            push!(state_metrics, (k, e_k, norm_gk, δ²H_k, err_k))
        end

        if options.verbose > 1
            step_counter[] += 1
            norm_g = norm(total_grad)
            target_L = sum(weights .* e_scales)
            error = total_L - target_L

            @printf(" SSVQE Eval %04d\n", step_counter[])
            @printf(" f: %.14f   |g|: %.3e   err: %.3e   time: %.3fs\n",
                total_L, norm_g, error, time_ops)
            show_ssvqe_optimze(state_metrics)
        end

        return total_L, total_grad
    end

    _, x_opt = @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)

    e_opts = zeros(Float64, K_states)
    v_opts = [zeros(Tv, basis.dim) for _ in 1:K_states]

    for k in 1:K_states
        v_opts[k] .= v0s[k]

        for i in eachindex(idxs)
            f_expm(idxs[i], x_opt[i], v_opts[k])
        end

        f_hvec(v_opts[k], rv)
        e_opts[k] = real(dot(v_opts[k], rv))
    end

    return e_opts, v_opts, x_opt
end


function run_adapt_ssvqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0s::Vector{Vector{Tv}},
    weights::Vector{Float64},
    e_scales::Vector{Float64};
    amplitudes::Vector{Float64}=Float64[],
    selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}
    println("SSVQE Target States: $(length(v0s))\n")
    f_hvec = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_tran = get_tvec(basis, pool, expm=true, grad=true)

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]


    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    _adapt_ssvqe(
        f_hvec, f_expm, f_grad, idxs, v0s, weights, lv, rv,
        e_scales, amplitudes, selec_idxs, adapt_options, vqe_options
    )
end


function generate_ssvqe_inputs(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V};
    k_states::Int=2,
    weight_decay::Float64=0.5
) where {Ti,Tv,K,V}
    """
    自动生成 SSVQE 所需的正交初态 v0s 和严格递减的 weights。
    策略: 基于哈密顿量对角元，选取能量最低的 K 个独立计算基矢(Slater行列式)。
    """

    println("--- Generating SSVQE Initial States ---")
    @assert k_states > 0 && k_states <= basis.dim "k_states must be within basis dimension"

    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)

    print("Generating Diag elements vector ... ")
    time_ops = @elapsed diags = get_diags(basis, otf, Tv)
    @printf("Done in %.4f seconds\n", time_ops)

    # 2. 找到对角元能量最低的 K 个构型的索引
    # sortperm 会返回从小到大排序的索引集
    sorted_idxs = sortperm(diags)
    selected_idxs = sorted_idxs[1:k_states]

    # 3. 构造正交初始态 (v0s)
    v0s = Vector{Vector{Tv}}(undef, k_states)
    for k in 1:k_states
        v = zeros(Tv, basis.dim)
        idx = selected_idxs[k]
        v[idx] = 1.0  # 设置为计算基矢 (One-hot 向量)，天然相互正交
        v0s[k] = v

        # 打印选出的基矢能量，用于物理检查 (比如基态是不是 HF 态)
        @printf("  State %d -> Basis Index: %-8d Zero-order Energy: %.6f\n", k, idx, diags[idx])
    end

    # 4. 构造递减权重 (weights)
    # 使用指数衰减策略: 1.0, 0.5, 0.25... (归一化以防止梯度爆炸)
    raw_weights = [weight_decay^(k - 1) for k in 1:k_states]
    weights = raw_weights ./ sum(raw_weights)

    # 5. 生成对应的 e_scales (用于打印误差参考，如果没有 FCI 参考可以设为零阶能量)
    e_scales = [diags[idx] for idx in selected_idxs]

    println("  Weights   : ", round.(weights, digits=4))
    println("---------------------------------------\n")

    return v0s, weights, e_scales
end


function run_krylov_rte(
    hvec!::Function,
    v0::Vector{ComplexF64};
    dt::Float64=0.05,
    krylov_dim::Int=20,
    max_step::Int64=2000,
    E_ref::Float64=0.0,
)
    v = copy(v0)
    normalize!(v)

    V = [zeros(ComplexF64, basis.dim) for _ in 1:krylov_dim]
    w = zeros(ComplexF64, basis.dim)
    α = zeros(Float64, krylov_dim)
    β = zeros(Float64, krylov_dim)

    # 记录自相关函数 C(t) = <psi(0)|psi(t)>
    C_t = zeros(ComplexF64, max_step)

    print("Running Krylov Real-Time Evolution ")

    time_ops = @elapsed for step in 1:max_step
        C_t[step] = dot(v0, v)

        copyto!(V[1], v)
        m_actual = krylov_dim

        for j in 1:krylov_dim
            v_j = V[j]
            hvec!(v_j, w)

            α[j] = real(dot(v_j, w))

            @. w = w - α[j] * v_j
            if j > 1
                @. w = w - β[j-1] * V[j-1]
            end

            # 完全正交化
            for i in 1:j
                c = dot(V[i], w)
                @. w = w - c * V[i]
            end

            norm_w = norm(w)

            if j < krylov_dim
                if norm_w < 1e-12
                    m_actual = j
                    break
                end
                β[j] = norm_w
                @. V[j+1] = w / norm_w
            end
        end

        Tm = SymTridiagonal(α[1:m_actual], β[1:m_actual-1])

        # 【核心修改点】：在计算矩阵指数前，减去参考能量！
        # 这意味着我们在演化 H' = H - E_ref*I，相位就不再发生混叠。
        Tm_shifted = Matrix(Tm) - E_ref * I
        U = exp(-im * dt * Tm_shifted)

        c = U[:, 1]

        fill!(v, 0.0)
        for j in 1:m_actual
            @. v += c[j] * V[j]
        end

        if step % 500 == 0
            print(".") # 简单的进度指示
        end
    end

    @printf(" Done in %.4f seconds\n", time_ops)

    return C_t
end


function run_qpe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    v0::Vector{Tv};
    dt::Float64=0.05,
    max_step::Int64=4000,  # 推荐增加步长以提高分辨率
    krylov_dim::Int=20,
) where {Ti,Tv,TK,TV}

    println("\n--- Starting Quantum Phase Estimation (QPE) ---")
    hvec! = get_hvec(basis, ham, is_time=false)

    # 【新增 0】：计算输入态的期望能量作为参考零点
    w_temp = zeros(Tv, basis.dim)
    hvec!(v0, w_temp)
    E_ref = real(dot(v0, w_temp)) / norm(v0)^2

    @printf("Reference Energy: %.6f Hartree\n", E_ref)
    @printf("Time step (dt)  : %.4f\n", dt)
    @printf("Total steps     : %d\n", max_step)
    resolution = 2 * pi / (max_step * dt)
    @printf("Energy Resol.   : %.4f Hartree\n", resolution)

    t0 = time()

    # 【修改 1】：把 E_ref 传进去
    C_t = run_krylov_rte(hvec!, v0, dt=dt, krylov_dim=krylov_dim, max_step=max_step, E_ref=E_ref)

    # 信号处理与 FFT 保持不变
    window = [0.5 * (1 - cos(2 * pi * i / (max_step - 1))) for i in 0:(max_step-1)]
    C_t_windowed = C_t .* window
    S = fft(C_t_windowed)
    freqs = fftfreq(max_step, 2 * pi / dt)

    # 【修改 2】：提取出来的能量必须把 E_ref 加回来！
    energies = -freqs .+ E_ref
    powers = abs.(S)

    # 4. 寻峰算法 (找出局部最大值)
    peaks = []
    # 过滤掉强度太低的背景噪声峰 (阈值设为最大峰值的 1%)
    threshold = 0.01 * maximum(powers)

    for i in 2:(max_step-1)
        if powers[i] > powers[i-1] && powers[i] > powers[i+1] && powers[i] > threshold
            push!(peaks, (energies[i], powers[i]))
        end
    end

    # 按照峰值强度(Power)降序排列
    sort!(peaks, by=x -> x[2], rev=true)

    t1 = time()
    @printf("QPE Analysis   ... Done in %.4f seconds\n", t1 - t0)

    # 5. 打印结果
    println("\n--- QPE Extracted Energy Spectrum (Top Peaks) ---")
    @printf("  %-6s %-18s %-18s\n", "Peak", "Energy", "Relative Power")

    if isempty(peaks)
        println("  No significant peaks found.")
    else
        max_power = peaks[1][2]
        # 最多打印前 10 个最强的峰
        for (i, (E, P)) in enumerate(peaks[1:min(10, length(peaks))])
            @printf("  %03d    % 15.10f      % 10.4f\n", i, E, P / max_power)
        end
    end
    println("=================================================================\n")

    return peaks
end


function run_qpe_ode(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    v0::Vector{Tv};
    dt::Float64=0.05,
    max_step::Int64=4000,
    ode_tol::Float64=1e-8,
) where {Ti,Tv,TK,TV}

    println("\n--- Starting Quantum Phase Estimation (QPE via ODE) ---")
    hvec! = get_hvec(basis, ham, is_time=false)

    v0_c = complex.(v0)
    w_temp = zeros(ComplexF64, basis.dim)

    # 计算参考能量 E_ref，用于平移能谱避免高频相位混叠
    hvec!(v0_c, w_temp)
    E_ref = real(dot(v0_c, w_temp)) / norm(v0_c)^2

    @printf("Reference Energy: %.6f Hartree\n", E_ref)
    @printf("Time step (dt)  : %.4f\n", dt)
    @printf("Total steps     : %d\n", max_step)
    resolution = 2 * pi / (max_step * dt)
    @printf("Energy Resol.   : %.4f Hartree\n", resolution)

    # 2. 定义薛定谔方程 ODE: dψ/dt = -i * (H - E_ref) * ψ
    f_schrodinger! = (du, u, p, t) -> begin
        hvec!(u, w_temp)
        # 提取公共因子以利用 SIMD 加速
        @. du = -im * (w_temp - E_ref * u)
    end

    # 3. 设定采样时间点
    # FFT 需要严格等距的时间采样点：0, dt, 2dt, ..., (max_step-1)*dt
    tspan = (0.0, (max_step - 1) * dt)
    save_times = range(0.0, step=dt, length=max_step)

    # 4. 执行自适应 ODE 演化
    prob = ODEProblem(f_schrodinger!, v0_c, tspan)

    print("Running ODE Real-Time Evolution ... ")
    t_evo = @elapsed begin
        # saveat=save_times 是灵魂：求解器内部会自动按自适应步长积分，
        # 但只在我们指定的时间点通过高阶插值把波函数保存下来！
        sol = solve(prob, Tsit5(), saveat=save_times, abstol=ode_tol, reltol=ode_tol)
    end
    @printf("Done in %.4f seconds\n", t_evo)

    # 5. 提取自相关函数 C(t)
    C_t = zeros(ComplexF64, max_step)
    for i in 1:max_step
        # sol.u[i] 就是精确在 t = (i-1)*dt 时刻的波函数
        C_t[i] = dot(v0_c, sol.u[i])
    end

    # 6. 信号处理与 FFT 
    window = [0.5 * (1 - cos(2 * pi * i / (max_step - 1))) for i in 0:(max_step-1)]
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

    # 8. 打印结果
    println("\n--- QPE Extracted Energy Spectrum (Top Peaks) ---")
    @printf("  %-6s %-18s %-18s\n", "Peak", "Energy", "Relative Power")
    if isempty(peaks)
        println("  No significant peaks found.")
    else
        max_power = peaks[1][2]
        for (i, (E, P)) in enumerate(peaks[1:min(10, length(peaks))])
            @printf("  %03d    % 15.10f      % 10.4f\n", i, E, P / max_power)
        end
    end
    println("=================================================================\n")

    return peaks
end


function run_exact_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS(),
    n_steps::Int=50,
) where {Ti,Tv,K,V}
    println("============================================================================")
    println("--- Optimized Exact UCC VQE (ODE Adjoint Method) ---")

    hvec! = get_hvec(basis, ham, is_time=false)
    f_expm, tvec!, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, tvec=true)

    if !isempty(x0)
        @assert length(x0) == length(pool)
    else
        x0 = zeros(Float64, length(pool))
    end

    vt = zeros(Tv, basis.dim)
    v = zeros(Tv, basis.dim)
    Hv = zeros(Tv, basis.dim)
    vs = [zeros(Tv, basis.dim) for _ in 1:4]
    Hvs = [zeros(Tv, basis.dim) for _ in 1:4]
    dτ = 1.0 / n_steps

    obj_func = x -> begin
        if norm(x) < 1e-12
            v .= v0
            hvec!(v, Hv)
            E = real(dot(v, Hv))
            ln = norm(v)^2
            rn = norm(Hv)^2
            δ²H = max(0.0, rn / ln - E^2)

            Hv .*= 2.0
            g_tot = zeros(Float64, length(pool))
            for i in eachindex(pool)
                tvec!(i, v, vt)
                g_tot[i] = real(dot(vt, Hv))
            end

            options.verbose > 0 && show_optimze(E, norm(g_tot), δ²H, abs(E - e_scale))

            return E, g_tot
        end

        # 常规通道：线性组合并生成演化算子
        T = linearcombine(pool, x, 0.0, 1e-12)
        if T == zero(T)
            Tvec! = (src, dst) -> fill!(dst, 0.0)
        else
            T_otf = OTF(basis, T)
            Tvec! = (src, dst) -> hvec_otf!(basis, T_otf, src, dst)
        end

        # 1. 正向演化
        v .= v0
        for _ in 1:n_steps
            Tvec!(v, vs[1])
            @. vt = v + vs[1] * dτ / 2
            Tvec!(vt, vs[2])
            @. vt = v + vs[2] * dτ / 2
            Tvec!(vt, vs[3])
            @. vt = v + vs[3] * dτ
            Tvec!(vt, vs[4])
            @. v += (vs[1] + 2 * vs[2] + 2 * vs[3] + vs[4]) * dτ / 6
        end

        normalize!(v)
        hvec!(v, Hv)
        E = real(dot(v, Hv))
        ln = norm(v)^2
        rn = norm(Hv)^2
        δ²H = max(0.0, rn / ln - E^2)

        Hv .*= 2.0
        g_tot = zeros(Float64, length(pool))
        g_curr = zeros(Float64, length(pool))
        g_next = zeros(Float64, length(pool))
        for i in eachindex(pool)
            tvec!(i, v, vt)
            g_curr[i] = real(dot(vt, Hv))
        end

        # 2. 反向伴随演化与梯度积分
        for _ in 1:n_steps
            # 演化波函数 v
            Tvec!(v, vs[1])
            @. vt = v - vs[1] * dτ / 2
            Tvec!(vt, vs[2])
            @. vt = v - vs[2] * dτ / 2
            Tvec!(vt, vs[3])
            @. vt = v - vs[3] * dτ
            Tvec!(vt, vs[4])
            @. v -= (vs[1] + 2 * vs[2] + 2 * vs[3] + vs[4]) * dτ / 6

            # 演化伴随态 Hv
            Tvec!(Hv, Hvs[1])
            @. vt = Hv - Hvs[1] * dτ / 2
            Tvec!(vt, Hvs[2])
            @. vt = Hv - Hvs[2] * dτ / 2
            Tvec!(vt, Hvs[3])
            @. vt = Hv - Hvs[3] * dτ
            Tvec!(vt, Hvs[4])
            @. Hv -= (Hvs[1] + 2 * Hvs[2] + 2 * Hvs[3] + Hvs[4]) * dτ / 6

            ln = norm(v)^2
            for i in eachindex(pool)
                tvec!(i, v, vt)
                g_next[i] = real(dot(vt, Hv))
            end

            @. g_tot += (g_curr + g_next) * dτ / 2
            g_curr .= g_next
        end

        options.verbose > 0 && show_optimze(E, norm(g_tot), δ²H, abs(E - e_scale))

        return E, g_tot
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end


function run_exact_vqe_adaptive(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS(),
    ode_tol::Float64=1e-8, # ODE 积分精度
) where {Ti,Tv,K,V}
    println("============================================================================")
    println("--- Adaptive Exact UCC VQE (Augmented ODE Adjoint Method) ---")

    hvec! = get_hvec(basis, ham, is_time=false)
    f_expm, f_tvec, f_grad, f_backgrad, f_batchgrad, f_tran = get_tvec(basis, pool, tran=true)

    if !isempty(x0)
        @assert length(x0) == length(pool)
    else
        x0 = zeros(Float64, length(pool))
    end

    Hv = zeros(Tv, basis.dim)
    a = zeros(Tv, basis.dim)
    gt = zeros(Tv, length(pool))

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        if norm(x) < 1e-12
            hvec!(v0, Hv)
            E = real(dot(v0, Hv))
            δ²H = max(0.0, norm(Hv)^2 / norm(v0)^2 - E^2)

            @. a = 2.0 * Hv
            f_tran(v0, a, gt)

            options.verbose > 0 && show_optimze(E, norm(gt), δ²H, abs(E - e_scale))

            return E, real.(gt)
        end

        T = linearcombine(pool, x, 0.0, 1e-12)
        if T == zero(T)
            Tvec! = (src, dst) -> fill!(dst, 0.0)
        else
            T_otf = OTF(basis, T)
            Tvec! = (src, dst) -> hvec_otf!(basis, T_otf, src, dst)
        end

        f_forward! = (du, u, p, s) -> Tvec!(u, du)
        prob_fwd = ODEProblem(f_forward!, v0, (0.0, 1.0))
        sol_fwd = solve(prob_fwd, Tsit5(), abstol=ode_tol, reltol=ode_tol, save_everystep=false)

        v_final = sol_fwd.u[end]
        normalize!(v_final)

        hvec!(v_final, Hv)
        E = real(dot(v_final, Hv))
        δ²H = max(0.0, norm(Hv)^2 / norm(v_final)^2 - E^2)

        @. a = 2.0 * Hv
        g_init = zeros(Float64, length(pool))
        u_back_init = ArrayPartition(v_final, a, g_init)
        f_backward! = (du, u, p, s) -> begin
            ψ_curr = u.x[1]
            a_curr = u.x[2]
            dψ = du.x[1]
            da = du.x[2]
            dg = du.x[3]

            Tvec!(ψ_curr, dψ)
            Tvec!(a_curr, da)

            f_tran(ψ_curr, a_curr, gt)
            @. dg = -real(gt)
        end

        prob_bwd = ODEProblem(f_backward!, u_back_init, (1.0, 0.0))
        sol_bwd = solve(prob_bwd, Tsit5(), abstol=ode_tol, reltol=ode_tol, save_everystep=false)
        g_tot = sol_bwd.u[end].x[3]

        options.verbose > 0 && show_optimze(E, norm(g_tot), δ²H, abs(E - e_scale))

        return E, g_tot
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end

