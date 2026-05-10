function run_fci(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv};
    net::String="agg"
) where {Ti,Tv,K,V}
    psi_space = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)
    diags = get_diags(basis, ham)

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        aop! = (src, dst) -> @time hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        aop! = (src, dst) -> @time hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    # @time ham_sp = to_sparse_matrix(ham, pbc.norb, pbc.nelec)
    # @time λ, ϕ = eigs(ham_sp, nev=1, which=:SR)
    # println(λ)
    
    return @time davidson(aop!, v0, diags, tol=1e-5)
end


function run_krylovkit_diag(fci_basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}; k::Int=1) where {Ti,Tv,K,V}
    net = OTF(fci_basis, ham)

    aop = (src::Vector{Tv}) -> begin
        dst = similar(src) 
        hvec_otf!(fci_basis, net, src, dst) 
        return dst 
    end

    v0   = randn(Tv, fci_basis.dim)
    v0 ./= norm(v0)

    println("Running KrylovKit Arnoldi Solver...")

    @time vals, vecs, info = eigsolve(
        aop,                # 传入修改后的单参数函数
        v0,                 # 纯随机正态分布初始向量
        k,                  # 找 k 个特征值
        :SR,                # 找实部最小的 (Smallest Real)
        tol = 1e-5,         # 容差
        krylovdim = 20,     # Krylov 子空间最大维度 
        verbosity = 0       # 打印详细迭代日志
    )

    println("Energys: ", real.(vals))
    println("Convergence info: ", info)
    println("\n")

    return real.(vals), vecs
end


function run_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    net::String="agg",
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS()
) where {Ti,Tv,K,V}
    println("Num symmetry allowed elements: $(basis.dim)")
    println("Operator pool size: $(length(pool))")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        print("Pre-compiling Ham AGG ... ")
        time_ops = @elapsed ham_agg = AGG(basis, ham)
        @printf("Done in %.4f seconds\n", time_ops)

        print("Pre-compiling Pool NET ... ")
        time_ops = @elapsed pool_net = NET(basis, pool)
        @printf("Done in %.4f seconds\n", time_ops)

        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        print("Pre-compiling Ham OTF ... ")
        time_ops = @elapsed ham_otf  = OTF(basis, ham)
        @printf("Done in %.4f seconds\n", time_ops)

        print("Pre-compiling Pool OTF ... ")
        time_ops = @elapsed pool_otf = OTF(basis, pool)
        @printf("Done in %.4f seconds\n", time_ops)

        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        lv .= v0
        result = @timed energy_objective(f_hvec, f_tvec, f_grad, idxs, x, lv, rv)
        energy, grad, δ²H = result.value
        norm_g = norm(grad)
        error = energy - e_scale
        options.verbose > 0 && show_optimze(energy, norm_g, δ²H, error)
        options.verbose > 1 && show_time(result)

        return energy, grad
    end

    e_opt, x_opt = @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)

    lv .= v0
    
    for i in eachindex(idxs)
        f_tvec(idxs[i], x_opt[i], lv)
    end

    return e_opt, x_opt, lv
end


function run_adapt_vqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv},
    e_scale::Float64;
    net::String="agg",
    amplitudes::Vector{Float64}=Float64[], 
    selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}

    println("Num symmetry allowed elements: $(basis.dim)")
    println("Operator pool size: $(length(pool))")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        print("Pre-compiling Ham AGG ... ")
        time_ops = @elapsed ham_agg = AGG(basis, ham)
        @printf("Done in %.4f seconds\n", time_ops)

        print("Pre-compiling Pool NET ... ")
        time_ops = @elapsed pool_net = NET(basis, pool)
        @printf("Done in %.4f seconds\n", time_ops)

        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        print("Pre-compiling Ham OTF ... ")
        time_ops = @elapsed ham_otf  = OTF(basis, ham)
        @printf("Done in %.4f seconds\n", time_ops)

        print("Pre-compiling Pool OTF ... ")
        time_ops = @elapsed pool_otf = OTF(basis, pool)
        @printf("Done in %.4f seconds\n", time_ops)
        
        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    _adapt_vqe(
        f_hvec,
        f_tvec, 
        f_grad,  
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
end


function run_rk4(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,K,V}, 
    v0::Vector{Tv}, 
    e_scale::Float64;
    dτ::Float64=0.02, 
    max_step::Int64=1000, 
    tol::Float64=1e-8,
    net::String="agg",
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

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    v = v0
    ws::Vector{Vector{Tv}} = [zeros(Tv, basis.dim) for _ in 1:5]

    E_hist   = Float64[]
    dH_hist  = Float64[]

    step = 0
    while step <= max_step
        step += 1
        
        hvec!(v, ws[1])

        ln = norm(v) ^ 2
        rn = norm(ws[1]) ^ 2
        E  = real(dot(v, ws[1])) / ln
        dH = max(0.0, rn / ln - E ^ 2)
        push!(E_hist,  E)
        push!(dH_hist, dH)
        
        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        @printf("  Step %03d     E %.14f    Err %.3e    dE %.3e    δ²H %.3e   τ %.2f\n",
                step, E, abs(E-e_scale), dE, dH, step * dτ)

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


function run_euler(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,K,V}, 
    v0::Vector{Tv}, 
    e_scale::Float64;
    dτ::Float64=0.0,
    max_step::Int64=5000, 
    tol::Float64=1e-8,
    net::String="agg",
) where {Ti,Tv,K,V}
    """
    一阶 Euler 虚时演化 (1 次 hvec/步):
    dpsi/dtau = -H|psi>
    psi_new = psi - dtau * H * psi
    """

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    if iszero(dτ)
        dτ = estimate_max_step(hvec!, basis.dim, e_scale)
    end
    
    v = copy(v0)
    w = zeros(Tv, basis.dim) 

    E_hist   = Float64[]
    dH_hist  = Float64[]

    step = 0
    while step <= max_step
        step += 1
        
        hvec!(v, w)

        ln = norm(v) ^ 2
        rn = norm(w) ^ 2
        E  = real(dot(v, w)) / ln
        dH = max(0.0, rn / ln - E ^ 2)
        push!(E_hist,  E)
        push!(dH_hist, dH)
        
        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        if dτ <= 10
            @printf("  Step %04d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e    τ %.2f\n",
                    step, E, abs(E-e_scale), dE, dH, step * dτ)
        else
            @printf("  Step %04d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e\n",
                    step, E, abs(E-e_scale), dE, dH)
        end

        abs(dE) < tol && break

        @. v -= dτ * w

        # @. v += dτ * (E * v - w)

        normalize!(v)
    end 
    
    println("  \nConverged at step $step\n")

    return E_hist[end]
end


function run_krylov(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,TK,TV}, 
    v0::Vector{Tv}, 
    e_scale::Float64;
    dτ::Float64=1.0, 
    krylov_dim::Int=20, 
    max_step::Int64=200, 
    tol::Float64=1e-8,
    net::String="agg",
) where {Ti,Tv,TK,TV}
    """
    Krylov 子空间指数法虚时演化 (m 次 hvec/步):
    利用 Lanczos 算法构建 m 维子空间, 投影哈密顿量为三对角矩阵 Tm
    psi(τ + dτ) ≈ V * exp(-dτ * Tm) * e1
    """

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds\n")
        print_info(agg)
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    v = v0
    normalize!(v)

    V = [zeros(Tv, basis.dim) for _ in 1:krylov_dim]
    w = zeros(Tv, basis.dim) 
    α = zeros(Float64, krylov_dim)
    β = zeros(Float64, krylov_dim)

    E_hist   = Float64[]
    dH_hist  = Float64[]

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
                E  = real(dot(v_j, w))
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
                step, E_hist[end], abs(E_hist[end]-e_scale), dE, dH_hist[end], step * dτ)

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
    net::String="agg",
    ref_tol::Float64=1e-4, 
    level_shift::Float64=0.0
) where {Ti,Tv,K,V}
    """
    Epstein-Nesbet 二阶微扰理论 (ENPT2) 后处理校正
    利用已收敛的近似波函数 v0, 计算残差并估计动态相关能。
    
    公式: E^(2) = sum_{i ∉ ref} |<i|H - E0|v0>|^2 / (E0 - H_ii - shift)
    """
    println("\n--- Starting ENPT2 Post-Processing ---")
    
    # 获取哈密顿量对角元作为零阶哈密顿量 H0
    diags = get_diags(basis, ham)

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds")
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

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
    n_states::Int=5
) where {Ti,Tv,K,V}

    println("\n--- Starting Quantum Subspace Expansion (QSE) ---\n")

    N_pool = length(pool)
    N_sub  = N_pool + 1
    println("Operator pool size: $(N_pool)")

    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed ham_otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)
    
    print("Pre-compiling operator OTFs ... ")
    time_ops = @elapsed pool_otfs = [OTF(basis, op) for op in pool]
    @printf("Done in %.4f seconds\n", time_ops)

    hvec!  = (src, dst) -> hvec_otf!(basis, ham_otf, src, dst)
    get_V! = (k, dst) -> begin
        if k == 1
            copyto!(dst, v0)
        else
            hvec_otf!(basis, pool_otfs[k-1], v0, dst)
        end
    end

    print("Building S and H Matrices ... ")
    v_i_buf  = zeros(Tv, basis.dim)
    v_j_buf  = zeros(Tv, basis.dim)
    hv_j_buf = zeros(Tv, basis.dim)
    S_mat    = zeros(Tv, N_sub, N_sub)
    H_mat    = zeros(Tv, N_sub, N_sub)

    time_ops = @elapsed begin
        for j in 1:N_sub
            # 1. 生成列向量 |V_j> 存入 v_j_buf
            get_V!(j, v_j_buf)
            # 2. 计算 H|V_j> 存入 hv_j_buf
            hvec!(v_j_buf, hv_j_buf)
            # 3. 扫过行索引 i (利用厄米对称性, 只算上三角 i <= j)
            for i in 1:j
                if i == j
                    # 对角元直接自己和自己内积
                    S_mat[j, j] = real(dot(v_j_buf, v_j_buf))
                    H_mat[j, j] = real(dot(v_j_buf, hv_j_buf))
                else
                    # 重新生成行向量 |V_i> (这就是牺牲时间换取空间的核心)
                    get_V!(i, v_i_buf)
                    
                    s_val = real(dot(v_i_buf, v_j_buf))
                    h_val = real(dot(v_i_buf, hv_j_buf))
                    # 对称赋值
                    S_mat[i, j] = s_val; S_mat[j, i] = s_val
                    H_mat[i, j] = h_val; H_mat[j, i] = h_val
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
    N_valid   = length(valid_idx)
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
    states = [i == 1 ? "000 (GS)" : @sprintf("%03d", i-1) for i in 1:n_print]
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


function run_ssvqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0s::Vector{Vector{Tv}},
    weights::Vector{Float64},
    e_scales::Vector{Float64};
    net::String="agg",
    x0::Vector{Float64}=Float64[],
    options::VQE_OPTIONS=VQE_OPTIONS()
) where {Ti,Tv,K,V}
    K_states = length(v0s)
    @assert length(weights) == K_states "Number of weights must match number of initial states"
    @assert length(e_scales) == K_states "Number of energy scales must match number of initial states"

    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")
    println("SSVQE Target States: $(K_states)\n")

    # 复用内存缓存
    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successfully Generate Pool AGG in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successfully Generate Pool OTF in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

    if !isempty(x0)
        @assert length(x0) == length(idxs)
    else
        x0 = zeros(Float64, length(pool))
    end

    obj_func = x -> begin
        if !isempty(options.save_path)
            jldopen(options.save_path, "w") do file
                file["x"] = x
            end
        end

        total_L = 0.0
        total_grad = zeros(Float64, length(x))
        max_δ²H = 0.0
        
        # 将各态的代价和梯度按权重累加
        for k in 1:K_states
            lv .= v0s[k]
            result = @timed energy_objective(f_hvec, f_tvec, f_grad, idxs, x, lv, rv)
            e_k, g_k, δ²H_k = result.value
            
            total_L += weights[k] * e_k
            total_grad .+= weights[k] .* g_k
            max_δ²H = max(max_δ²H, δ²H_k)
        end

        norm_g = norm(total_grad)
        target_L = sum(weights .* e_scales)
        error = total_L - target_L

        options.verbose > 0 && show_optimze(total_L, norm_g, max_δ²H, error)
        
        return total_L, total_grad
    end

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
end


function run_adapt_ssvqe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0s::Vector{Vector{Tv}},
    weights::Vector{Float64},
    e_scales::Vector{Float64};
    net::String="agg",
    amplitudes::Vector{Float64}=Float64[], 
    selec_idxs::Vector{Int64}=Int64[],
    adapt_options::ADAPT_OPTIONS=ADAPT_OPTIONS(),
    vqe_options::VQE_OPTIONS=VQE_OPTIONS(ftol=1.0e-10, maxiter=1000, verbose=1),
) where {Ti,Tv,K,V}

    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")
    println("SSVQE Target States: $(length(v0s))\n")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successfully Generate Pool AGG in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successfully Generate Pool OTF in $(ret.time) seconds\n")
        f_hvec = (lvec, rvec) -> hvec_otf!(basis, ham_otf, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_otf, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_otf, idx, x, lvec, rvec)
    else
        error("Undefined NET name $(net)")
    end

    if !isempty(amplitudes) && !isempty(selec_idxs)
        @assert length(amplitudes) == length(selec_idxs)
    else
        amplitudes = Float64[]
        selec_idxs = Int64[]
    end

    _adapt_ssvqe(
        f_hvec, f_tvec, f_grad, idxs, v0s, weights, lv, rv, 
        e_scales, amplitudes, selec_idxs, adapt_options, vqe_options
    )
end


function generate_ssvqe_inputs(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,K,V}, 
    K_states::Int; 
    weight_decay::Float64 = 0.5
) where {Ti,Tv,K,V}
    """
    自动生成 SSVQE 所需的正交初态 v0s 和严格递减的 weights。
    策略: 基于哈密顿量对角元，选取能量最低的 K 个独立计算基矢(Slater行列式)。
    """

    println("--- Generating SSVQE Initial States ---")
    @assert K_states > 0 && K_states <= basis.dim "K_states must be within basis dimension"

    # 1. 获取对角元 (零阶能量)
    diags = get_diags(basis, ham)
    
    # 2. 找到对角元能量最低的 K 个构型的索引
    # sortperm 会返回从小到大排序的索引集
    sorted_idxs = sortperm(diags)
    selected_idxs = sorted_idxs[1:K_states]

    # 3. 构造正交初始态 (v0s)
    v0s = Vector{Vector{Tv}}(undef, K_states)
    for k in 1:K_states
        v = zeros(Tv, basis.dim)
        idx = selected_idxs[k]
        v[idx] = 1.0  # 设置为计算基矢 (One-hot 向量)，天然相互正交
        v0s[k] = v
        
        # 打印选出的基矢能量，用于物理检查 (比如基态是不是 HF 态)
        @printf("  State %d -> Basis Index: %-8d Zero-order Energy: %.6f\n", k, idx, diags[idx])
    end

    # 4. 构造递减权重 (weights)
    # 使用指数衰减策略: 1.0, 0.5, 0.25... (归一化以防止梯度爆炸)
    raw_weights = [weight_decay^(k-1) for k in 1:K_states]
    weights = raw_weights ./ sum(raw_weights)

    # 5. 生成对应的 e_scales (用于打印误差参考，如果没有 FCI 参考可以设为零阶能量)
    e_scales = [diags[idx] for idx in selected_idxs]

    println("  Weights   : ", round.(weights, digits=4))
    println("---------------------------------------\n")

    return v0s, weights, e_scales
end


function run_qeom(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}},
    v0::Vector{Tv};
    e_scales::Vector{Float64}=Float64[],
    S_tol::Float64=1e-8,
    n_states::Int=5
) where {Ti,Tv,K,V}

    println("\n--- Starting Quantum Equation-of-Motion (qEOM) ---\n")

    N_pool = length(pool)
    println("Operator pool size: $(N_pool)")

    print("Pre-compiling Ham OTF ... ")
    time_ops = @elapsed ham_otf = OTF(basis, ham)
    @printf("Done in %.4f seconds\n", time_ops)

    print("Pre-compiling operator OTFs ... ")
    time_ops = @elapsed pool_otfs = [OTF(basis, op) for op in pool]
    @printf("Done in %.4f seconds\n", time_ops)

    hvec! = (src, dst) -> hvec_otf!(basis, ham_otf, src, dst)
    pool_hvec! = (k, src, dst) -> hvec_otf!(basis, pool_otfs[k], src, dst)

    # 2. 准备 qEOM 核心的辅助态
    v0_tilde = zeros(Tv, basis.dim)
    hvec!(v0, v0_tilde)            # |v0_tilde> = H |v0>
    
    E_ref = real(dot(v0, v0_tilde)) # VQE 参考态能量

    print("Building qEOM M and S Matrices ... ")
    time_ops = @elapsed begin
        v_i       = zeros(Tv, basis.dim)
        v_i_tilde = zeros(Tv, basis.dim)
        v_j       = zeros(Tv, basis.dim)
        v_j_tilde = zeros(Tv, basis.dim)
        hv_j      = zeros(Tv, basis.dim)
        S_mat     = zeros(Tv, N_pool, N_pool)
        M_mat     = zeros(Tv, N_pool, N_pool)

        for j in 1:N_pool
            # 生成列向基底
            pool_hvec!(j, v0, v_j)               # |v_j> = O_j |v0>
            pool_hvec!(j, v0_tilde, v_j_tilde)   # |v_j_tilde> = O_j |v0_tilde> = O_j H |v0>
            
            hvec!(v_j, hv_j)                     # H |v_j>
            
            for i in 1:j
                # 生成行向基底
                pool_hvec!(i, v0, v_i)
                pool_hvec!(i, v0_tilde, v_i_tilde)
                
                # 矩阵元计算 (利用多重换位子展开式)
                S_val = real(dot(v_i, v_j))
                
                term1 = real(dot(v_i, hv_j))       # <v_i | H | v_j>
                term2 = real(dot(v_i_tilde, v_j))  # <v_i_tilde | v_j>
                term3 = real(dot(v_i, v_j_tilde))  # <v_i | v_j_tilde>
                
                # qEOM 核心公式：M_ij = term1 - 0.5 * term2 - 0.5 * term3
                M_val = term1 - 0.5 * term2 - 0.5 * term3
                
                # 对称赋值
                S_mat[i, j] = S_val; S_mat[j, i] = S_val
                M_mat[i, j] = M_val; M_mat[j, i] = M_val
            end
        end
    end
    S_mat = Hermitian(S_mat)
    M_mat = Hermitian(M_mat)
    @printf("Done in %.4f seconds\n", time_ops)

    # 4. Canonical Orthogonalization (正则正交化处理)
    λ_S, U_S = eigen(S_mat)
    
    valid_idx = findall(x -> x > S_tol, λ_S)
    N_valid   = length(valid_idx)
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
            i <= N ? @sprintf("%.3e", abs((E_ref + ΔE_qeom[i]) - e_scales[i])) : "-"
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


function run_qpe(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,ComplexF64,K,V},
    v0::Vector{ComplexF64};
    net::String="agg",
    dτ::Float64=0.05, 
    max_step::Int=2000, 
    window::Bool=true
) where {Ti,K,V}
    """
    量子相位估值 (QPE) 原生复数版
    通过实时间演化 |ψ(t)> = exp(-iHt)|ψ(0)> 收集自相关函数，并进行 FFT 提取能谱。
    充分利用底层 hvec! 的复数支持，实现最高效的 RK4 积分。
    """
    println("============================================================================")
    println("--- Starting Quantum Phase Estimation (QPE) ---")
    @printf("Time step (dτ) : %.4f\n", dτ)
    @printf("Total steps    : %d\n", max_step)
    @printf("Energy Resol.  : %.4f\n\n", 2 * π / (max_step * dτ))

    if net == "agg"
        ret = @timed agg = AGG(basis, ham)
        println("Successfully Generate Ham AGG in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)
    elseif net == "otf"
        ret = @timed otf = OTF(basis, ham)
        println("Successfully Generate Ham OTF in $(ret.time) seconds\n")
        hvec! = (src, dst) -> hvec_otf!(basis, otf, src, dst)
    else
        error("Undefined NET name $(net)")
    end

    v   = copy(v0) 
    hv  = zeros(ComplexF64, basis.dim)
    tmp = zeros(ComplexF64, basis.dim)
    ks  = [zeros(ComplexF64, basis.dim) for _ in 1:4]

    # 闭包：应用算符 d|ψ>/dt = -i H |ψ> 
    # 因为底层支持复数，直接乘 -im 即可
    apply_m_iH! = (src, dst) -> begin
        hvec!(src, hv)
        @. dst = -im * hv
    end

    C_t = Vector{ComplexF64}(undef, max_step)

    print("Running Real-Time Evolution ... ")
    time_evo = @elapsed begin
        for step in 1:max_step
            # 记录自相关函数 C(t) = <v0 | v(t)>
            C_t[step] = dot(v0, v)

            # RK4 演化 (数学形式极其干净)
            apply_m_iH!(v, ks[1])

            @. tmp = v + 0.5 * dτ * ks[1]
            apply_m_iH!(tmp, ks[2])

            @. tmp = v + 0.5 * dτ * ks[2]
            apply_m_iH!(tmp, ks[3])

            @. tmp = v + dτ * ks[3]
            apply_m_iH!(tmp, ks[4])

            @. v += (dτ / 6.0) * (ks[1] + 2*ks[2] + 2*ks[3] + ks[4])
            
            # 维持数值稳定性，归一化
            normalize!(v)
        end
    end
    @printf("Done in %.4f seconds\n\n", time_evo)

    # 2. 信号处理：应用汉宁窗
    if window
        for i in 1:max_step
            C_t[i] *= 0.5 * (1 - cos(2 * π * (i - 1) / (max_step - 1)))
        end
    end

    # 3. 执行 FFT 提取相位/能谱
    spectrum = fft(C_t)
    power    = abs2.(spectrum)
    
    freqs    = fftfreq(max_step, 1.0 / dτ)
    energies = 2 .* π .* freqs

    # 4. 寻峰算法
    peaks = []
    for i in 2:(max_step-1)
        if power[i] > power[i-1] && power[i] > power[i+1]
            push!(peaks, (energies[i], power[i]))
        end
    end
    
    sort!(peaks, by=x->x[2], rev=true)

    println("--- QPE Extracted Energy Spectrum (Top Peaks) ---")
    @printf("  %-5s %-18s %-15s\n", "Peak", "Energy", "Relative Power")
    
    max_power = isempty(peaks) ? 1.0 : peaks[1][2]
    for (i, peak) in enumerate(peaks[1:min(5, length(peaks))])
        rel_power = peak[2] / max_power
        if rel_power > 1e-4 
            @printf("  %03d   % 15.10f    % 10.4f\n", i, peak[1], rel_power)
        end
    end
    println("============================================================================\n")

    return energies, power, peaks
end

