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
    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successifully Generate Pool AGG in $(ret.time) seconds")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successifully Generate Pool OTF in $(ret.time) seconds\n")
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

    return @time optimze_fg!(x0, obj_func, options.optimizer, options.options, options.verbose)
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

    println("Num symmetry allowed elements: $(basis.dim)\n")
    println("Operator pool size: $(length(pool))\n")

    lv = zeros(Tv, basis.dim)
    rv = zeros(Tv, basis.dim)
    idxs = [i for i in eachindex(pool)]

    if net == "agg"
        ret = @timed ham_agg = AGG(basis, ham)
        println("Successifully Generate Ham AGG in $(ret.time) seconds")
        print_info(ham_agg)
        ret = @timed pool_net = NET(basis, pool)
        println("Successifully Generate Pool AGG in $(ret.time) seconds")
        f_hvec = (lvec, rvec) -> hvec_direct_agg!(basis, ham_agg, lvec, rvec)
        f_tvec = (idx, x, vec) -> tvec_svd!(basis, pool_net, idx, x, vec)
        f_grad = (idx, x, lvec, rvec) -> return grad_svd(basis, pool_net, idx, x, lvec, rvec)
    elseif net == "otf"
        ret = @timed ham_otf  = OTF(basis, ham)
        println("Successifully Generate Ham OTF in $(ret.time) seconds\n")
        ret = @timed pool_otf = OTF(basis, pool)
        println("Successifully Generate Pool OTF in $(ret.time) seconds\n")
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


function estimate_max_eigen(hvec!, dim::Int; maxiter::Int=20)
    v = randn(Float64, dim)
    w = zeros(Float64, dim)
    normalize!(v)
    
    λ_max = 0.0
    # 迭代 15-20 次即可获得相当准确的最高本征值
    for _ in 1:maxiter
        hvec!(v, w)
        λ_max = real(dot(v, w))
        nw = norm(w) 
        @. v = w / nw
    end
    
    println("Estimated λ_max: $λ_max")

    return λ_max
end


# function estimate_max_step(hvec!, dim::Int, E_ground_guess::Float64)
#     v = rand(Float64, dim)
#     w = zeros(Float64, dim)
#     normalize!(v)
    
#     λ_max = 0.0
#     # 迭代 15-20 次即可获得相当准确的最高本征值
#     for _ in 1:20
#         hvec!(v, w)
#         λ_max = real(dot(v, w))
#         nw = norm(w) 
#         @. v = w / nw
#     end
    
#     # 根据理论公式计算绝对极限步长
#     dτ_limit = 2.0 / (λ_max + E_ground_guess)
    
#     println("Estimated λ_max: $λ_max")
#     println("Theoretical dτ limit: $dτ_limit")
    
#     # 实际运行时为了安全，通常取极限值的 0.95 倍
#     return dτ_limit * 0.95 
# end

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
        
        # 此时得到的本征值是平移后的，需要加回来
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
    dτ::Float64=1.0,       # Krylov 允许极其激进的大步长 (甚至可以直接设为 1.0)
    krylov_dim::Int=20,    # Krylov 子空间维度 m
    max_step::Int64=200,   # 因为单步迈得远，总步数会大幅减少
    tol::Float64=1e-8,
    net::String="agg"
) where {Ti,Tv,TK,TV}
    """
    Krylov 子空间指数法虚时演化 (m 次 hvec/步):
    利用 Lanczos 算法构建 m 维子空间，投影哈密顿量为三对角矩阵 Tm
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

    v = copy(v0)
    normalize!(v)

    # 【内存预分配】
    # V 矩阵存储 Krylov 基底 (按列排布以利用 Julia 的列主序连续内存)
    V = zeros(Tv, basis.dim, krylov_dim)
    w = zeros(Tv, basis.dim) # 用于存放 H|v_j>
    α = zeros(Float64, krylov_dim)
    β = zeros(Float64, krylov_dim)

    E_hist   = Float64[]
    dH_hist  = Float64[]

    step = 0
    while step <= max_step
        step += 1
        
        # 1. 初始基向量
        copyto!(view(V, :, 1), v)
        m_actual = krylov_dim

        # 2. Lanczos 迭代构建子空间
        for j in 1:krylov_dim
            v_j = V[:, j]
            hvec!(v_j, w) # w = H|v_j>

            # 第一步时提取当前态的能量期望和方差 (因为 v_1 就是当前的波函数 v)
            if j == 1
                rn = norm(w)^2
                E  = real(dot(v_j, w))
                dH = max(0.0, rn - E^2) # 这里 ln 恒为 1.0，因为已经 normalize
                push!(E_hist, E)
                push!(dH_hist, dH)
            end

            α[j] = real(dot(v_j, w))

            # Gram-Schmidt 正交化 (计算残差向量)
            @. w = w - α[j] * v_j
            if j > 1
                v_prev = view(V, :, j-1)
                @. w = w - β[j-1] * v_prev
            end

            # 再次施加完全正交化(Full Reorthogonalization)以抵抗浮点误差带来的基底坍塌
            for i in 1:j
                v_i = view(V, :, i)
                c = dot(v_i, w)
                @. w = w - c * v_i
            end

            norm_w = norm(w)
            
            # Krylov 空间提前闭合判定 (命中精确不变量子空间)
            if j < krylov_dim
                if norm_w < 1e-12
                    m_actual = j
                    break
                end
                β[j] = norm_w
                v_next = view(V, :, j+1)
                @. v_next = w / norm_w
            end
        end

        dE = step > 1 ? E_hist[end] - E_hist[end-1] : E_hist[end]

        @printf("  Step %03d    E %.14f    Err %.3e    dE %.3e    δ²H %.3e    τ %.2f\n",
                step, E_hist[end], abs(E_hist[end]-e_scale), dE, dH_hist[end], step * dτ)

        abs(dE) < tol && break

        # 3. 构造子空间投影的三对角矩阵 Tm 并求指数
        Tm = SymTridiagonal(α[1:m_actual], β[1:m_actual-1])
        # 将矩阵转为密集矩阵求 exp，由于维度极小(如20x20)，这一步耗时几乎为 0
        U = exp(-dτ * Matrix(Tm)) 
        
        # 演化后的新系数，就是 U 的第一列
        c = U[:, 1]

        # 4. 映射回全空间：|v_new> = V * c
        fill!(v, 0.0)
        for j in 1:m_actual
            v_j = view(V, :, j)
            @. v += c[j] * v_j
        end

        normalize!(v)
    end 
    
    println("  Converged at step $step\n")

    return E_hist[end]
end

