function rte_rk4_step!(f_hvec::Function, v::T, ws::Vector{T}, dt::Float64) where {Tv,T<:AbstractArray{Tv,1}}
    vt = ws[5]
    # k1 = -i * H * v_exact
    f_hvec(v, ws[1])
    e = real(dot(v, ws[1]))
    @. ws[1] = -im * (ws[1] - e * v)

    # k2 = -i * H * (v + dt/2 * k1)
    @. vt = v + ws[1] * dt / 2
    f_hvec(vt, ws[2])
    @. ws[2] = -im * (ws[2] - e * vt)

    # k3 = -i * H * (v + dt/2 * k2)
    @. vt = v + ws[2] * dt / 2
    f_hvec(vt, ws[3])
    @. ws[3] = -im * (ws[3] - e * vt)

    # k4 = -i * H * (v + dt * k3)
    @. vt = v + ws[3] * dt
    f_hvec(vt, ws[4])
    @. ws[4] = -im * (ws[4] - e * vt)

    # ψ(t + dt) = ψ(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)
    @. v += (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4]) * dt / 6

    normalize!(v)
end


function rte_rk4_step2!(f_hvec::Function, v::T, ws::Vector{T}, dt::Float64) where {Tv,T<:AbstractArray{Tv,1}}
    vt = ws[5]
    # k1 = -i * H * v_exact
    f_hvec(v, ws[1])
    e = real(dot(v, ws[1]))
    ws[1] *= -im

    # k2 = -i * H * (v + dt/2 * k1)
    @. vt = v + ws[1] * dt / 2
    f_hvec(vt, ws[2])
    ws[2] *= -im

    # k3 = -i * H * (v + dt/2 * k2)
    @. vt = v + ws[2] * dt / 2
    f_hvec(vt, ws[3])
    ws[3] *= -im

    # k4 = -i * H * (v + dt * k3)
    @. vt = v + ws[3] * dt
    f_hvec(vt, ws[4])
    ws[4] *= -im

    # ψ(t + dt) = ψ(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)
    @. v += (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4]) * dt / 6

    normalize!(v)
end


function run_vqrte_tdva(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}}, v0::Vector{Tv},
    obs_X::BinaryQubitAABB{Ti,Tv,TK,TV}, obs_Z::BinaryQubitAABB{Ti,Tv,TK,TV};
    dt::Float64=1e-2, max_step::Int=500, tikhonov_eps::Float64=1e-3, per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "TDVA requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"
    println("============================================================================")
    println("--- Time-Dependent Variational Algorithm (TDVA) ---")
    println("--- DOI: https://doi.org/10.1103/PhysRevX.7.021050 ---\n")

    funcs = OTF_Functions(basis, ham, pool, time_print=false)
    xfuncs = OTF_Functions(basis, obs_X, eltype(pool)[], info_print=false, time_print=false)
    zfuncs = OTF_Functions(basis, obs_Z, eltype(pool)[], info_print=false, time_print=false)
    obs_X_hist = Float64[]
    obs_Z_hist = Float64[]
    exact_obs_X_hist = Float64[]
    exact_obs_Z_hist = Float64[]

    N = length(pool)
    e_hist = Float64[]

    x = zeros(Float64, N)
    xs = [zeros(Float64, N) for _ in 1:5]
    xt = xs[5]

    vs = [zeros(Tv, basis.dim) for _ in 1:N]
    ws = [zeros(Tv, basis.dim) for _ in 1:5]
    D = zeros(Tv, N, basis.dim)

    ve = copy(v0)
    v = ws[1]
    Hv = ws[2]

    function compute_xdot_and_energy!(x_in, dx_out)
        v .= v0
        for k in 1:N
            vs[k] .= v
            funcs.expm(k, x_in[k], v)
        end

        funcs.hvec(v, Hv)
        E_val = real(dot(v, Hv))

        for k in 1:N
            funcs.tvec(k, vs[k], ws[3])
            D[k, :] .= ws[3]
        end

        for j in 1:N
            funcs.batchexpm(j, x_in[j], D, N, j)
        end

        @. Hv = conj(-im * (Hv - E_val * v))
        V = real.(D * Hv)
        M = real.(D * D')

        if any(isnan, M) || any(isinf, M)
            println("WARNING: M matrix contains NaN or Inf! Returning zero update.")
            dx_out .= 0.0
            return E_val, cond(M)
        end

        shift_val = 1e-12
        for i in 1:N
            M[i, i] += shift_val
        end

        F = eigen(Symmetric(M))
        inv_S = [1.0 / (abs(val) + tikhonov_eps) for val in F.values]
        M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        dx_out .= M_pinv * V

        return E_val, cond(M)
    end

    @printf("  Step        Energy           δE       Trj Fid     cond(M)       |dx|      Time\n")
    time_ops = @elapsed for step in 1:max_step
        e_curr, cond_M = compute_xdot_and_energy!(x, xs[1])
        push!(e_hist, e_curr)

        @. xt = x + 0.5 * dt * xs[1]
        compute_xdot_and_energy!(xt, xs[2])

        @. xt = x + 0.5 * dt * xs[2]
        compute_xdot_and_energy!(xt, xs[3])

        @. xt = x + dt * xs[3]
        compute_xdot_and_energy!(xt, xs[4])

        @. x += dt / 6 * (xs[1] + 2 * xs[2] + 2 * xs[3] + xs[4])

        rte_rk4_step!(funcs.hvec, ve, ws, dt)

        v .= v0
        for k in 1:N
            funcs.expm(k, x[k], v)
        end

        xfuncs.hvec(v, ws[5])
        push!(obs_X_hist, real(dot(v, ws[5])))
        zfuncs.hvec(v, ws[5])
        push!(obs_Z_hist, real(dot(v, ws[5])))
        xfuncs.hvec(ve, ws[5])
        push!(exact_obs_X_hist, real(dot(ve, ws[5])))
        zfuncs.hvec(ve, ws[5])
        push!(exact_obs_Z_hist, real(dot(ve, ws[5])))

        fid = abs2(dot(v, ve))

        if step % per_print == 0 || step == 1
            δe = step > 1 ? (e_hist[end] - e_hist[end-1]) : 0.0
            @printf("  %-4.d  % 15.10f    % 8.2e    %.6f    %8.2e    %8.2e    %-6.4g\n",
                step, e_curr, δe, fid, cond_M, norm(xs[1]), step * dt)
        end
    end

    @printf("\nTDVA (RK4) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    for i in eachindex(obs_X_hist)
        @printf("  step: %04d  x : % 8.4f  xe : % 8.4f  z : % 8.4f  ze : % 8.4f\n",
            i, obs_X_hist[i], exact_obs_X_hist[i], obs_Z_hist[i], exact_obs_Z_hist[i])
    end

end


function run_adapt_vqrte_tdva(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}}, v0::Vector{Tv};
    dt::Float64=1e-2, max_step::Int=500, adapt_tol::Float64=1e-3, max_ansatz::Int=150, tikhonov_eps::Float64=1e-4, per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv <: Complex
    println("============================================================================")
    println("--- ADAPT-VQRTE (Forward Batching) with McLachlan Trajectory Tracking ---")

    funcs = OTF_Functions(basis, ham, pool, time_print=false)
    measure_ops = [QubitOperatorAABB([(0, "Z")], 1.0, Ti, Tv), QubitOperatorAABB([(0, "Z"), (1, "Z")], 1.0, Ti, Tv)]
    measure_funcs = OTF_Functions(basis, BinaryQubitAABB{Ti,Tv,TK,TV}(), measure_ops, info_print=false, time_print=false)
    measures = zeros(Tv, length(measure_ops))
    measures_exact = zeros(Tv, length(measure_ops))

    D_full = zeros(Tv, max_ansatz, basis.dim) # D 矩阵自带了所有切向量
    xs_full = [zeros(Float64, max_ansatz) for _ in 1:4] # RK4 的 4 步速度缓存
    vs = [zeros(Tv, basis.dim) for _ in 1:max_ansatz]
    ws = [zeros(Tv, basis.dim) for _ in 1:5]
    ve = copy(v0)
    v = ws[1]
    Hv = ws[2]
    R = zeros(Tv, basis.dim)
    zx = zeros(Float64, length(pool))
    zg = zeros(Tv, length(pool))

    active_idxs = Int[]
    x = Float64[]

    # ==========================================================
    # 核心闭包：极速推导 D 矩阵、M、V、x_dot 与 L^2 误差
    # ==========================================================
    function eval_kinematics!(x_val, dx_out)
        N_act = length(x_val)

        v .= v0
        for k in 1:N_act
            vs[k] .= v
            funcs.expm(active_idxs[k], x_val[k], v)
        end

        funcs.hvec(v, Hv)
        E_curr = real(dot(v, Hv))
        @. Hv = -im * (Hv - E_curr * v)  # 剔除动力学相位的精确薛定谔演化

        if N_act == 0
            dx_out .= 0.0
            return norm(Hv)^2, E_curr
        end

        D_act = @view D_full[1:N_act, :]

        # 极速组装切空间 D 矩阵
        for k in 1:N_act
            funcs.tvec(active_idxs[k], vs[k], ws[3])
            D_act[k, :] .= ws[3]
        end

        # 调用底层的 SIMD 批处理
        for j in 1:N_act
            funcs.batchexpm(active_idxs[j], x_val[j], D_full, max_ansatz, j)
        end

        # 全局矩阵投影
        @. ws[4] = conj(Hv)
        V_act = real.(D_act * ws[4])
        M_act = real.(D_act * D_act')

        # Tikhonov 刚性装甲
        shift_val = 1e-12
        for i in 1:N_act
            M_act[i, i] += shift_val
        end

        # 极速对称特征值分解求伪逆
        F = eigen(Symmetric(M_act))
        inv_S = [1.0 / (abs(val) + tikhonov_eps) for val in F.values]
        M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        dx_out .= M_pinv * V_act

        # McLachlan 误差（几何投影余弦定理）
        L2_err = norm(Hv)^2 - dot(dx_out, V_act)

        return abs(L2_err), E_curr
    end

    @printf("  Step     N            E         Trj Fid      L²-Err      Time\n")

    time_ops = @elapsed for step in 1:max_step

        # ==========================================================
        # Phase 1: ADAPT 动态修补阶段 (拦截波函数泄漏)
        # ==========================================================
        k1_act = @view xs_full[1][1:length(active_idxs)]
        L2_error, E_curr = eval_kinematics!(x, k1_act)

        while L2_error > adapt_tol && length(active_idxs) < max_ansatz
            N_act = length(active_idxs)
            dx_act = @view xs_full[1][1:N_act]
            D_act = @view D_full[1:N_act, :]

            # 残差提取：|R> = |Hv> - D^T * \dot{x}
            R .= Hv
            if N_act > 0
                mul!(R, transpose(D_act), dx_act, -1.0, 1.0)
            end

            funcs.batchgrad(v, R, zg, zx)
            max_grad, max_idx = findmax(abs.(real.(zg))) # 获取最大的真实物理投影梯度

            # ==========================================================
            # 【核心防御机制 1：池子枯竭拦截】
            # 如果全池子的投影都极小，说明现有的池子已经无法进一步拟合残差。
            # 即使 L2_error 依然 > adapt_tol，也必须强行退出，否则会加一堆垃圾算符。
            # ==========================================================
            if max_grad < 1e-6
                println("  -> Pool functionally exhausted, break ADAPT")
                break
            end

            # ==========================================================
            # 【核心防御机制 2：防贪心停滞拦截】
            # 如果程序试图连续两次添加同一个算符，说明该算符的更新速度被 sv_tol 截断了。
            # 此时残差已死锁，必须强行退出让 RK4 时间往前推演以打破死锁。
            # ==========================================================
            if !isempty(active_idxs) && max_idx == active_idxs[end]
                println("  -> Stagnation detected (repeated operator), break ADAPT")
                break
            end

            # 膨胀 Ansible (初始化参数为 0)
            push!(active_idxs, max_idx)
            push!(x, 0.0)

            # 重新校验流形是否已封堵
            k1_new = @view xs_full[1][1:length(active_idxs)]
            L2_error, E_curr = eval_kinematics!(x, k1_new)
        end

        # ==========================================================
        # Phase 2: RK4 物理真实时间推演
        # ==========================================================
        N_act = length(active_idxs)
        if N_act > 0
            x_act = @view x[1:N_act]
            k1_act = @view xs_full[1][1:N_act]
            k2_act = @view xs_full[2][1:N_act]
            k3_act = @view xs_full[3][1:N_act]
            k4_act = @view xs_full[4][1:N_act]

            # 此时的 k1_act 已经是修补完美后的梯度，直接拿来用！
            x_temp = x_act .+ 0.5 .* dt .* k1_act
            eval_kinematics!(x_temp, k2_act)

            x_temp .= x_act .+ 0.5 .* dt .* k2_act
            eval_kinematics!(x_temp, k3_act)

            x_temp .= x_act .+ dt .* k3_act
            eval_kinematics!(x_temp, k4_act)

            # 推演物理时间
            @. x_act += dt / 6 * (k1_act + 2 * k2_act + 2 * k3_act + k4_act)
        end

        # ==========================================================
        # Phase 3: 保真度与轨迹监控
        # ==========================================================
        rte_rk4_step!(funcs.hvec, ve, ws, dt)

        v .= v0
        for i in 1:N_act
            funcs.expm(active_idxs[i], x[i], v)
        end

        measure_funcs.batchtran(v, v, measures)
        measure_funcs.batchtran(ve, ve, measures_exact)
        fid = abs2(dot(v, ve))

        if step % per_print == 0 || step == 1
            @printf("  %04d    %03d  % 15.10f    %.6f    %.3e    %4.4g \t % .4f    % .4f    % .4f    % .4f\n",
                step, N_act, E_curr, fid, L2_error, step * dt, real.(measures)..., real.(measures_exact)...)
        end
    end

    @printf("\nADAPT-VQRTE (Forward Batching) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x, active_idxs=active_idxs)
end


function run_rk4_rte(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, v0::Vector{Tv};
    dt::Float64=1e-2, max_step::Int64=1000, per_print::Int=100,
) where {Ti,Tv,TK,TV}
    @assert Tv <: Complex

    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,TK,TV}[], time_print=false)
    ws = [zeros(Tv, basis.dim) for _ in 1:5]
    v = copy(v0)
    Hv = ws[1]
    vt = ws[5]
    funcs.hvec(v, Hv)
    e = real(dot(v, Hv))

    c_e = Ref(0.0)
    c_δe = Ref(0.0)
    c_step = Ref(0)

    println("Performing RTE with 4-Runge-Kutta ... ")

    step = 0
    time_ops = @elapsed for step in 1:max_step
        @. ws[1] = -im * (ws[1] - e * v)

        @. vt = v + ws[1] * dt / 2
        funcs.hvec(vt, ws[2])
        @. ws[2] = -im * (ws[2] - e * vt)

        @. vt = v + ws[2] * dt / 2
        funcs.hvec(vt, ws[3])
        @. ws[3] = -im * (ws[3] - e * vt)

        @. vt = v + ws[3] * dt
        funcs.hvec(vt, ws[4])
        @. ws[4] = -im * (ws[4] - e * vt)

        @. v += (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4]) * dt / 6

        normalize!(v)

        funcs.hvec(v, ws[1])
        e_new = real(dot(v, Hv))
        δe = e_new - e
        e = e_new
        c_e[], c_δe[], c_step[] = e, δe, step

        if step % per_print == 0 || step == 1
            @printf("  Step: %-5.d   E: %.14f    δE: %.3e    Time: %.3f\n", step, e, δe, step * dt)
        end
    end

    @printf("\nConverged in %.4f seconds with:\n  Step: %-5.d   E: %.14f    δE: %.3e    Time: %.3f\n\n", time_ops, c_step[], c_e[], c_δe[], c_step[] * dt)
end


function run_krylov_rte(hvec!::Function, v0::Vector{ComplexF64};
    dt::Float64=0.05, krylov_dim::Int=20, max_step::Int64=2000, E_ref::Float64=0.0,
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


function run_vqrte_pvqd(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}}, v0::Vector{Tv},
    obs_X, obs_Z;
    dt::Float64=1e-2, max_step::Int=500, max_opt_steps::Int=50, tol_infidelity::Float64=1e-6, lr::Float64=1e-3, per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "p-VQD requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"
    println("============================================================================")
    println("--- projected Variational Quantum Dynamics (p-VQD) ---")
    println("--- Doi: https://doi.org/10.22331/q-2021-07-28-512 ---\n")

    funcs = OTF_Functions(basis, ham, pool, time_print=false)
    # ops_mx = BinaryQubitAABB{Ti,Tv,TK,TV}[]
    # ops_mz = BinaryQubitAABB{Ti,Tv,TK,TV}[]
    # nq = basis.norb * 2
    # for i in 0:nq-1
    #     push!(ops_mx, QubitOperatorAABB([(i, "X")], 1.0, Ti, Tv))
    #     push!(ops_mz, QubitOperatorAABB([(i, "Z")], 1.0, Ti, Tv))
    # end
    # obs_X = linearcombine(ops_mx, ones(Float64, nq), 0.0, 1e-12)
    # obs_Z = linearcombine(ops_mz, ones(Float64, nq), 0.0, 1e-12)
    # obs_X = QubitOperatorAABB([(i, "X") for i in 0:nq-1], 1.0, Ti, Tv)
    # obs_Z = QubitOperatorAABB([(i, "Z") for i in 0:nq-1], 1.0, Ti, Tv)

    xfuncs = OTF_Functions(basis, obs_X, eltype(pool)[], info_print=false, time_print=false)
    zfuncs = OTF_Functions(basis, obs_Z, eltype(pool)[], info_print=false, time_print=false)
    x_hist = Float64[]
    z_hist = Float64[]
    exact_x_hist = Float64[]
    exact_z_hist = Float64[]

    N = length(pool)

    # 物理参数与优化增量
    x = zeros(Float64, N)
    dx = zeros(Float64, N)
    x_dx = zeros(Float64, N) # 当前测试参数 x + dx

    # 存放底层返回的精确梯度
    zg = zeros(Tv, N)

    # 独立的态缓存 (避免与 RK4 内部的 ws 冲突)
    ve = copy(v0)
    v = copy(v0)
    vt = copy(v0)
    v_dx = copy(v0)
    Hv = copy(v0)
    ws = [zeros(Tv, basis.dim) for _ in 1:5] # 专供 rte_rk4_step! 使用的缓存

    # Adam 优化器内部状态
    m_adam = zeros(Float64, N)
    v_adam = zeros(Float64, N)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    @printf("  Step     OptSteps      Loss(1-F)       Energy           Trj Fid      Time\n")
    time_ops = @elapsed for step in 1:max_step

        # ==========================================================
        # Phase 1: 构建当前基准态 |ψ_w(t)>
        # ==========================================================
        v .= v0
        for k in 1:N
            funcs.expm(k, x[k], v)
        end

        # ==========================================================
        # Phase 2: 生成目标投影态 |ϕ(t+dt)> 
        # ==========================================================
        vt .= v
        rte_rk4_step!(funcs.hvec, vt, ws, dt)

        # 承接上一步的最优 dx 作为先验 (附录 D 技巧)
        fill!(m_adam, 0.0)
        fill!(v_adam, 0.0)

        opt_k = 0
        loss = 1.0

        # ==========================================================
        # Phase 3: Exact Backprop 梯度优化 (Adam)
        # ==========================================================
        for k in 1:max_opt_steps
            opt_k = k

            # 3.1 前向传播: 生成试验态 |ψ_{w+dx}>
            @. x_dx = x + dx
            v_dx .= v0
            for i in 1:N
                funcs.expm(i, x_dx[i], v_dx)
            end

            # 3.2 评估 Step-Infidelity
            ov = dot(vt, v_dx) # <ϕ(δt) | ψ_{w+dx}>
            fid_step = abs2(ov)
            loss = 1.0 - fid_step

            # 达到容忍度跳出迭代
            if loss < tol_infidelity
                break
            end

            # 3.3 反向传播: 计算算符池的精确全导数 ∇_x <vt | U(x_dx) | v0>
            # C++ 引擎将结果直接写入 zg
            funcs.batchgrad(vt, v0, zg, x_dx)

            # 3.4 组装损失函数梯度并执行 Adam 更新
            for i in 1:N
                # F = <ϕ | ψ> <ψ | ϕ>
                # ∂F/∂x_i = 2 Re( <ϕ | ∂_i ψ> * <ψ | ϕ> )
                # zg[i] 即为 <ϕ(δt) | ∂_i ψ_{w+dx}>
                # conj(ov) 即为 <ψ_{w+dx} | ϕ(δt)>
                grad_i = -2.0 * real(zg[i] * conj(ov))

                # Adam 动量与方差追踪
                m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * grad_i
                v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * grad_i^2

                m_hat = m_adam[i] / (1.0 - beta1^opt_k)
                v_hat = v_adam[i] / (1.0 - beta2^opt_k)

                # 参数更新
                dx[i] -= lr * m_hat / (sqrt(v_hat) + eps_adam)
            end
        end

        # ==========================================================
        # Phase 4: 物理时间推演
        # ==========================================================
        @. x += dx

        # ==========================================================
        # Phase 5: 物理量评估与监控
        # ==========================================================
        rte_rk4_step!(funcs.hvec, ve, ws, dt)

        v .= v0
        for i in 1:N
            funcs.expm(i, x[i], v)
        end

        funcs.hvec(v, Hv)
        E_curr = real(dot(v, Hv))
        fid_global = abs2(dot(v, ve))

        xfuncs.hvec(v, vt)
        push!(x_hist, real(dot(v, vt)))
        zfuncs.hvec(v, vt)
        push!(z_hist, real(dot(v, vt)))
        xfuncs.hvec(ve, vt)
        push!(exact_x_hist, real(dot(ve, vt)))
        zfuncs.hvec(ve, vt)
        push!(exact_z_hist, real(dot(ve, vt)))

        if step % per_print == 0 || step == 1
            @printf("  %04d      %03d         %.3e      % 15.10f    %.6f    %4.4g\n",
                step, opt_k, loss, E_curr, fid_global, step * dt)
        end
    end

    @printf("\np-VQD completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    for i in eachindex(x_hist)
        @printf("  step: %04d  x : % 8.4f  xe : % 8.4f  z : % 8.4f  ze : % 8.4f\n",
            i, x_hist[i], exact_x_hist[i], z_hist[i], exact_z_hist[i])
    end

    return x
end

