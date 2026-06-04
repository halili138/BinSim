function run_vqrte_forward(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv},
    e_scale::Float64;
    dt::Float64=0.01,
    max_step::Int=500,
    tikhonov_eps::Float64=1e-4, # 正则化阻尼因子
    per_print::Int=10,
) where {Ti,Tv,TK,TV}
    @assert Tv == ComplexF64 "VQRTE requires Tv=ComplexF64 for Hamiltonian, Pool, and initial state!"

    println("============================================================================")
    println("--- VQRTE with Strict Trajectory Tracking (RK4 Engine + forward method) ---")

    f_hvec = get_hvec(basis, ham, is_time=false)
    print("Pre-compiling Pool OTF ... ")
    time_ops = @elapsed pool_otf = OTF(basis, pool)
    @printf("Done in %.4f seconds\n", time_ops)
    f_expm = (idx, θ, vec) -> expm_svd!(basis, pool_otf, idx, θ, vec)
    f_tvec = (idx, lvec, rvec) -> tvec_svd!(basis, pool_otf, idx, lvec, rvec)
    f_batchexpm = (idx, θ, mat, N, i) -> batch_expm_svd!(basis, pool_otf, idx, θ, mat, N, i)

    N = length(pool)

    e_hist = Float64[]
    fid_hist = Float64[]
    cond_hist = Float64[]

    x = zeros(Float64, N)
    x_hist = [copy(x)]
    xs = [zeros(Float64, N) for _ in 1:5]
    xt = xs[5]

    vs = [zeros(ComplexF64, basis.dim) for _ in 1:N]
    D  = zeros(ComplexF64, N, basis.dim)
    ws = [zeros(ComplexF64, basis.dim) for _ in 1:5]

    v_exact = copy(v0)
    v = ws[1]
    Hv = ws[2]
    vt = ws[5]

    z0 = QubitOperatorAABB([(0, "Z")], 1.0, Ti, Tv)
    z0_otf = OTF(basis, z0)
    f_z0vec = (lv, rv) -> begin
        hvec_otf!(basis, z0_otf, lv, rv)
        return real(dot(lv, rv))
    end
        
    function compute_xdot_and_energy!(x_in, dx_out)
        v .= v0
        for k in 1:N
            vs[k] .= v
            f_expm(k, x_in[k], v)
        end

        f_hvec(v, Hv)
        E_val = real(dot(v, Hv))

        # # 当你把 @view D[k, :] 传给底层的 f_tvec 时, Julia 会把这个视图的首地址(即 D[k, 1] 的地址)作为一个裸指针(Tv*)丢给 C++
        # # 但是, C++ 里的 f_tvec 函数根本不知道什么叫"视图", 它认为传入的是一个绝对连续的一维数组!
        # # 结果就是所有的导数态不仅没有横向写入 D 的行中, 反而竖着把其他的波函数全部覆盖污染了!
        # for k in 1:N
        #     f_tvec(k, vs[k], @view D[k, :]) 
        # end
        for k in 1:N
            # 1. 让 C++ 将结果写入绝对连续的一维缓存 ws[3]
            f_tvec(k, vs[k], ws[3]) 
            # 2. 让 Julia 将缓存安全地 broadcast 赋值到 D 的行中
            D[k, :] .= ws[3]
        end

        for j in 1:N
            f_batchexpm(j, x_in[j], D, N, j) 
        end

        # Re(D* Hv) ≡ Re(D Hv*)
        @. Hv = conj(-im * (Hv - E_val * v))     
        # (N x dim) * (dim x 1) -> N x 1   
        V = real.(D * Hv)
        # (N x dim) * (dim x N) -> N x N
        M = real.(D * D')       

        if dx_out === xs[1]
            push!(cond_hist, cond(M))
        end

        if any(isnan, M) || any(isinf, M)
            println("WARNING: M matrix contains NaN or Inf! Returning zero update.")
            dx_out .= 0.0
            return E_val
        end

        # 2. Tikhonov 正则化 (Shift)
        # 对角线上加上一个极小的阈值 (例如 1e-12)
        # 这就像在矩阵的极度病态深渊里垫了一层钢板, 强行将其从奇异(Inf)拉回到正定!
        # 加上这微小的一点, LAPACK 的 eigen 就再也不会崩溃了
        # 这里为了不破坏物理轨迹, 我们加的值非常小 (比如 1e-12)
        shift_val = 1e-12
        for i in 1:N
            M[i, i] += shift_val
        end

        # # 在 N > 4000 时, SVD 的 O(N^3) 开销是极其恐怖的
        # # 在变分量子动力学中, 由于 M = Re(D*D^\dagger), 它在数学上绝对是一个实对称的半正定矩阵 (Symmetric Positive Semi-Definite)
        # # 对于实对称矩阵, 求解特征值分解(Eigen Decomposition)的速度远快于全量 SVD 分解, 而且在 LAPACK 中, 对称特征值分解(dsyevd, 分治法)的多线程并行效率极高
        # # 告诉 Julia 这是一个对称矩阵, 触发极速的对称求解器
        # # (D*D' 在数值上可能有 1e-16 的不对称, Symmetric 会强制取上三角)
        # F = eigen(Symmetric(M)) 
        
        # # 特征值可能因为数值误差出现微小的负数, 截断它们
        # inv_S = [val > sv_tol ? 1.0 / val : 0.0 for val in F.values]
        
        # # 伪逆重组： M_pinv = V * D^{-1} * V^T
        # M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        # dx_out .= M_pinv * V

        F = eigen(Symmetric(M))
        inv_S = [1.0 / (abs(val) + tikhonov_eps) for val in F.values]
        M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        dx_out .= M_pinv * V

        return E_val
    end

    @printf("  Step          Energy         Error      Trj Fid      cond(M)      |θ_dot|     Time\n")
    time_ops = @elapsed for step in 1:max_step
        # @time begin
        e_curr = compute_xdot_and_energy!(x, xs[1])
        push!(e_hist, e_curr)

        @. xt = x + 0.5 * dt * xs[1]
        compute_xdot_and_energy!(xt, xs[2])

        @. xt = x + 0.5 * dt * xs[2]
        compute_xdot_and_energy!(xt, xs[3])

        @. xt = x + dt * xs[3]
        compute_xdot_and_energy!(xt, xs[4])

        @. x += dt / 6 * (xs[1] + 2 * xs[2] + 2 * xs[3] + xs[4])
        push!(x_hist, copy(x))

        rk4_step!(f_hvec, v_exact, vt, ws, -im, dt)

        v .= v0
        for k in 1:N
            f_expm(k, x[k], v)
        end

        fid = abs2(dot(v, v_exact))
        push!(fid_hist, fid)
        println(f_z0vec(v, vt))

        if step % per_print == 0 || step == 1
            cond_M = isempty(cond_hist) ? NaN : cond_hist[end]
            @printf("  %04d    % 15.10f    %.3e    %.6f    %9.3e    %.3e    %.4g\n",
                step, e_curr, abs(e_curr - e_scale), fid, cond_M, norm(xs[1]), step * dt)
        end
        # end
    end

    @printf("\nNative Complex VQRTE (RK4) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x_hist, e=e_hist, fid=fid_hist, cond=cond_hist)
end


function run_adapt_vqrte_tfim_forward(
    basis::BasisManager, 
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    pool::Vector{BinaryQubitAABB{Ti,Tv,TK,TV}},
    v0::Vector{Tv};
    dt::Float64=0.01, max_step::Int=500,
    adapt_tol::Float64=1e-3,    # L^2 误差的容忍度 (流形防泄漏阈值)
    max_ansatz::Int=150,        # 池子动态生长上限
    tikhonov_eps::Float64=1e-4, 
    per_print::Int=10
) where {Ti,Tv,TK,TV}
    println("============================================================================")
    println("--- ADAPT-VQRTE (Forward Batching) with McLachlan Trajectory Tracking ---")

    f_hvec = get_hvec(basis, ham, is_time=false)
    pool_otf = OTF(basis, pool)
    
    f_expm = (idx, θ, vec) -> expm_svd!(basis, pool_otf, idx, θ, vec)
    f_tvec = (idx, lv, rv) -> tvec_svd!(basis, pool_otf, idx, lv, rv)
    f_batchexpm = (idx, θ, mat, N, i) -> batch_expm_svd!(basis, pool_otf, idx, θ, mat, N, i)
    f_batchgrad = (lv, rv, g, x) -> batch_grad_svd(basis, pool_otf, x, lv, rv, g)

    z0 = QubitOperatorAABB([(0, "Z")], 1.0, Ti, Tv)
    z0_otf = OTF(basis, z0)
    f_z0 = (lv, rv) -> begin
        hvec_otf!(basis, z0_otf, lv, rv)
        return real(dot(lv, rv))
    end

    z0z1 = QubitOperatorAABB([(0, "Z"), (1, "Z")], 1.0, Ti, Tv)
    z0z1_otf = OTF(basis, z0z1)
    f_z0z1 = (lv, rv) -> begin
        hvec_otf!(basis, z0z1_otf, lv, rv)
        return real(dot(lv, rv))
    end


    active_idxs = Int[]
    x = Float64[]

    # ==========================================================
    # 极度干净的零分配预分配内存池
    # ==========================================================
    # 最大的革命：D 矩阵自带了所有切向量
    D_full = zeros(ComplexF64, max_ansatz, basis.dim)
    vs     = [zeros(ComplexF64, basis.dim) for _ in 1:max_ansatz]
    
    xs_full = [zeros(Float64, max_ansatz) for _ in 1:4] # RK4 的 4 步速度缓存
    
    v_exact = copy(v0)
    ws = [zeros(ComplexF64, basis.dim) for _ in 1:5]
    v = ws[1]; Hv = ws[2]; R_vec = zeros(ComplexF64, basis.dim) # 独立分配 R_vec 更安全
    
    zero_amp   = zeros(Float64, length(pool))
    zero_grads = zeros(ComplexF64, length(pool))

    # ==========================================================
    # 核心闭包：极速推导 D 矩阵、M、V、x_dot 与 L^2 误差
    # ==========================================================
    function eval_kinematics!(x_val, dx_out)
        N_act = length(x_val)
        
        # 1. 顺水推舟：一趟完成波函数演化和状态备份
        v .= v0
        for k in 1:N_act
            vs[k] .= v
            f_expm(active_idxs[k], x_val[k], v)
        end
        
        # 2. 目标演化方向
        f_hvec(v, Hv)
        E_curr = real(dot(v, Hv))
        @. Hv = -im * (Hv - E_curr * v)  # 剔除动力学相位的精确薛定谔演化

        if N_act == 0
            dx_out .= 0.0
            return norm(Hv) ^ 2, E_curr
        end

        D_act = @view D_full[1:N_act, :]

        # 3. 极速组装切空间 D 矩阵
        for k in 1:N_act
            f_tvec(active_idxs[k], vs[k], ws[3]) 
            D_act[k, :] .= ws[3]
        end

        # 调用底层的 SIMD 批处理怪兽
        for j in 1:N_act
            f_batchexpm(active_idxs[j], x_val[j], D_full, max_ansatz, j) 
        end

        # 4. 全局矩阵投影 (零多余分配)
        @. ws[4] = conj(Hv)
        V_act = real.(D_act * ws[4])
        M_act = real.(D_act * D_act') 

        # Tikhonov 刚性装甲
        shift_val = 1e-12
        for i in 1:N_act
            M_act[i, i] += shift_val
        end

        # # 5. 极速对称特征值分解求伪逆
        F = eigen(Symmetric(M_act))
        inv_S = [1.0 / (abs(val) + tikhonov_eps) for val in F.values]
        M_pinv = F.vectors * Diagonal(inv_S) * F.vectors'
        dx_out .= M_pinv * V_act

        # 6. McLachlan 误差（几何投影余弦定理，零额外开销）
        L2_err = norm(Hv) ^ 2 - dot(dx_out, V_act)
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
            D_act  = @view D_full[1:N_act, :]
            
            # 1. 神乎其技的残差提取：|R> = |Hv> - D^T * \dot{x}
            R_vec .= Hv
            if N_act > 0
                mul!(R_vec, transpose(D_act), dx_act, -1.0, 1.0)
            end
            
            # 2. 扫池子寻找“救世主”算符
            f_batchgrad(v, R_vec, zero_grads, zero_amp)
            
            # 获取最大的真实物理投影梯度
            grads_real = abs.(real.(zero_grads))
            max_grad, max_idx = findmax(grads_real)
            
            # ==========================================================
            # 【核心防御机制 1：池子枯竭拦截】
            # 如果全池子的投影都极小，说明现有的池子已经无法进一步拟合残差。
            # 即使 L2_error 依然 > adapt_tol，也必须强行退出，否则会加一堆垃圾算符。
            # ==========================================================
            if max_grad < 1e-6
                # println("  -> Pool functionally exhausted, break ADAPT")
                break
            end
            
            # ==========================================================
            # 【核心防御机制 2：防贪心停滞拦截】
            # 如果程序试图连续两次添加同一个算符，说明该算符的更新速度被 sv_tol 截断了。
            # 此时残差已死锁，必须强行退出让 RK4 时间往前推演以打破死锁。
            # ==========================================================
            if !isempty(active_idxs) && max_idx == active_idxs[end]
                # println("  -> Stagnation detected (repeated operator), break ADAPT")
                break
            end
            
            # 3. 膨胀 Ansible (初始化参数为 0)
            push!(active_idxs, max_idx)
            push!(x, 0.0)
            
            # 4. 重新校验流形是否已封堵
            k1_new = @view xs_full[1][1:length(active_idxs)]
            L2_error, E_curr = eval_kinematics!(x, k1_new)
        end
        
        # ==========================================================
        # Phase 2: RK4 物理真实时间推演
        # ==========================================================
        N_act = length(active_idxs)
        if N_act > 0
            x_act  = @view x[1:N_act]
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
        rk4_step!(f_hvec, v_exact, ws[5], ws, -im, dt)
        
        v .= v0
        for i in 1:N_act
            f_expm(active_idxs[i], x[i], v)
        end

        z0exp       = f_z0(v, ws[5])
        z0exp_e     = f_z0(v_exact, ws[5])
        z0z1exp     = f_z0z1(v, ws[5])
        z0z1exp_e   = f_z0z1(v_exact, ws[5])

        fid = abs2(dot(v, v_exact))
        
        if step % per_print == 0 || step == 1
            @printf("  %04d    %03d  % 15.10f    %.6f    %.3e    %4.4g    \t % .4f    % .4f    % .4f    % .4f\n",
                    step, N_act, E_curr, fid, L2_error, step * dt, z0exp, z0exp_e, z0z1exp, z0z1exp_e)
        end
    end

    @printf("\nADAPT-VQRTE (Forward Batching) completed in %.4f seconds.\n", time_ops)
    println("============================================================================\n")

    return (x=x, active_idxs=active_idxs)
end

