function run_rk4_ite(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}, e_scale::Float64;
    dt::Float64=1e-2, max_step::Int64=1000, tol::Float64=1e-8, per_print::Int=10,
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
    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=false)
    ws = [zeros(Tv, basis.dim) for _ in 1:5]
    v  = copy(v0)
    Hv = ws[1]

    c_e    = Ref(0.0)
    c_δe   = Ref(0.0)
    c_err  = Ref(0.0)
    c_δ²H  = Ref(0.0)
    c_step = Ref(0)

    e_hist = Float64[]
    δ²H_hist = Float64[]
    step = 0

    println("Performing ITE with 4-Runge-Kutta ... ")

    time_ops = @elapsed while step <= max_step
        step += 1

        funcs.hvec(v, Hv)
        vn  = norm(v) ^ 2
        Hvn = norm(Hv) ^ 2
        e   = real(dot(v, Hv)) / vn
        err = abs(e - e_scale)
        δ²H = max(0.0, Hvn / vn - e ^ 2)

        push!(e_hist, e)
        push!(δ²H_hist, δ²H)
        δe = step > 1 ? e_hist[end] - e_hist[end-1] : e_hist[end]
        c_e[], c_δe[], c_err[], c_δ²H[], c_step[] = e, δe, err, δ²H, step

        if step % per_print == 0 || step == 1
            @printf("  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e    Time: %.3f\n", step, e, err, δe, δ²H, step * dt)
        end

        abs(δe) < tol && break

        vt = ws[5]

        ws[1] .*= -1.0

        @. vt = v + ws[1] * dt / 2
        funcs.hvec(vt, ws[2])
        ws[2] .*= -1.0

        @. vt = v + ws[2] * dt / 2
        funcs.hvec(vt, ws[3])
        ws[3] .*= -1.0

        @. vt = v + ws[3] * dt
        funcs.hvec(vt, ws[4])
        ws[4] .*= -1.0

        @. v += (ws[1] + 2 * ws[2] + 2 * ws[3] + ws[4]) * dt / 6

        normalize!(v)
    end

    @printf("\nConverged in %.4f seconds with:\n  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e    Time: %.3f\n\n",
            time_ops, c_step[], c_e[], c_err[], c_δe[], c_δ²H[], c_step[] * dt)

    return e_hist[end]
end

function estimate_max_step(hvec::Function, dim::Int64, E_ground_guess::Float64; trial_step::Int64=40)
    v = randn(Float64, dim)
    w = zeros(Float64, dim)
    normalize!(v)

    λ_max = 0.0
    for _ in 1:trial_step
        hvec(v, w)

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

function run_euler_ite(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}, e_scale::Float64;
    dt::Float64=1e-2, max_step::Int64=5000, tol::Float64=1e-8, save_path::String="", per_print::Int=10,
) where {Ti,Tv,K,V}
    """
    一阶 Euler 虚时演化 (1 次 hvec/步):
    dpsi/dt = -H|psi>
    psi_new = psi - dt * H * psi
    """

    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=false)

    if iszero(dt)
        dt = estimate_max_step(funcs.hvec, basis.dim, e_scale)
    end

    v  = copy(v0)
    Hv = zeros(Tv, basis.dim)

    c_e    = Ref(0.0)
    c_δe   = Ref(0.0)
    c_err  = Ref(0.0)
    c_δ²H  = Ref(0.0)
    c_step = Ref(0)

    e_hist = Float64[]
    δ²H_hist = Float64[]
    step = 0

    println("Performing ITE with 1-Euler ... ")

    time_ops = @elapsed while step <= max_step
        step += 1

        if !isempty(save_path) && (step % 5 == 0)
            jldopen(save_path, "w") do file
                file["v"] = v
            end
            println("Successifully save wave function to $(save_path) at step = $(step)")
        end

        funcs.hvec(v, Hv)
        vn  = norm(v) ^ 2
        Hvn = norm(Hv) ^ 2
        e   = real(dot(v, Hv)) / vn
        err = abs(e - e_scale)
        δ²H = max(0.0, Hvn / vn - e ^ 2)

        push!(e_hist, e)
        push!(δ²H_hist, δ²H)
        δe = step > 1 ? e_hist[end] - e_hist[end-1] : e_hist[end]
        c_e[], c_δe[], c_err[], c_δ²H[], c_step[] = e, δe, err, δ²H, step

        if step % per_print == 0 || step == 1
            if dt <= 10
                @printf("  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e    Time: %.3f\n", step, e, err, δe, δ²H, step * dt)
            else
                @printf("  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e\n", step, e, err, δe, δ²H)
            end
        end

        abs(δe) < tol && break

        @. v -= dt * Hv

        # @. v += dt * (E * v - w)

        normalize!(v)
    end

    if dt <= 10
        @printf("\nConverged in %.4f seconds with:\n  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e    Time: %.3f\n\n",
                time_ops, c_step[], c_e[], c_err[], c_δe[], c_δ²H[], c_step[] * dt)
    else
        @printf("\nConverged in %.4f seconds with:\n  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e\n\n",
                time_ops, c_step[], c_e[], c_err[], c_δe[], c_δ²H[])
    end

    return e_hist[end]
end

function run_krylov_ite(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,TK,TV}, v0::Vector{Tv}, e_scale::Float64;
    dt::Float64=1.0, krylov_dim::Int=20, max_step::Int64=200, tol::Float64=1e-8,
) where {Ti,Tv,TK,TV}
    """
    Krylov 子空间指数法虚时演化 (m 次 hvec/步):
    利用 Lanczos 算法构建 m 维子空间, 投影哈密顿量为三对角矩阵 Tm
    psi(τ + dt) ≈ V * exp(-dt * Tm) * e1
    """

    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=false)

    v = v0
    normalize!(v)

    V = [zeros(Tv, basis.dim) for _ in 1:krylov_dim]
    w = zeros(Tv, basis.dim)
    α = zeros(Float64, krylov_dim)
    β = zeros(Float64, krylov_dim)

    e_hist = Float64[]
    δ²H_hist = Float64[]
    step = 0

    println("Performing ITE with Krylov ... ")

    time_ops = @elapsed while step <= max_step
        step += 1

        # 1. 初始基向量
        copyto!(V[1], v)
        m_actual = krylov_dim

        # 2. Lanczos 迭代构建子空间
        for j in 1:krylov_dim
            v_j = V[j]    # 极速获取引用, 类型为纯正的 Vector{Tv}
            funcs.hvec(v_j, w) # 这里 hvec! 接收的将是完美的纯向量, 毫无阻碍

            if j == 1
                vn  = norm(v_j) ^ 2
                rn  = norm(w) ^ 2
                e   = real(dot(v_j, w)) / vn
                δ²H = max(0.0, rn / vn - e ^ 2)
                push!(e_hist, e)
                push!(δ²H_hist, δ²H)
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

        δe = step > 1 ? e_hist[end] - e_hist[end-1] : e_hist[end]

        @printf("  Step: %-5.d   E: %.14f    Err: %.3e    δe: %.3e    δ²H: %.3e    Time: %.3f\n", step, e_hist[end], abs(e_hist[end] - e_scale), δe, δ²H_hist[end], step * dt)

        abs(δe) < tol && break

        # 3. 构造子空间投影的三对角矩阵 Tm 并求指数
        Tm = SymTridiagonal(α[1:m_actual], β[1:m_actual-1])
        U = exp(-dt * Matrix(Tm))

        c = U[:, 1]

        # 4. 映射回全空间
        fill!(v, 0.0)
        for j in 1:m_actual
            v_j = V[j]
            @. v += c[j] * v_j
        end

        normalize!(v)
    end

    @printf("\nConverged in %.4f seconds\n\n", time_ops)

    return e_hist[end]
end

