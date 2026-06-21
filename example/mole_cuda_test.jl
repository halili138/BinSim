_alg   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
_name  = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0

ENV["OMP_NUM_THREADS"] = 1
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "cores"

include("../jl/cunetwork.jl")

function run_euler_ite_cuda(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v::T, e_scale::Float64;
    dτ::Float64=0.1, max_step::Int64=5000, tol::Float64=1e-10,
) where {Ti,Tv,K,V,T<:AbstractArray{Tv,1}}
    funcs    = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,K,V}[])
    cu_funcs = CuOTF_Functions(basis, funcs.ham, funcs.pool, time_print=true)
    w        = CUDA.zeros(Tv, basis.dim)

    E_hist = Float64[]
    dH_hist = Float64[]

    step = 0
    while step <= max_step
        step += 1
        cu_funcs.hvec(v, w)
        ln = norm(v)^2
        rn = norm(w)^2
        E = real(dot(v, w)) / ln
        dH = max(0.0, rn / ln - E^2)
        push!(E_hist, E)
        push!(dH_hist, dH)

        dE = step > 1 ? abs(E_hist[end] - E_hist[end-1]) : abs(E_hist[end])

        @printf("  Step %04d  E %.14f  Err %.3e  dE %.3e  δ²H %.3e\n",
            step, E, abs(E - e_scale), dE, dH)

        dE < tol && break

        @. v -= dτ * w
        normalize!(v)
    end

    println("\n  Converged at step $step")
    return E_hist[end]
end

function run_vqe_cuda(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}, v0_idxs::T1, v0_vals::T2, e_scale::Float64;
    x0::Vector{Float64}=Float64[], options::VQE_OPTIONS=VQE_OPTIONS(verbose=3),
) where {Ti,Tv,K,V,T1<:AbstractArray{Int64,1},T2<:AbstractArray{Tv,1}}
    funcs    = OTF_Functions(basis, ham, pool)
    cu_funcs = CuOTF_Functions(basis, funcs.ham, funcs.pool)
    lv       = CUDA.zeros(Tv, basis.dim)
    rv       = CUDA.zeros(Tv, basis.dim)
    idxs     = [i for i in eachindex(pool)]

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

        fill!(lv, 0.0)
        lv[v0_idxs] .= v0_vals

        result = @timed energy_objective(cu_funcs.hvec, cu_funcs.expm, cu_funcs.backgrad, idxs, x, lv, rv)
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

    for i in eachindex(idxs)
        cu_funcs.expm(idxs[i], x_opt[i], lv)
    end

    return e_opt, lv, x_opt
end

function test1(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    
    run_euler_ite_cuda(basis, ham, CuArray{Float64,1,CUDA.DeviceMemory}(get_hf(basis, mole.nelec)), mole.e_scale,
        dτ       = 0.1, 
        max_step = 10, 
        tol      = 1e-10
    )
end

function test2(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)
    v0    = get_hf(basis, mole.nelec)

    h_v0_idxs = findall(x -> x != 0, v0) 
    h_v0_vals = v0[h_v0_idxs]
    d_v0_idxs = CuArray{Int64,1,CUDA.DeviceMemory}(h_v0_idxs)
    d_v0_vals = CuArray{Float64,1,CUDA.DeviceMemory}(h_v0_vals)
    
    run_vqe_cuda(basis, ham, pool, d_v0_idxs, d_v0_vals, mole.e_scale)
end

function test3(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)

    funcs    = OTF_Functions(basis, eltype(pool)(), pool)
    cu_funcs = CuOTF_Functions(basis, funcs.ham, funcs.pool)

    nparas = length(pool)
    amps   = rand(Float64, nparas)
    idxs   = [i for i in 1:nparas]
    lv     = CUDA.rand(Float64, basis.dim)

    for _ in 1:10
        @time begin
            for i in 1:nparas
                cu_funcs.expm(idxs[i], amps[i], lv)
            end
            sync_device!() 
        end
    end
end

function test4(name, ratio, basis)
    mole = Mole()
    mole.name  = name
    mole.ratio = ratio
    mole.basis = basis

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)

    funcs    = OTF_Functions(basis, ham, pool)
    cu_funcs = CuOTF_Functions(basis, funcs.ham, funcs.pool)

    nparas = length(pool)
    amps   = rand(Float64, nparas)
    idxs   = [i for i in 1:nparas]
    lv     = CUDA.rand(Float64, basis.dim)
    rv     = CUDA.zeros(Float64, basis.dim)

    for _ in 1:10
        @time begin
            for i in 1:nparas
                cu_funcs.expm(idxs[i], amps[i], lv)
            end
            cu_funcs.hvec(lv, rv)
            real(dot(lv, rv)) / norm(lv) ^ 2
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _alg == 1 && test1(_name, _ratio, _basis)
    _alg == 2 && test2(_name, _ratio, _basis)
    _alg == 3 && test3(_name, _ratio, _basis)
    _alg == 4 && test4(_name, _ratio, _basis)
end
