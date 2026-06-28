ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 1)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/cunetwork.jl")


function test_hvec(mole, nsteps)
    basis   = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham     = JW_hamiltonian(mole)
    h_v     = get_hf(basis, mole.nelec)
    d_v     = CuArray{Float64,1,CUDA.DeviceMemory}(h_v)
    d_w     = CUDA.zeros(Float64, basis.dim)
    h_funcs = OTF_Functions(basis, ham, typeof(ham)[])
    d_funcs = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool, time_print=true)

    for _ in 1:nsteps
        d_funcs.hvec(d_v, d_w)
        println("")
        @. d_v -= 1e-2 * d_w
	println(real(dot(d_v, d_w)))
        normalize!(d_v)
    end
end


function test_cost_fun(mole, nsteps)
    basis       = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham         = JW_hamiltonian(mole)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=false)
    pool        = FEB(orbs)
    h_v         = get_hf(basis, mole.nelec)
    d_v         = CuArray{Float64,1,CUDA.DeviceMemory}(h_v)
    d_w         = CUDA.zeros(Float64, basis.dim)
    h_funcs     = OTF_Functions(basis, ham, pool, time_print=false)
    d_funcs     = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool, time_print=false)

    h_v0_idxs   = findall(x -> x != 0, h_v) 
    h_v0_vals   = h_v[h_v0_idxs]
    d_v0_idxs   = CuArray{Int64,1,CUDA.DeviceMemory}(h_v0_idxs)
    d_v0_vals   = CuArray{Float64,1,CUDA.DeviceMemory}(h_v0_vals)
    amplitudes  = rand(Float64, length(pool))

    for _ in 1:nsteps
        @time begin
            fill!(d_v, 0.0)
            d_v[d_v0_idxs] .= d_v0_vals

            for (i, t) in enumerate(amplitudes)
                d_funcs.expm(i, t, d_v)
            end

            d_funcs.hvec(d_v, d_w)

            real(dot(d_v, d_w)) / norm(d_v) ^ 2
        end
    end
end


function test_trotter(mole, nsteps)
    basis       = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=false)
    pool        = FEB(orbs)
    h_v         = get_hf(basis, mole.nelec)
    d_v         = CuArray{Float64,1,CUDA.DeviceMemory}(h_v)
    h_funcs     = OTF_Functions(basis, eltype(pool)(), pool, time_print=false)
    d_funcs     = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool, time_print=false)

    h_v0_idxs   = findall(x -> x != 0, h_v) 
    h_v0_vals   = h_v[h_v0_idxs]
    d_v0_idxs   = CuArray{Int64,1,CUDA.DeviceMemory}(h_v0_idxs)
    d_v0_vals   = CuArray{Float64,1,CUDA.DeviceMemory}(h_v0_vals)
    amplitudes  = rand(Float64, length(pool))

    for _ in 1:nsteps
        @time begin
            fill!(d_v, 0.0)
            d_v[d_v0_idxs] .= d_v0_vals

            for (i, t) in enumerate(amplitudes)
                d_funcs.expm(i, t, d_v)
            end

            sync_device!()
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    mole.orbsym = Int64.(mole.orbsym .% 10)

    ARGS[3] == "1" && test_hvec(mole, 10)
    ARGS[3] == "2" && test_cost_fun(mole, 10)
    ARGS[3] == "3" && test_trotter(mole, 10)
end

