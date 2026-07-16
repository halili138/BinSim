ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 1)

include("../jl/cubinsim.jl")


function test_hvec(mole, nsteps)
    basis   = BasisManager(mole)
    ham     = JW_hamiltonian(mole)
    h_v     = get_hf(basis)
    d_v     = CuArray{Float64,1,CUDA.DeviceMemory}(h_v)
    d_w     = CUDA.zeros(Float64, basis.dim)
    h_funcs = OTF_Functions(basis, ham, typeof(ham)[])
    d_funcs = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool, time_print=false)

    for step in 1:10
        t_start = time_ns()

        d_funcs.hvec(d_v, d_w)
        energy  = real(dot(d_v, d_w))
        @. d_v -= 1e-2 * d_w
        normalize!(d_v)

        t_iter = (time_ns() - t_start) / 1.0e9

        @printf("  Step %2d | E: %14.10f | Time: %.4f s\n", step, energy, t_iter)
    end
end


function test_trotter(mole, nsteps)
    basis       = BasisManager(mole)
    orbs        = Orbitals(); kernel(mole, orbs, generalize=false)
    pool        = FEB(orbs)
    h_v         = get_hf(basis)
    d_v         = CuArray{Float64,1,CUDA.DeviceMemory}(h_v)
    h_funcs     = OTF_Functions(basis, eltype(pool)(), pool, time_print=false)
    d_funcs     = CuOTF_Functions(basis, h_funcs.ham, h_funcs.pool, time_print=false)

    h_v0_idxs   = findall(x -> x != 0, h_v) 
    h_v0_vals   = h_v[h_v0_idxs]
    d_v0_idxs   = CuArray{Int64,1,CUDA.DeviceMemory}(h_v0_idxs)
    d_v0_vals   = CuArray{Float64,1,CUDA.DeviceMemory}(h_v0_vals)
    amplitudes  = rand(Float64, length(pool))

    d_v_test    = CuArray{Float64,1,CUDA.DeviceMemory}(ones(eltype(d_v), length(d_v)))
    normalize!(d_v_test)

    for step in 1:nsteps
        rng = Xoshiro(step)
        amplitudes = rand(rng, Float64, length(pool))

        @time begin
            fill!(d_v, 0.0)
            d_v[d_v0_idxs] .= d_v0_vals

            for (i, t) in enumerate(amplitudes)
                d_funcs.expm_2d(i, t, d_v)
                i % 1000 == 0 && println(real(dot(d_v, d_v_test)))
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

    ARGS[3] == "1" && test_hvec(mole, 10)
    ARGS[3] == "2" && test_trotter(mole, 10)
end


