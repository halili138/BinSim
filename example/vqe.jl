include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name  = ARGS[1]
#     mole.ratio = 1.0
#     mole.basis = ARGS[2]
#     build(mole)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     psi_space = basis.dim * 8 / (1 << 30)
#     @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, basis.dim*8/(1<<30))
#     H0b = JW_hamiltonian(mole, based=0, spin="aabb")
#     ret = @timed agg = AGG(basis, H0b, mole.orbsym)
#     println("Successifully Generate AGG in $(ret.time) seconds")
#     print_info(agg)

#     hf = get_hf(basis, mole.nelec, mole.orbsym)
#     diags = get_diags(basis, agg)

#     aop! = (src::Array{Float64,1}, dst::Array{Float64,1}) -> begin
#         cpu_start = CPUtime_us()
#         t2 = @timed hvec_direct_agg!(basis, agg, src, dst)
#         cpu_time = (CPUtime_us() - cpu_start) / 1e6
#         @printf("hvec-time process: %.4f  wall: %.4f seconds ", cpu_time, t2.time)
#     end

#     @time davidson(aop!, hf, diags)
# end

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    run_adapt_vqe(mole,
                  adapt_options=ADAPT_OPTIONS(
                    Gtol=1e-3,
                    gtol=1e-4,
                    htol=1e-3,
                    Δtol=1e-8,
                    ),
                  vqe_options=VQE_OPTIONS(
                    ftol=1e-8,
                    gtol=1e-6,
                    maxiter=1000,
                    verbose=1,
                    ),
                  )
end


    # amplitudes, selec_idxs = load_idxs("save_temp.jld2")

# if abspath(PROGRAM_FILE) == @__FILE__
#     @ccall LIB_AGG.init_likwid()::Cvoid

#     mole = Mole()
#     mole.name  = ARGS[1]
#     mole.ratio = 1.0
#     mole.basis = ARGS[2]
#     build(mole)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     H0b = JW_hamiltonian(mole, based=0, spin="aabb")
#     agg = AGG(basis, H0b, mole.orbsym)
    
#     hf = get_hf(basis, mole.nelec, mole.orbsym)
#     diags = get_diags(basis, agg)

#     println("Warming up JIT and Hardware...")
#     dummy_src = rand(Float64, basis.dim)
#     dummy_dst = zeros(Float64, basis.dim)
#     hvec_direct_agg_benchmark!(basis, agg, dummy_src, dummy_dst, false) 
#     println("Warm-up completed.")

#     aop! = (src::Array{Float64,1}, dst::Array{Float64,1}) -> hvec_direct_agg_benchmark!(basis, agg, src, dst, true) 

#     @time davidson(aop!, hf, diags)

#     @ccall LIB_AGG.close_likwid()::Cvoid
# end

