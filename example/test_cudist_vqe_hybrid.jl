_name      = length(ARGS) >= 1 ? ARGS[1] : "h4"
_ratio     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
_basisname = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_nchunks   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8

ENV["OMP_NUM_THREADS"] = 1
ENV["OMP_PROC_BIND"] = "false"
delete!(ENV, "OMP_PLACES")
ENV["OMPI_MCA_btl"] = get(ENV, "OMPI_MCA_btl", "^openib")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

include("../jl/cudistnetwork.jl")

MPI.Init()

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basisname
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)
    @assert !isempty(pool) "FEB produced an empty operator pool"

    funcs = CuDistributedFunctions(ModeHybrid, basis, ham, pool, comm; num_chunks=_nchunks)

    lv = funcs.get_hf(mole.nelec)
    funcs.normalize(lv)
    rv = funcs.zeros()

    idxs = [1]
    x0 = zeros(Float64, length(idxs))

    if rank == 0
        println("="^60)
        println("=== CuDistributedFunctions HybridOOC VQE Smoke Test ===")
        @printf("Command: mpiexec -n 2 julia --project=. example/test_cudist_vqe_hybrid.jl 1 h4 1.0 sto-3g 8\n")
        @printf("MPI ranks: %d, OpenMP threads: %s\n", nranks, _nts)
        @printf("Molecule: %s / ratio %.6f / basis %s\n", _name, _ratio, _basisname)
        @printf("Basis dim: %d, local dim(rank0): %d, pool size: %d\n", basis.dim, funcs.local_dim, length(pool))
    end

    funcs.expm(idxs[1], 0.01, lv)
    funcs.hvec(lv, rv)
    grad_scalar = funcs.backgrad(idxs[1], 0.01, lv, rv)
    @assert isfinite(grad_scalar) "backgrad returned a non-finite scalar"

    lv .= funcs.get_hf(mole.nelec)
    funcs.normalize(lv)
    fill!(rv, 0.0)

    energy, grads, variance = energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, idxs, x0, lv, rv)
    @assert isfinite(energy) "energy_objective returned a non-finite energy"
    @assert all(isfinite, grads) "energy_objective returned a non-finite gradient"
    @assert isfinite(variance) "energy_objective returned a non-finite variance"

    if rank == 0
        @printf("Energy: %.12f\n", energy)
        @printf("Gradient norm: %.12e\n", norm(grads))
        @printf("Variance: %.12e\n", variance)
        println("="^60)
    end
    return nothing
end

try
    main()
finally
    MPI.Finalize()
end
