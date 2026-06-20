_name   = length(ARGS) >= 1 ? ARGS[1] : "h12"
_basis  = length(ARGS) >= 2 ? ARGS[2] : "sto-3g"
_ratio  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
_vk     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
_phases = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1

_seed   = 1234
_ntry   = 64

ENV["OMP_NUM_THREADS"] = 1
ENV["OMP_PROC_BIND"] = "false"
delete!(ENV, "OMP_PLACES")
ENV["OMPI_MCA_btl"] = get(ENV, "OMPI_MCA_btl", "^openib")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

include("../jl/binsim.jl")
include("../jl/cuda_distributed.jl")
CUDADistributed.load!(@__MODULE__)

MPI.Init()

if abspath(PROGRAM_FILE) == @__FILE__
    comm = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)

    ham = JW_hamiltonian(mole)
    funcs, basis = CuDistributedFunctions(
        ModeNVLink, mole, ham, comm;
        virtual_k=_vk,
        virtual_seed=_seed,
        virtual_ntry=_ntry,
        num_phases=_phases,
    )

    if my_rank == 0
        println("="^60)
        println("=== Virtual-symmetry CuDistributedFunctions NVLink Smoke Test ===")
        @printf(
            "Molecule: %s / %s  ratio: %.6f  virtual_k: %d  ntry: %d  phases: %d  dim: %d\n",
            _name, _basis, _ratio, _vk, _ntry, _phases, basis.dim,
        )
        println("="^60)
    end

    v = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()

    for step in 1:10
        t_start = time_ns()

        funcs.hvec(v, Hv)
        energy    = funcs.inner(v, Hv)
        @. v     -= 1e-2 * Hv
        funcs.normalize(v)

        t_iter = (time_ns() - t_start) / 1.0e9

        if my_rank == 0
            @printf("  Step %2d | E: %14.10f | Time: %.4f s\n", step, energy, t_iter)
        end
    end

    MPI.Finalize()
end
