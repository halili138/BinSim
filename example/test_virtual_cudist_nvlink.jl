_nts    = length(ARGS) >= 1 ? ARGS[1] : "1"
_name   = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis  = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2
_seed   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1234
_ntry   = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 64
_phases = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 2

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

using MPI
MPI.Init()

include("../jl/cudistnetwork.jl")

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
    funcs.hvec(v, Hv)

    energy = funcs.inner(v, Hv)
    norm = sqrt(funcs.inner(v, v))

    if my_rank == 0
        @printf("<HF|H|HF>: %14.10f\n", energy)
        @printf("Norm: %.8f\n", norm)
    end

    MPI.Finalize()
end
