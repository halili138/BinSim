_nts    = length(ARGS) >= 1 ? ARGS[1] : "4"
_name   = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis  = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
_phases = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2

_seed  = 1234
_ntry  = 64

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

include("../jl/binsim.jl")

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
    funcs, basis = DistributedFunctions(mole, ham, comm; virtual_k=_vk, virtual_seed=_seed, virtual_ntry=_ntry, num_phases=_phases)

    my_rank == 0 && println("="^60)
    my_rank == 0 && println("=== Virtual-symmetry DistributedFunctions Test ===")
    my_rank == 0 && @printf("Molecule: %s / %s  virtual_k: %d  ntry: %d  phases: %d  dim: %d\n", _name, _basis, _vk, _ntry, _phases, basis.dim)
    my_rank == 0 && println("="^60)

    v = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()

    for step in 1:5
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
