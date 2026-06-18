_nts   = length(ARGS) >= 1 ? ARGS[1] : "4"
_name  = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk    = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
_seed  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1234

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

using MPI
MPI.Init()

include("../jl/binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    comm = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)

    ham = JW_hamiltonian(mole)
    funcs, basis = DistributedFunctions(mole, ham, comm; virtual_k=_vk, virtual_seed=_seed)

    my_rank == 0 && println("="^60)
    my_rank == 0 && println("=== Virtual-symmetry DistributedFunctions Test ===")
    my_rank == 0 && @printf("Molecule: %s / %s  virtual_k: %d  dim: %d\n", _name, _basis, _vk, basis.dim)
    my_rank == 0 && println("="^60)

    v = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()

    funcs.hvec(v, Hv)
    energy = funcs.inner(v, Hv)
    norm_v = sqrt(funcs.inner(v, v))

    if my_rank == 0
        @printf("  <HF|H|HF>: %.14f\n", energy)
        @printf("  ||HF||:     %.14f\n", norm_v)
    end

    MPI.Finalize()
end
