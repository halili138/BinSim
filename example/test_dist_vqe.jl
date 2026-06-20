_nts   = length(ARGS) >= 1 ? ARGS[1] : "4"
_name  = length(ARGS) >= 2 ? ARGS[2] : "h4"
_ratio = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
_basis = length(ARGS) >= 4 ? ARGS[4] : "sto-3g"

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

include("../jl/binsim.jl")
include("../jl/cpu_distributed.jl")
CPUDistributed.load!(@__MODULE__)

MPI.Init()

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)
    @assert !isempty(pool) "FEB produced an empty operator pool"

    funcs = DistributedFunctions(basis, ham, pool, comm; num_phases=1)

    idxs = [1]
    x0 = zeros(Float64, length(idxs))

    lv = funcs.get_hf(mole.nelec)
    funcs.normalize(lv)
    rv = funcs.zeros()

    rank == 0 && println("="^60)
    rank == 0 && println("=== DistributedFunctions VQE Smoke Test ===")
    rank == 0 && @printf("Command: mpiexec -n 2 julia --project=. example/test_dist_vqe.jl 4 h4 1.0 sto-3g\n")
    rank == 0 && @printf("MPI ranks: %d\n", nranks)
    rank == 0 && @printf("Molecule: %s / ratio %.6f / basis %s\n", _name, _ratio, _basis)
    rank == 0 && @printf("Basis dim: %d, local dim(rank0): %d, pool size: %d\n", basis.dim, funcs.local_dim, length(pool))

    funcs.expm(idxs[1], 0.01, lv)
    rank == 0 && println("expm smoke check passed")

    funcs.hvec(lv, rv)
    grad_scalar = funcs.backgrad(idxs[1], 0.01, lv, rv)
    @assert isfinite(grad_scalar) "Distributed backgrad returned a non-finite scalar"
    rank == 0 && @printf("backgrad smoke check passed: %.12e\n", grad_scalar)

    fill!(lv, 0.0)
    lv .= funcs.get_hf(mole.nelec)
    funcs.normalize(lv)
    fill!(rv, 0.0)

    energy, grads, variance = energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, idxs, x0, lv, rv)
    @assert isfinite(energy) "energy_objective returned a non-finite energy"
    @assert all(isfinite, grads) "energy_objective returned a non-finite gradient"
    @assert isfinite(variance) "energy_objective returned a non-finite variance"

    rank == 0 && @printf("Energy: %.12f\n", energy)
    rank == 0 && @printf("Gradient norm: %.12e\n", norm(grads))
    rank == 0 && @printf("Variance: %.12e\n", variance)

    if nranks == 1
        serial_funcs = OTF_Functions(basis, ham, pool, info_print=false, time_print=false)
        serial_lv = get_hf(basis, mole.nelec)
        serial_rv = zeros(eltype(serial_lv), length(serial_lv))
        serial_energy, serial_grads, serial_variance = energy_objective(
            serial_funcs.hvec,
            serial_funcs.expm,
            serial_funcs.backgrad,
            idxs,
            copy(x0),
            serial_lv,
            serial_rv,
        )
        rank == 0 && @printf("Serial one-rank comparison: ΔE=%.3e, Δ|g|=%.3e, Δvariance=%.3e\n",
            abs(energy - serial_energy), abs(norm(grads) - norm(serial_grads)), abs(variance - serial_variance))
    end

    rank == 0 && println("="^60)
    return nothing
end

try
    main()
finally
    MPI.Finalize()
end
