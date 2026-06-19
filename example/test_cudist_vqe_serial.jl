# CuDistributedFunctions SerialOOC VQE smoke test.
#
# Example:
#   julia --project=. example/test_cudist_vqe_serial.jl 4 h4 1.0 sto-3g 4

_nts       = length(ARGS) >= 1 ? ARGS[1] : "4"
_name      = length(ARGS) >= 2 ? ARGS[2] : "h4"
_ratio     = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
_basisname = length(ARGS) >= 4 ? ARGS[4] : "sto-3g"
_nchunks   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 4

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

using LinearAlgebra
using Printf

include("../jl/cudistnetwork.jl")

function main()
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

    funcs = CuDistributedFunctions(ModeSerial, basis, ham, pool; num_chunks=_nchunks)

    lv = funcs.get_hf(mole.nelec)
    funcs.normalize(lv)
    rv = funcs.zeros()

    idxs = [1]
    x0 = zeros(Float64, length(idxs))

    println("="^60)
    println("=== CuDistributedFunctions SerialOOC VQE Smoke Test ===")
    @printf("Command: julia --project=. example/test_cudist_vqe_serial.jl 4 h4 1.0 sto-3g 4\n")
    @printf("OpenMP threads: %s\n", _nts)
    @printf("Molecule: %s / ratio %.6f / basis %s\n", _name, _ratio, _basisname)
    @printf("Basis dim: %d, local dim: %d, pool size: %d\n", basis.dim, funcs.local_dim, length(pool))

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

    @printf("Energy: %.12f\n", energy)
    @printf("Gradient norm: %.12e\n", norm(grads))
    @printf("Variance: %.12e\n", variance)
    println("="^60)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
