_nts     = length(ARGS) >= 1 ? ARGS[1] : "1"
_name    = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis   = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk      = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2
_seed    = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1234
_ntry    = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 64
_nchunk  = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 4
_gpu_id  = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 0

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

include("../jl/cudistnetwork.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)

    ham = JW_hamiltonian(mole)
    funcs, basis = CuDistributedFunctions(
        ModeSerial, mole, ham;
        virtual_k=_vk,
        virtual_seed=_seed,
        virtual_ntry=_ntry,
        num_chunks=_nchunk,
        gpu_id=_gpu_id,
    )

    num_blocks = _num_wavefunction_symmetry_blocks(basis)
    effective_chunks = min(_nchunk, max(1, num_blocks))

    println("="^60)
    println("=== Virtual-symmetry CuDistributedFunctions Serial Smoke Test ===")
    @printf("OpenMP threads: %s  GPU: %d\n", _nts, _gpu_id)
    @printf(
        "Molecule: %s / %s  ratio: %.6f  norb: %d  nelec: (%d, %d)\n",
        _name, _basis, _ratio, mole.norb, mole.nelec[1], mole.nelec[2],
    )
    @printf("Basis dim: %d  (%.3f GB)\n", basis.dim, basis.dim * 8 / (1024^3))
    @printf(
        "Virtual settings: virtual_k: %d  virtual_seed: %d  virtual_ntry: %d\n",
        _vk, _seed, _ntry,
    )
    @printf(
        "Chunks: requested: %d  effective: %d  symmetry blocks: %d\n",
        _nchunk, effective_chunks, num_blocks,
    )
    println("="^60)

    v = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()
    funcs.hvec(v, Hv)
    energy = funcs.inner(v, Hv)

    @printf("<HF|H|HF>: %14.10f\n", energy)
end
