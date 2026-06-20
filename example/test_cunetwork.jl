_name   = length(ARGS) >= 1 ? ARGS[1] : "n2"
_basis  = length(ARGS) >= 2 ? ARGS[2] : "6-31g"
_ratio  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
_nchunk = length(ARGS) >= 4 ? parse(Int, ARGS[4])    : 4

ENV["OMP_NUM_THREADS"] = "1"
ENV["OMP_PROC_BIND"]   = "false"

include("../jl/cudistribute.jl")
include("../jl/cudistnetwork_common.jl")
include("../jl/cudistnetwork_serial.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = _name; mole.ratio = _ratio; mole.basis = _basis
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)

    println("="^60)
    println("=== CuDistributedFunctions SerialOOC ITE Test ===")
    @printf("Molecule: %s / %s   Chunks: %d\n", _name, _basis, _nchunk)
    @printf("Basis dim: %d  (%.3f GB)\n", basis.dim, basis.dim * 8 / (1024^3))
    println("="^60)

    funcs = CuDistributedFunctions(ModeSerial, basis, ham; num_chunks=_nchunk)

    v  = funcs.get_hf(mole.nelec)
    Hv = funcs.zeros()

    dτ = 1e-1
    println("--- Euler ITE (10 steps) ---")

    for step in 1:10
        t_start = time_ns()

        funcs.hvec(v, Hv)
        energy    = funcs.inner(v, Hv) / funcs.inner(v, v)
        @. v     -= dτ * Hv
        pre_norm2 = funcs.inner(v, v)
        funcs.normalize(v)

        t_iter = (time_ns() - t_start) / 1.0e9
        @printf("  Step %2d | E: %14.10f | Norm: %.8f | Time: %.4f s\n",
                step, energy, sqrt(pre_norm2), t_iter)
    end

    println("="^60)
end
