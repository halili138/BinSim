ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "8")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")
include("../jl/sci_bitstr.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole, spin="aabb")
    mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

    run_sci_bitstr(mole; max_iter=20, max_size=5000, eps=1e-6, verbose=true)
end
