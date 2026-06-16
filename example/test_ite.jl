include("../jl/binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    @time run_euler_ite(basis, ham, get_hf(basis, mole.nelec), mole.e_scale,
        dt=0.1, max_step=5, tol=1e-10,
        # save_path=joinpath(@__DIR__, "callback/ite_$(ARGS[1])_$(ARGS[2]).jld2"),
    )

    @time run_rk4_ite(basis, ham, v0, mole.e_scale,
        dt=0.1, max_step=10000, tol=1e-10)

    @time run_krylov_ite(basis, ham, v0, mole.e_scale,
        dt=1.0, krylov_dim=20, tol=1e-8)
end

