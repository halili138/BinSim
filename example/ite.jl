# include("../binsim.jl")

# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name = ARGS[1]
#     mole.ratio = 1.0
#     mole.basis = ARGS[2]

#     build(mole)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham = JW_hamiltonian(mole)
#     v0 = get_hf(basis, mole.nelec)

#     # @time run_euler_ite(basis, ham, v0, mole.e_scale,
#     #     dτ=1e8, max_step=10000, tol=1e-10, 
#     #     save_path=joinpath(@__DIR__, "callback/ite_$(ARGS[1])_$(ARGS[2]).jld2"),
#     # )
#     mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

#     @time run_euler_ite(basis, ham, v0, mole.e_scale,
#         dτ=1e2, max_step=10000, tol=1e-10,
#         # save_path=joinpath(@__DIR__, "callback/ite_$(ARGS[1])_$(ARGS[2]).jld2"),
#     )


#     # @time run_rk4_ite(basis, ham, v0, mole.e_scale,
#     #     dτ=0.5, max_step=10000, tol=1e-10, net="agg")

#     # @time run_krylov_ite(basis, ham, v0, mole.e_scale,
#     #     dτ=5.0, krylov_dim=20, tol=1e-8, net="otf")
# end

mole = Mole()
mole.name = "n2"
mole.ratio = 1.0
mole.basis = "sto-3g"

build(mole)

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham = JW_hamiltonian(mole)
v0 = get_hf(basis, mole.nelec)

mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

@time run_euler_ite(basis, ham, v0, mole.e_scale,
    dτ=1e-2, max_step=10000, tol=1e-10,
)
