include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    nq = parse(Int64, ARGS[1]) 
    J = 1.0 
    h = 0.5

    norb  = nq ÷ 2
    N = 1 << norb - 1
    astrs = [UInt32(i) for i in 0:N]
    bstrs = [UInt32(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))

    v0 = ones(Float64, basis.dim)
    normalize!(v0)

    ham = ising_module(nq, J, h, is_pbc=true)

    e_scale, _ = run_fci(basis, ham, v0, net="otf")

    # @time run_euler_ite(basis, ham, v0, e_scale,
    #     max_step=10000, tol=1e-10, net="otf")

    @time run_krylov_ite(basis, ham, v0, e_scale,
        max_step=10000, tol=1e-10, net="otf")
end

