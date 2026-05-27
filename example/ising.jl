include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    nq = parse(Int64, ARGS[1])
    J = 1.0
    h = 0.5

    norb = nq ÷ 2
    N = 1 << norb - 1
    astrs = [UInt32(i) for i in 0:N]
    bstrs = [UInt32(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))

    v0 = ones(Float64, basis.dim)
    normalize!(v0)

    ham = ising_module(nq, J, h, is_pbc=true)

    e_scale, v_fci = run_fci(basis, ham, v0)

    println(length(findall(x->abs(x)>1e-12, v_fci)))
    # @time run_euler_ite(basis, ham, v0, e_scale,
    #     max_step=10000, tol=1e-10, net="otf")
    # dτ = parse(Float64, ARGS[2])
    # @time run_rk4_ite(basis, ham, v0, e_scale,
    #     dτ=dτ, max_step=10000, tol=1e-10, net="otf")

    # ham = BinaryQubitAABB(ham.axs, ham.bxs, ham.azs, ham.bzs, ComplexF64.(ham.cs))
    # v0 = ones(ComplexF64, basis.dim)
    # normalize!(v0)

    # @time run_qpe(basis, ham, v0, dt=0.01)

    # @time run_krylov_ite(basis, ham, v0, e_scale,
    #     dτ=1.0, max_step=10000, tol=1e-10, net="otf")
end

