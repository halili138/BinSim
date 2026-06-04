include("../binsim.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    # nq    = parse(Int64, ARGS[1])
    # J     = 1.0
    # h     = 0.5
    # Tv    = ComplexF64
    # norb  = nq ÷ 2
    # astrs = [UInt32(i) for i in 0:(1 << norb - 1)]
    # bstrs = [UInt32(i) for i in 0:(1 << norb - 1)]
    # basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))
    # ham   = ising_module(nq, J, h, Tv=Tv, is_pbc=true)
    # e_fci, v_fci = run_fci(basis, ham, normalize!(ones(Tv, basis.dim)))
    # pool  = ising_pool(nq, Tv=Tv, is_pbc=true, pool_type="local")

    # ham_quench = ising_module(nq, J, 0.6, Tv=Tv, is_pbc=true)
    # run_fci(basis, ham_quench, normalize!(ones(Tv, basis.dim)))

    # run_vqrte_forward(
    #     basis, ham_quench, pool, v_fci, e_fci,
    #     max_step = 10,
    #     per_print = 1
    # )


    Tv = ComplexF64
    nq = parse(Int64, ARGS[1])
    Nmax = parse(Int64, ARGS[2])
    norb = nq ÷ 2
    N = 1 << norb - 1
    astrs = [UInt32(i) for i in 0:N]
    bstrs = [UInt32(i) for i in 0:N]
    basis = BasisManager(norb, astrs, bstrs, zeros(Int64, norb))

    v0 = get_reference_state(basis, UInt32[0], UInt32[UInt32(1) << norb - 1], Tv[1])
    normalize!(v0)

    ham  = heisenberg_module(nq, 1.0, 0.8, 0.6, Tv=Tv, is_pbc=false)
    pool = heisenberg_pool(nq, Tv=Tv, is_pbc=false, pool_type="nlocal")

    # run_vqrte_forward(
    #     basis, ham, pool, v0, 0.0,
    #     max_step = 10,
    #     per_print = 1
    # )

    run_adapt_vqrte_tfim_forward(
        basis, ham, pool, v0,
        max_step    = 200,
        max_ansatz  = Nmax,
        adapt_tol   = 1e-3,
        per_print   = 10
    )
end

