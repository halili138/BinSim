include("../jl/binsim.jl")


if abspath(PROGRAM_FILE) == @__FILE__

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

    # run_vqrte_tdva(
    #     basis, ham, pool, v0,
    #     max_step = 100,
    #     per_print = 10
    # )

    run_vqrte_pvqd(
        basis, ham, pool, v0,
        max_step = 100,
        per_print = 10
    )


    # run_adapt_vqrte(
    #     basis, ham, pool, v0,
    #     max_step    = 200,
    #     max_ansatz  = Nmax,
    #     adapt_tol   = 1e-3,
    #     per_print   = 10
    # )
end

