include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    pbc = Pbc()
    pbc.name   = "1d-h"
    pbc.ratio  = 1.0
    pbc.basis  = "gth-szv"
    pbc.pseudo = "gth-pade"
    pbc.mesh   = [parse(Int, ARGS[1]),1,1]
    pbc.scaled_center = [0,0,0]

    build(pbc)

    pbc.orbsym = ones(Int64, pbc.norb)
    basis      = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
    psi_space  = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)

    ham   = JW_hamiltonian(pbc)
    ham   = apply_constraint(ham, pbc.norb, pbc.nelec, (0.5, 0.5, 0.5))
    hf    = get_hf(basis, pbc.nelec, Tv=ComplexF64)
    diags = get_diags(basis, ham)

    # @time ham_sp = to_sparse_matrix(ham, pbc.norb, pbc.nelec)
    # @time λ, ϕ = eigs(ham_sp, nev=1, which=:SR)
    # println(λ)

    ret = @timed agg = AGG(basis, ham)
    println("Successifully Generate AGG in $(ret.time) seconds")
    print_info(agg)

    aop! = (src, dst) -> hvec_direct_agg!(basis, agg, src, dst)

    @time davidson(aop!, hf, diags)
end

