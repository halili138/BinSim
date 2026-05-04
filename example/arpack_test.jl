include("../binsim.jl")
using Arpack
using LinearMaps

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
    fci_basis  = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
    psi_space  = fci_basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", fci_basis.dim, psi_space)

    ham   = JW_hamiltonian(pbc)
    ham   = apply_constraint(ham, pbc.norb, pbc.nelec, (0.5, 0.5, 0.5))

    ret = @timed agg = AGG(fci_basis, ham)
    println("Successifully Generate AGG in $(ret.time) seconds")
    print_info(agg)

    agg_map = LinearMap{ComplexF64}(
        (dst, src) -> hvec_direct_agg!(fci_basis, agg, src, dst), 
        fci_basis.dim, 
        ismutating=true, 
        ishermitian=true
    )

    println("Running Arpack directly on C++ AGG Network...")
    @time λ_aggs, ϕ_aggs = eigs(agg_map, nev=3, which=:SR)
    println("Arpack on AGG energies: ", real.(λ_aggs))
end

