include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)

    # orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    # pool  = FEB(orbs)

    # svd_groups = compress_by_svd(ham)
    # ranks  = Dict{Int, Int}()
    # num_azs = Dict{Int, Int}()
    # num_bzs = Dict{Int, Int}()
    # for g in svd_groups
    #     r = g.rank
    #     naz = length(g.azs)
    #     nbz = length(g.bzs)
    #     !haskey(ranks, r) ? (ranks[r] = 1) : (ranks[r] += 1)
    #     !haskey(num_azs, naz) ? (num_azs[naz] = 1) : (num_azs[naz] += 1)
    #     !haskey(num_bzs, nbz) ? (num_bzs[nbz] = 1) : (num_bzs[nbz] += 1)
    # end

    # println(ranks)
    # println(num_azs)
    # println(num_bzs)
end



# function test_mole(name, ratio, basis)
#     mole = Mole()
#     mole.name = name
#     mole.ratio = ratio
#     mole.basis = basis

#     build(mole)

#     num_astrs = Dict{Int, Int}()
#     num_bstrs = Dict{Int, Int}()
#     blocks, block_map = get_sym_blocks(mole.norb, mole.nelec, mole.orbsym, 0, UInt32)
    
#     for b in blocks 
#         na = b.num_a
#         nb = b.num_b
#         !haskey(num_astrs, na) ? (num_astrs[na] = 1) : (num_astrs[na] += 1)
#         !haskey(num_bstrs, nb) ? (num_bstrs[nb] = 1) : (num_bstrs[nb] += 1)
#     end

#     println(num_astrs)
#     println(num_bstrs)
#     println("\n")
# end

# if abspath(PROGRAM_FILE) == @__FILE__
#     for (name, basis) in [("h14", "sto-3g"), ("c2h4", "sto-3g"), ("c2h6", "sto-3g"), ("c3h6", "sto-3g"), ("c3h8", "sto-3g")]
#         test_mole(name, 1.0, basis)
#     end
#     for (name, basis) in [("n2", "6-31g"), ("co", "6-31g"), ("hcn", "6-31g"), ("h2co", "6-31g")]
#         test_mole(name, 1.0, basis)
#     end
#     for (name, basis) in [("h2o", "cc-pvdz"), ("nh3", "cc-pvdz"), ("n2", "cc-pvdz")]
#         test_mole(name, 1.0, basis)
#     end
# end
