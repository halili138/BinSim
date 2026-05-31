include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)
    # blocks, block_map = get_sym_blocks(mole.norb, mole.nelec, mole.orbsym, 0, UInt32)
    # for b in blocks
    #     println(b.num_a * b.num_b)
    # end
    # basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    # ham = JW_hamiltonian(mole)

    # mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

    # n2 = [4, 8, 42, 47, 137, 142, 203, 208, 229, 403, 405, 409, 410, 415, 417, 421, 422, 429, 435, 439, 440, 443, 575, 577, 587, 589, 598, 604, 610, 747, 749, 753, 754, 759, 761, 765, 766, 773, 779, 783, 784, 787, 919, 921, 931, 933, 942, 948, 954, 1041, 1042, 1047, 1053, 1057, 1058, 1061, 1067, 1069, 1073, 1074, 1157, 1163, 1167, 1168, 1171, 1177, 1178, 1181, 1183, 1187, 1188, 1319, 1321, 1325, 1326, 1331, 1333, 1337, 1338, 1345, 1351, 1355, 1356, 1359, 1794, 1799, 1889, 1894, 1955, 1960, 1981]
    # n2_x0 = [8.831080078994632e-5, 0.008810347078085016, 0.0009204443878783982, 0.0009204443878784063, -0.012487362094145542, -0.012487362094145481, -0.048809981890802434, 0.03259130852664034, 0.03259130852664034, 0.00030956769417988167, 0.0006564265237750743, 0.0006629488614781899, -0.0003889895092836023, 0.0003095676941785102, 0.0006564265237751061, 0.0006629488614781884, -0.0003889895092836096, 0.0009884065150085296, 0.0004215702839174989, -0.0002574955264001958, -0.00025749552640020327, -0.00038015358255699303, 0.00033542098241593957, 0.0009375329868929999, 0.00033542098241458817, 0.0009375329868930277, 8.831080078994453e-5, 0.001026090970821618, 0.0007383133416652411, 0.0006564265237750765, 0.01093724771405313, 0.01843320616502922, -0.008443049975733644, 0.0006564265237751102, 0.010937247714053328, 0.018433206165029296, -0.008443049975733695, 0.0004215702839175004, 0.024326222296275703, 0.030920568259176693, 0.030920568259176374, 0.015798598881130825, 0.0009375329868930106, 0.05141854689957618, 0.0009375329868930316, 0.05141854689957634, 0.00881034707808524, 0.0007383133416652486, 0.013347963050812692, 0.016325116497625483, 0.03676262703357771, -0.000257495526400199, 0.030920568259176832, 0.13866035245559533, 0.08557260892439275, 0.031920105246493705, 0.0006629488614781962, 0.018433206165029848, 0.012130117081908398, -0.000671203280148387, -0.00025749552640019947, 0.03092056825917695, 0.08557260892439428, 0.13866035245559305, 0.031920105246494135, 0.03676262703357553, 0.01632511649762566, 0.0006629488614781912, 0.01843320616502969, 0.01213011708190846, -0.0006712032801483887, -0.0003889895092836039, -0.008443049975733814, -0.000671203280148415, 0.022321249019510287, -0.0003889895092836109, -0.008443049975733818, -0.0006712032801484286, 0.022321249019511737, -0.0003801535825569948, 0.015798598881130967, 0.03192010524649464, 0.031920105246494024, 0.02831119544703066, 0.0009204443878783908, 0.0009204443878783933, -0.012487362094145532, -0.0124873620941455, -0.04880998189080341, 0.03259130852664142, 0.03259130852664162]

    # orbs = Orbitals()
    # kernel(mole, orbs, generalize=true)
    # pool = FEB(orbs)

    # # x0 = zeros(Float64, length(pool))
    # # x0[n2] .= n2_x0

    # opt_e, opt_x = run_exact_vqe_adaptive(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     # x0=x0,
    #     options=VQE_OPTIONS(
    #         ftol=1e-10,
    #         gtol=1e-8,
    #         maxiter=100000,
    #         verbose=2,
    #         # save_path=joinpath(@__DIR__, "callback/exact_vqe_$(ARGS[1])_$(ARGS[2])_$(ARGS[3]).jld2")
    #     ),
    # )
    # println(opt_x)

    # opt_x, sele_idxs = run_adapt_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     adapt_options=ADAPT_OPTIONS(
    #         Gtol=1e-6,
    #         gtol=1e-8,
    #         htol=1e-6,
    #         Δtol=1e-14,
    #         verbose=1,
    #     ),
    #     vqe_options=VQE_OPTIONS(
    #         ftol=1e-14,
    #         gtol=1e-10,
    #         maxiter=10000,
    #         verbose=1,
    #     )
    # )
    # println(opt_x)
    # println(sele_idxs)


    # e_opt, lv, x_opt = run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), mole.e_scale,
    #     options=VQE_OPTIONS(
    #         ftol=1e-14,
    #         gtol=1e-10,
    #         maxiter=100000,
    #         verbose=2,
    #     )
    # )
    # println(x_opt)
end


# if abspath(PROGRAM_FILE) == @__FILE__
#     mole = Mole()
#     mole.name = ARGS[1]
#     mole.ratio = parse(Float64, ARGS[2])
#     mole.basis = ARGS[3]

#     build(mole)

#     basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
#     ham = JW_hamiltonian(mole)

#     mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=false)
#     pool1 = FEB(orbs)

#     orbs = Orbitals()
#     kernel(mole, orbs, generalize=true)
#     pool2 = FEB(orbs)

#     idxs = []
#     for (i, op) in enumerate(pool1)
#         idx = findfirst(x -> x==op, pool2)
#         push!(idxs, idx)
#         if isnothing(idx)
#             println(i)
#         end
#     end
#     println(idxs)
# end
