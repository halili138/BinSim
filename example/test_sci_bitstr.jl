ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "8")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")
include("../jl/sci_bitstr.jl")
include("data/fcis.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]
    build(mole)

    # basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    # ham   = JW_hamiltonian(mole, spin="aabb")
    # mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))
    mole.e_scale = n2_6_31g[1.0]
    mole.orbsym .%= 10

    select_mode = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :external_block
    run_sci_bitstr(mole; max_iter=20, max_size=5000, eps=1e-5, verbose=true,
                   debug_compare_fci=false, select_mode=select_mode,
                   debug_external_select=(select_mode != :full))

    if get(ENV, "BINSIM_SCI_BITSTR_COMPARE_SELECT_MODES", "0") == "1"
        for mode in (:full, :external_block, :external_links)
            @printf("\n[SCI bitstr] comparing select_mode=%s\n", String(mode))
            run_sci_bitstr(mole; max_iter=20, max_size=5000, eps=1e-5, verbose=true,
                           debug_compare_fci=false, select_mode=mode,
                           debug_external_select=(mode != :full))
        end
    end
end
