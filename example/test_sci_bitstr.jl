ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "8")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")
include("data/fcis.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]
    build(mole)

    mole.e_scale = n2_6_31g[parse(Float64, ARGS[2])]
    mole.orbsym .%= 10

    run_sci_bitstr(mole; max_iter=20, eps=parse(Float64, ARGS[4]), verbose=true)
end
