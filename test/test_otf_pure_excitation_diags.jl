ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

basis = BasisManager(2, (1, 0), zeros(Int64, 2))
pure_excitation = QubitOperatorAABB([(0, "X")], 1.0, UInt32, Float64)
otf = OTF(basis, pure_excitation)

diags = get_diags(basis, otf, Float64)

@assert length(diags) == basis.dim
@assert all(iszero, diags)
