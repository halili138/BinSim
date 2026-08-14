include("binsim.jl")

try
    using CUDA
catch
    using Pkg
    Pkg.add(CUDA)
end

include("cuda/cunetwork.jl")
