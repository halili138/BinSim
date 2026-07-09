include("binsim.jl")

try
    using CUDA
catch
    using Pkg
    Pkg.add(CUDA)
end

include("cuda/cunetwork.jl")
include("cuda/cusci.jl")
include("cuda/cudist_davidson.jl")
include("cuda/cudist_common.jl")
include("cuda/cudist_serial.jl")
include("cuda/cudist_nvlink.jl")
include("cuda/cudist_hybrid.jl")
