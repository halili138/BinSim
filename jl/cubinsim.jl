include("binsim.jl")

try
    using CUDA
catch
    using Pkg
    Pkg.add(CUDA)
end

include("cuda/cunetwork.jl")
include("cuda/cusci_bitstr.jl")
include("cuda/cudistnetwork_common.jl")
include("cuda/cudistnetwork_serial.jl")
include("cuda/cudistnetwork_nvlink.jl")
include("cuda/cudistnetwork_davidson.jl")
include("cuda/cudistnetwork_hybrid.jl")
