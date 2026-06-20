module CUDADistributed

include("cpu_distributed.jl")
include("cuda_network.jl")
using .CPUDistributed
using .CUDANetwork

export FILES, load!

const FILES = (
    "cudistnetwork_common.jl",
    "cudistnetwork_serial.jl",
    "cudistnetwork_nvlink.jl",
    "cudistnetwork_davidson.jl",
    "cudistnetwork_hybrid.jl",
)

function load!(target::Module)
    CPUDistributed.load!(target)
    CUDANetwork.load!(target)
    isdefined(target, :CuSubTopology) && return nothing

    for file in FILES
        Base.include(target, joinpath(@__DIR__, file))
    end
    return nothing
end

end # module CUDADistributed
