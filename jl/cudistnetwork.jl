include("binsim.jl")
include("cuda_distributed.jl")

CUDADistributed.load!(@__MODULE__)
