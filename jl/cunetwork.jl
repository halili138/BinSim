include("binsim.jl")
include("cuda_network.jl")

CUDANetwork.load!(@__MODULE__)
