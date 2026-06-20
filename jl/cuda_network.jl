module CUDANetwork

export FILES, load!

const FILES = ("cunetwork_impl.jl",)

function load!(target::Module)
    isdefined(target, :CuBasisManager) && return nothing

    Base.eval(target, :(using CUDA))
    for file in FILES
        Base.include(target, joinpath(@__DIR__, file))
    end
    return nothing
end

end # module CUDANetwork
