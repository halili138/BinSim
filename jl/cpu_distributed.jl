module CPUDistributed

export FILES, load!

const FILES = ("distnetwork.jl",)

function load!(target::Module)
    isdefined(target, :GlobalMemMap) && return nothing

    for file in FILES
        Base.include(target, joinpath(@__DIR__, file))
    end
    return nothing
end

end # module CPUDistributed
