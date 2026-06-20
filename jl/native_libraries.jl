module NativeLibraries

export jld2path, pypath, libpath,
    LIB_BASIS, LIB_HAM, LIB_OTF, LIB_DIAG, LIB_DIST, LIB_CUDIST, LIB_CUOTF

const jld2path = joinpath(@__DIR__, "../jld2file/")
const pypath = joinpath(@__DIR__, "../py/")
const libpath = joinpath(@__DIR__, "../src/lib/")

const LIB_BASIS = joinpath(libpath, "libbasis.so")
const LIB_HAM = joinpath(libpath, "libham.so")
const LIB_OTF = joinpath(libpath, "libotf.so")
const LIB_DIAG = joinpath(libpath, "libdiag.so")
const LIB_DIST = joinpath(libpath, "libdist.so")
const LIB_CUDIST = joinpath(libpath, "libcudist.so")
const LIB_CUOTF = joinpath(libpath, "libcuotf.so")

end # module NativeLibraries
