using LinearAlgebra
using Base.Order
using Base.Threads
using BitIntegers
using Printf
using JLD2
using Dates
using Combinatorics
using SparseArrays

using MPI
using Random
using Optim
using NLSolversBase
using LineSearches
using CPUTime
using FFTW
using DataFrames
using Arpack
using LinearMaps
using DifferentialEquations
using RecursiveArrayTools
using PyCall

include("runtime_config.jl")
using .RuntimeConfig: RuntimeSettings, apply_blas_threads!, is_rank0_or_serial,
    print_runtime_settings

include("native_libraries.jl")
using .NativeLibraries: jld2path, pypath, libpath, LIB_BASIS, LIB_HAM, LIB_OTF,
    LIB_DIAG, LIB_DIST, LIB_CUDIST, LIB_CUOTF

include("tolerances.jl")
using .Tolerances: eps1, eps2, eps3

const runtime_settings = RuntimeSettings()
apply_blas_threads!(runtime_settings)
print_runtime_settings(runtime_settings)

include("integer.jl")
include("tools.jl")
include("geo.jl")
include("save_int.jl")
include("load_data.jl")
include("binqubitabab.jl")
include("binqubitaabb.jl")
include("hamiltonian.jl")
include("symm.jl")
include("network.jl")
include("davidson.jl")
include("ansatz.jl")
include("vqe.jl")
include("method.jl")
include("vqite.jl")
include("vqrte.jl")
