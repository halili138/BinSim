using LinearAlgebra
using Base.Order
using Base.Threads
using BitIntegers
using Printf
using JLD2
using Dates
using Combinatorics
using SparseArrays

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

BLAS.set_num_threads(1)
slurm_cpus      = get(ENV, "SLURM_CPUS_PER_TASK", "Not Set")
omp_threads     = get(ENV, "OMP_NUM_THREADS", "Not Set")
omp_proc_bind   = get(ENV, "OMP_PROC_BIND", "Not Set")
omp_places      = get(ENV, "OMP_PLACES", "Not Set")

println("Sys.CPU_THREADS       $(Sys.CPU_THREADS)")
println("SLURM_CPUS_PER_TASK   $(slurm_cpus)")
println("OMP_NUM_THREADS       $(omp_threads)")
println("OMP_PROC_BIND         $(omp_proc_bind)")
println("OMP_PLACES            $(omp_places)")
println("BLAS_NUM_THREADS      $(BLAS.get_num_threads())")
println("Threads.nthreads()    $(Threads.nthreads())")
println("")

const jld2path::String = joinpath(@__DIR__, "../jld2file/")
const eps1::Float64 = 1e-8
const eps2::Float64 = 1e-12
const eps3::Float64 = 1e-16

include("integer.jl")
include("tools.jl")
include("load_data.jl")
include("binqubitabab.jl")
include("binqubitaabb.jl")
include("hamiltonian.jl")
include("network.jl")
include("davidson.jl")
include("ansatz.jl")
include("vqe.jl")
include("method.jl")

