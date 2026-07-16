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
using MPI

# 辅助函数：多进程场景下，仅 rank 0 打印诊断信息
function _preinit_mpi_rank()
    for key in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "SLURM_PROCID",
        "MV2_COMM_WORLD_RANK",
    )
        if haskey(ENV, key)
            return parse(Int, ENV[key])
        end
    end
    return 0
end

function is_rank0_or_serial()
    return MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) == 0 : _preinit_mpi_rank() == 0
end

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

slurm_cpus      = get(ENV, "SLURM_CPUS_PER_TASK", "Not Set")
omp_threads     = get(ENV, "OMP_NUM_THREADS",     "Not Set")
omp_proc_bind   = get(ENV, "OMP_PROC_BIND",       "Not Set")
omp_places      = get(ENV, "OMP_PLACES",          "Not Set")

BLAS.set_num_threads(parse(Int, omp_threads))

if is_rank0_or_serial()
    println("Sys.CPU_THREADS       $(Sys.CPU_THREADS)"       )
    println("SLURM_CPUS_PER_TASK   $(slurm_cpus)"            )
    println("OMP_NUM_THREADS       $(omp_threads)"           )
    println("OMP_PROC_BIND         $(omp_proc_bind)"         )
    println("OMP_PLACES            $(omp_places)"            )
    println("BLAS_NUM_THREADS      $(BLAS.get_num_threads())")
    println("Threads.nthreads()    $(Threads.nthreads())"    )
    println("")
end

const jld2path   = joinpath(@__DIR__, "../jld2file/")
const pypath     = joinpath(@__DIR__, "../py/")
const libpath    = joinpath(@__DIR__, "../src/lib/")

const LIB_BASIS         = joinpath(libpath, "libbasis.so"  )
const LIB_HAM           = joinpath(libpath, "libham.so"    )
const LIB_HAM_REAL      = joinpath(libpath, "libham_real.so")
const LIB_OTF           = joinpath(libpath, "libotf.so"    )
const LIB_DIAG          = joinpath(libpath, "libdiag.so"   )
const LIB_DIST          = joinpath(libpath, "libdist.so"   )
const LIB_CUDIST        = joinpath(libpath, "libcudist.so" )
const LIB_CUOTF         = joinpath(libpath, "libcuotf.so"  )
const LIB_CUDA_SCI      = joinpath(libpath, "libcuda_sci_bitstr.so") 
const LIB_SCI_BITSTR    = joinpath(libpath, "libsci_otf.so")

const eps1::Float64 = 1e-8
const eps2::Float64 = 1e-12
const eps3::Float64 = 1e-16

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
include("dist.jl")
include("sci.jl")
