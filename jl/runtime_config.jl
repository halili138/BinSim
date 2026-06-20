module RuntimeConfig

using LinearAlgebra
using MPI

export RuntimeSettings, is_rank0_or_serial, print_runtime_settings, apply_blas_threads!

const MPI_RANK_ENV_KEYS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
)

struct RuntimeSettings
    slurm_cpus::String
    omp_threads::String
    omp_proc_bind::String
    omp_places::String
end

RuntimeSettings() = RuntimeSettings(
    get(ENV, "SLURM_CPUS_PER_TASK", "Not Set"),
    get(ENV, "OMP_NUM_THREADS", "Not Set"),
    get(ENV, "OMP_PROC_BIND", "Not Set"),
    get(ENV, "OMP_PLACES", "Not Set"),
)

function preinit_mpi_rank()
    for key in MPI_RANK_ENV_KEYS
        if haskey(ENV, key)
            return parse(Int, ENV[key])
        end
    end
    return 0
end

function is_rank0_or_serial()
    return MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) == 0 : preinit_mpi_rank() == 0
end

function apply_blas_threads!(settings::RuntimeSettings)
    if settings.omp_threads != "Not Set"
        BLAS.set_num_threads(parse(Int, settings.omp_threads))
    end
    return BLAS.get_num_threads()
end

function print_runtime_settings(settings::RuntimeSettings)
    is_rank0_or_serial() || return nothing

    println("Sys.CPU_THREADS       $(Sys.CPU_THREADS)")
    println("SLURM_CPUS_PER_TASK   $(settings.slurm_cpus)")
    println("OMP_NUM_THREADS       $(settings.omp_threads)")
    println("OMP_PROC_BIND         $(settings.omp_proc_bind)")
    println("OMP_PLACES            $(settings.omp_places)")
    println("BLAS_NUM_THREADS      $(BLAS.get_num_threads())")
    println("Threads.nthreads()    $(Threads.nthreads())")
    println("")
    return nothing
end

end # module RuntimeConfig
