_mode   = lowercase(ARGS[1])
_name   = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis  = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
_count  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : (_mode == "nvlink" ? 2 : 4)

_seed   = 1234
_ntry   = 64
_gpu_id = 0

if !(_mode in ("serial", "nvlink", "hybrid"))
    error("mode must be one of: serial, nvlink, hybrid")
end

ENV["OMP_NUM_THREADS"] = 1
ENV["OMP_PROC_BIND"] = "false"
delete!(ENV, "OMP_PLACES")
ENV["OMPI_MCA_btl"] = get(ENV, "OMPI_MCA_btl", "^openib")
ENV["JULIA_CUDA_MEMORY_POOL"] = get(ENV, "JULIA_CUDA_MEMORY_POOL", "none")

include("../jl/cudistnetwork.jl")

if _mode != "serial"
    MPI.Init()
end

if abspath(PROGRAM_FILE) == @__FILE__
    comm = _mode == "serial" ? nothing : MPI.COMM_WORLD
    my_rank = _mode == "serial" ? 0 : MPI.Comm_rank(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)
    mole.orbsym .%= 10 
    ham = JW_hamiltonian(mole)

    if _mode == "serial"
        funcs, basis = CuDistributedFunctions(
            ModeSerial, mole, ham;
            virtual_k=_vk,
            virtual_seed=_seed,
            virtual_ntry=_ntry,
            num_chunks=_count,
            gpu_id=_gpu_id,
        )
    elseif _mode == "nvlink"
        funcs, basis = CuDistributedFunctions(
            ModeNVLink, mole, ham, comm;
            virtual_k=_vk,
            virtual_seed=_seed,
            virtual_ntry=_ntry,
            num_phases=_count,
        )
    else
        funcs, basis = CuDistributedFunctions(
            ModeHybrid, mole, ham, comm;
            virtual_k=_vk,
            virtual_seed=_seed,
            virtual_ntry=_ntry,
            num_chunks=_count,
        )
    end

    v = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()

    for step in 1:10
        t_start = time_ns()

        funcs.hvec(v, Hv)
        energy    = funcs.inner(v, Hv)
        @. v     -= 1e-2 * Hv
        funcs.normalize(v)

        t_iter = (time_ns() - t_start) / 1.0e9

        if my_rank == 0
            @printf("  Step %2d | E: %14.10f | Time: %.4f s\n", step, energy, t_iter)
        end
    end

    if _mode != "serial"
        MPI.Finalize()
    end
end
