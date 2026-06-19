using Printf
using MPI

if length(ARGS) < 1
    error("Usage: julia example/test_virtual_cudist.jl <serial|nvlink|hybrid> [omp_threads=1] [molecule=h12] [basis=sto-3g] [ratio=1.0] [virtual_k=2] [virtual_seed=1234] [virtual_ntry=64] [chunk_or_phase_count=4] [gpu_id=0 for serial]")
end

_mode = lowercase(ARGS[1])
_nts = length(ARGS) >= 2 ? ARGS[2] : "1"
_name = length(ARGS) >= 3 ? ARGS[3] : "h12"
_basis = length(ARGS) >= 4 ? ARGS[4] : "sto-3g"
_ratio = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
_vk = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2
_seed = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1234
_ntry = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 64
_count = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : (_mode == "nvlink" ? 2 : 4)
_gpu_id = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 0

if !(_mode in ("serial", "nvlink", "hybrid"))
    error("mode must be one of: serial, nvlink, hybrid")
end

ENV["OMP_NUM_THREADS"] = _nts
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
    funcs.hvec(v, Hv)
    energy = funcs.inner(v, Hv)
    norm_v = sqrt(funcs.inner(v, v))

    if my_rank == 0
        count_label = _mode == "nvlink" ? "phases" : "chunks"
        println("="^60)
        println("=== Virtual-symmetry CuDistributedFunctions Smoke Test ===")
        @printf("Mode: %s\n", _mode)
        @printf("OpenMP threads: %s%s\n", _nts, _mode == "serial" ? @sprintf("  GPU: %d", _gpu_id) : "")
        @printf("Molecule: %s / %s  ratio: %.6f\n", _name, _basis, _ratio)
        @printf("Virtual settings: virtual_k: %d  virtual_seed: %d  virtual_ntry: %d\n", _vk, _seed, _ntry)
        @printf("%s: %d\n", count_label, _count)
        @printf("Dimension: %d\n", basis.dim)
        @printf("<HF|H|HF>: %14.10f\n", energy)
        @printf("Norm: %.8f\n", norm_v)
        println("="^60)
    end

    if _mode != "serial"
        MPI.Finalize()
    end
end
