include("../binsim.jl")
using MPI, LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
myrank = MPI.Comm_rank(comm)

# ══════════════════════════════════════════════════════════════════════
# 1. OMP threads: total = 12, distributed across MPI ranks
# ══════════════════════════════════════════════════════════════════════
omp_threads = 12 ÷ nprocs
ENV["OMP_NUM_THREADS"] = string(omp_threads)
BLAS.set_num_threads(1)

# ══════════════════════════════════════════════════════════════════════
# 2. Load molecule (all ranks independently)
# ══════════════════════════════════════════════════════════════════════
mole = Mole()
mole.name = ARGS[1]
mole.ratio = parse(Float64, ARGS[2])
mole.basis = ARGS[3]
build(mole)

# ══════════════════════════════════════════════════════════════════════
# 3. Build BasisManager + Hamiltonian (all ranks)
# ══════════════════════════════════════════════════════════════════════
basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham = JW_hamiltonian(mole)

# ══════════════════════════════════════════════════════════════════════
# 4. Reference FCI (rank 0 only, then broadcast)
# ══════════════════════════════════════════════════════════════════════
if myrank == 0
    println("\n=== FCI (CPU) reference energy ===")
    v0 = get_hf(basis, mole.nelec)
    e_fci, _ = run_fci(basis, ham, v0)
    println("  FCI energy: $(e_fci)")
end
e_fci = MPI.bcast((myrank == 0 ? Ref(e_fci)[] : Ref(0.0))[], 0, comm)

# ══════════════════════════════════════════════════════════════════════
# 5. Build OTF + Distributed objects + local diags
# ══════════════════════════════════════════════════════════════════════
myrank == 0 && println("\n=== Distributed FCI ===")
myrank == 0 && @printf("  nprocs=%d  omp_threads=%d  local_dim=%d\n",
    nprocs, omp_threads, MPI.bcast(basis.dim, 0, comm) ÷ nprocs)

otf = OTF(basis, ham)
dbasis = DistributedBasisManager(comm, basis)
dnet = DistributedOTF(otf, basis.orbsym)

# compute global diags (all ranks independently), extract local portion
global_diags = get_diags(basis, otf, Float64)
local_diags = zeros(Float64, dbasis.local_dim)
extract_local_vec!(dbasis, global_diags, local_diags)

# ══════════════════════════════════════════════════════════════════════
# 6. Distributed hvec wrapper with Allreduce + per-step timing
# ══════════════════════════════════════════════════════════════════════
_step = Ref(0)
function dist_hvec!(src, dst)
    _step[] += 1
    BLAS.set_num_threads(1)
    t0 = time()
    hvec_otf_distributed!(dbasis, dnet, src, dst)
    dt = time() - t0
    if myrank == 0
        @printf("    [hvec %04d] %.6f s\n", _step[], dt)
    end
end

# ══════════════════════════════════════════════════════════════════════
# 7. Extract local initial state from global HF
# ══════════════════════════════════════════════════════════════════════
v0_global = get_hf(basis, mole.nelec)
v0_local = zeros(Float64, dbasis.local_dim)
extract_local_vec!(dbasis, v0_global, v0_local)

# ══════════════════════════════════════════════════════════════════════
# 8. Davidson (each rank independently, but hvec gives global result)
# ══════════════════════════════════════════════════════════════════════
myrank == 0 && println("\n  Running Davidson on Distributed OTF ...")
BLAS.set_num_threads(1)
t_total = @elapsed e_dist, v_local = davidson(dist_hvec!, v0_local, local_diags, tol=1e-5, comm=comm)

# ══════════════════════════════════════════════════════════════════════
# 9. Output
# ══════════════════════════════════════════════════════════════════════
if myrank == 0
    @printf("\n  Energy : %.14f\n", e_dist)
    @printf("  Error  : %.3e\n", abs(e_dist - e_fci))
    @printf("  Wall time: %.4f s\n", t_total)
    @printf("  nprocs=%d, omp=%d\n\n", nprocs, omp_threads)
end

# ══════════════════════════════════════════════════════════════════════
# 10. Cleanup C++ objects before MPI.Finalize (avoid PMI_free_mem error)
# ══════════════════════════════════════════════════════════════════════
if dbasis.ptr != C_NULL
    @ccall LIB_OTF_DIST.destroy_distributed_basis_f64(dbasis.ptr::Ptr{Cvoid})::Cvoid
    dbasis.ptr = C_NULL
end
if dnet.ptr != C_NULL
    @ccall LIB_OTF_DIST.destroy_distributed_net_f64(dnet.ptr::Ptr{Cvoid})::Cvoid
    dnet.ptr = C_NULL
end

MPI.Finalize()
