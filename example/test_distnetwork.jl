_nts   = length(ARGS) >= 1 ? ARGS[1] : "4"
_name  = length(ARGS) >= 2 ? ARGS[2] : "h12"
_basis = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

using MPI
MPI.Init()

include("../jl/distnetwork.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    # MPI.Init() already called above

    comm  = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(comm)
    n_procs = MPI.Comm_size(comm)

    # ============================================================
    # 构造分子体系（所有 rank 执行，利用 jld2file 缓存）
    # ============================================================
    mole = Mole()
    mole.name  = _name; mole.ratio = _ratio; mole.basis = _basis
    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)

    my_rank == 0 && println("="^60)
    my_rank == 0 && println("=== DistributedFunctions ITE Test ===")
    my_rank == 0 && @printf("Molecule: %s / %s\n", _name, _basis)
    my_rank == 0 && @printf("Basis dim: %d\n", basis.dim)
    my_rank == 0 && println("="^60)

    # ============================================================
    # 构建分布式函数集
    # ============================================================
    funcs = DistributedFunctions(basis, ham, comm)

    # ============================================================
    # 初态 + 虚时演化
    # ============================================================
    v  = funcs.get_hf(mole.nelec)
    funcs.normalize(v)
    Hv = funcs.zeros()

    dτ = 1e-1

    my_rank == 0 && println("--- Euler ITE ---")

    for step in 1:10
        t_start = time_ns()

        funcs.hvec(v, Hv)
        energy    = funcs.inner(v, Hv)
        @. v     -= dτ * Hv
        pre_norm2 = funcs.inner(v, v)
        funcs.normalize(v)

        t_iter = (time_ns() - t_start) / 1.0e9

        if my_rank == 0
            @printf("  Step %2d | E: %14.10f | Norm: %.8f | Time: %.4f s\n",
                    step, energy, sqrt(pre_norm2), t_iter)
        end
    end

    my_rank == 0 && println("="^60)

    MPI.Finalize()
end
