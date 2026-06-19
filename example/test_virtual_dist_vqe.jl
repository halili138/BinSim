_nts    = length(ARGS) >= 1 ? ARGS[1] : "1"
_name   = length(ARGS) >= 2 ? ARGS[2] : "h4"
_basis  = length(ARGS) >= 3 ? ARGS[3] : "sto-3g"
_ratio  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
_vk     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
_seed   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1234
_ntry   = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 16
_phases = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 4
_nops   = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 2

ENV["OMP_NUM_THREADS"] = _nts
delete!(ENV, "OMP_PROC_BIND")
delete!(ENV, "OMP_PLACES")

using MPI
MPI.Init()

include("../jl/binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    mole = Mole()
    mole.name = _name
    mole.ratio = _ratio
    mole.basis = _basis
    build(mole)
    mole.orbsym = Int64.(mole.orbsym .% 10)

    ham = JW_hamiltonian(mole)

    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)
    pool = FEB(orbs)

    funcs, basis = DistributedFunctions(
        mole, ham, pool, comm;
        virtual_k=_vk,
        virtual_seed=_seed,
        virtual_ntry=_ntry,
        num_phases=_phases,
    )

    v0 = funcs.get_hf(mole.nelec)
    lv = copy(v0)
    rv = funcs.zeros()

    idxs = collect(1:min(_nops, length(pool)))
    x = fill(0.01, length(idxs))
    energy, grads, δ²H = energy_objective(funcs.hvec, funcs.expm, funcs.backgrad, idxs, x, lv, rv)
    gnorm = norm(grads)

    if rank == 0
        @printf("energy: %.14f\n", energy)
        @printf("gradient norm: %.6e\n", gnorm)
        @printf("δ²H: %.6e\n", δ²H)
        @printf("phase count: %d\n", size(funcs.ham_sub_topos, 2))
        @printf("local/global dimensions: %d / %d\n", funcs.local_dim, basis.dim)
    end

    if nranks == 1
        otf_funcs = OTF_Functions(basis, ham, pool; info_print=false)
        otf_lv = get_hf(basis, mole.nelec)
        otf_rv = zeros(eltype(otf_lv), length(otf_lv))
        otf_energy, otf_grads, otf_δ²H = energy_objective(
            otf_funcs.hvec, otf_funcs.expm, otf_funcs.backgrad, idxs, copy(x), otf_lv, otf_rv)
        diff = abs(energy - otf_energy)
        grad_diff = norm(grads - otf_grads)
        var_diff = abs(δ²H - otf_δ²H)

        if rank == 0
            @printf("|distributed - OTF| energy: %.6e\n", diff)
            @printf("|distributed - OTF| gradient norm: %.6e\n", grad_diff)
            @printf("|distributed - OTF| δ²H: %.6e\n", var_diff)
        end
        @assert diff < 1e-8
        @assert grad_diff < 1e-8
        @assert var_diff < 1e-8
    end

    MPI.Finalize()
end
