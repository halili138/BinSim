ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

function reduce_gsd_T20_feb(mole; 
    Ti::DataType=UInt32, Tv::DataType=Float64, complete::Bool=false,
)
    # 原理是, 统计 occ 和 vir 轨道的数量, 最大占据数 = ne, 那么将轨道下标整除 ne, 结果 == 0 就代表是 occ
    # 如果 occ 的数量 == 0 就代表全是占据轨道, == 4 就代表全是空轨道, 这些就是待去除的轨道组合

    orbs = Orbitals()
    kernel(mole, orbs, generalize=true)

    max_occ_idx = sum(mole.nelec)
    
    g = a -> begin
        if length(a) == 2
            return true
        else
            idxs = @. Int(a ÷ max_occ_idx)
            num_zero = sum(idxs .== 0)
            return !(num_zero == 0 || num_zero == 4)
        end
    end

    orbs.spin_based = orbs.spin_based[g.(orbs.spin_based)]

    return FEB(orbs, Ti=Ti, Tv=Tv, complete=complete)
end

mole = Mole()
mole.name  = ARGS[1]
mole.ratio = 1.0
mole.basis = ARGS[2]

build(mole)

mole.orbsym = Int64.(mole.orbsym .% 10) # 非阿贝尔点群(Dooh, Cooh), 需通过此修正为最近的阿贝尔点群(D2h, C2h)

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
ham   = JW_hamiltonian(mole)

e_fci, v_fci = run_fci(basis, ham, get_hf(basis, mole.nelec))

orbs = Orbitals()
kernel(mole, orbs, generalize=true)
pool = FEB(orbs)

run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, options = VQE_OPTIONS(verbose=1))

pool = reduce_gsd_T20_feb(mole)

run_vqe(basis, ham, pool, get_hf(basis, mole.nelec), e_fci, options = VQE_OPTIONS(verbose=1))
