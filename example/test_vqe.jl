ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 8)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl") 

function reduce_uccsd_op(mole;
    Ti::DataType=UInt32, Tv::DataType=Float64, complete::Bool=false,
)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)          # 标准激发：占据 → 虚拟

    # 过滤：激发算符的轨道索引列表长度 == 2 为单激发，== 4 为双激发
    g = a -> length(a) in (2, 4)
    orbs.spin_based = orbs.spin_based[g.(orbs.spin_based)]

    return FEB(orbs, Ti=Ti, Tv=Tv, complete=complete)
end

function run_uccsd(mole::Mole)
    basis  = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)

    hf_vec = get_hf(basis, mole.nelec)     # Hartree-Fock 占据向量

    # ---------- 计算 FCI 参考能量 ----------
    println("Computing FCI reference energy ...")
    e_fci, v_fci = run_fci(basis, ham, hf_vec)
    println("FCI energy = ", e_fci, "\n")


    # ---------- UCCSD 池 VQE ----------
    println("=== Running VQE with UCCSD pool (only singles and doubles) ===")
    pool_uccsd = reduce_uccsd_op(mole)
    println("UCCSD pool size: ", length(pool_uccsd))

    println("run exact UCCSD... ...")
    run_exact_vqe(basis, ham, pool_uccsd, hf_vec, e_fci,
            options = VQE_OPTIONS(verbose=1))
    println("")


    println("run Trotterized UCCSD... ...")
    run_vqe(basis, ham, pool_uccsd, hf_vec, e_fci,
            options = VQE_OPTIONS(verbose=1))
    println("")

    println("All done!")

end


function scan_n2()
    mole = Mole()
    mole.name  = "n2"
    mole.basis = "sto-3g"
        
    for ilen in 0:21
        mole.ratio = (0.8+0.2*ilen)/1.1
        println("bond length: ", mole.ratio*1.1)
        mole.orbsym = Int64.(mole.orbsym .% 10)
    
        run_uccsd(mole)
    end
end


scan_n2()
