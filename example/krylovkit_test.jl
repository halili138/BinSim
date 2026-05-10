include("../binsim.jl")
using KrylovKit

# if abspath(PROGRAM_FILE) == @__FILE__
#     nkx = parse(Int, ARGS[1])
    
#     pbc = Pbc()
#     pbc.name   = "1d-h"
#     pbc.ratio  = 1.0
#     pbc.basis  = "gth-szv"
#     pbc.pseudo = "gth-pade"
#     pbc.mesh   = [nkx,1,1]
#     pbc.scaled_center = [0,0,0]

#     build(pbc)

#     pbc.orbsym = ones(Int64, pbc.norb)
#     fci_basis  = BasisManager(pbc.norb, pbc.nelec, pbc.orbsym)
#     psi_space  = fci_basis.dim * 8 / (1 << 30)
#     @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", fci_basis.dim, psi_space)

#     ham = JW_hamiltonian(pbc)
#     ham = apply_constraint(ham, pbc.norb, pbc.nelec, (0.5, 0.5, 0.5))

#     ret = @timed agg = AGG(fci_basis, ham)
#     println("Successifully Generate AGG in $(ret.time) seconds")
#     print_info(agg)

#     aop = (src::Vector{ComplexF64}) -> begin
#         dst = similar(src) 
#         hvec_direct_agg!(fci_basis, agg, src, dst) 
#         return dst 
#     end

#     v0   = randn(ComplexF64, fci_basis.dim)
#     v0 ./= norm(v0)

#     println("Running KrylovKit Arnoldi Solver...")

#     @time vals, vecs, info = eigsolve(
#         aop,                # 传入修改后的单参数函数
#         v0,                 # 纯随机正态分布初始向量
#         3,                  # 找 1 个特征值
#         :SR,                # 找实部最小的 (Smallest Real)
#         tol = 1e-5,         # 容差
#         krylovdim = 30,     # Krylov 子空间最大维度 
#         verbosity = 3       # 打印详细迭代日志
#     )

#     println("Energys: ", real.(vals))
#     println("Convergence info: ", info)
# end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    

    fci_basis  = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham = JW_hamiltonian(mole)
    ret = @timed net = OTF(fci_basis, ham)

    aop = (src::Vector{Float64}) -> begin
        dst = similar(src) 
        hvec_otf!(fci_basis, net, src, dst) 
        return dst 
    end

    v0   = randn(Float64, fci_basis.dim)
    v0 ./= norm(v0)

    println("Running KrylovKit Arnoldi Solver...")

    @time vals, vecs, info = eigsolve(
        aop,                # 传入修改后的单参数函数
        v0,                 # 纯随机正态分布初始向量
        1,                  # 找 k 个特征值
        :SR,                # 找实部最小的 (Smallest Real)
        tol = 1e-5,         # 容差
        krylovdim = 20,     # Krylov 子空间最大维度 
        verbosity = 3       # 打印详细迭代日志
    )

    println("Energys: ", real.(vals))
    println("Convergence info: ", info)
end
