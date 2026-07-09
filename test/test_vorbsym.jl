include("../jl/binsim.jl")

using Statistics

# ======================================================================
# 1. 基础算子：极速计算单根弦的虚拟对称性
# ======================================================================
function get_sym(str::UInt32, v_orbsym::Vector{Int})
    sym = 0
    # 假设轨道数不超过 32 (对应 UInt32)
    norb = length(v_orbsym)
    for i in 1:norb
        if (str >> (i - 1)) & 1 == 1
            sym ⊻= v_orbsym[i]
        end
    end
    return sym
end

# ======================================================================
# 2. 极速评估引擎：基于异或卷积 (XOR Convolution) 的分布统计
# ======================================================================
function evaluate_distribution_from_strings(
    astrs::Vector{UInt32},
    bstrs::Vector{UInt32},
    v_orbsym::Vector{Int},
    k::Int
)
    num_blocks = 1 << k

    # 分别统计 alpha 弦和 beta 弦在当前虚拟标签下的分布
    count_a = zeros(Int, num_blocks)
    count_b = zeros(Int, num_blocks)

    for astr in astrs
        count_a[get_sym(astr, v_orbsym)+1] += 1
    end

    for bstr in bstrs
        count_b[get_sym(bstr, v_orbsym)+1] += 1
    end

    # 异或卷积：合并 alpha 和 beta 得到完整的行列式 (Determinant) 分布
    dist = zeros(Int, num_blocks)
    for i in 0:(num_blocks-1)
        for j in 0:(num_blocks-1)
            # 行列式的虚拟对称性 = alpha对称性 ⊻ beta对称性
            s_det = i ⊻ j
            # 组合数相乘并累加到目标桶
            dist[s_det+1] += count_a[i+1] * count_b[j+1]
        end
    end

    return dist
end

# ======================================================================
# 3. 贪心分配器：寻找最优的虚拟轨道标签
# ======================================================================
function optimize_v_orbsym_from_strings(
    astrs::Vector{UInt32},
    bstrs::Vector{UInt32},
    norb::Int,
    k::Int;
    max_iter=5000
)
    num_blocks = 1 << k

    # 初始轮询分配
    v_orbsym = [i % num_blocks for i in 0:(norb-1)]

    best_orbsym = copy(v_orbsym)
    best_dist = evaluate_distribution_from_strings(astrs, bstrs, v_orbsym, k)
    best_variance = var(best_dist)

    println("--- 基于实际弦集合的寻优开始 ---")
    @printf("初始方差: %.2e | 最大块: %d | 最小块: %d\n",
        best_variance, maximum(best_dist), minimum(best_dist))

    for iter in 1:max_iter
        # 随机突变
        idx = rand(1:norb)
        old_sym = v_orbsym[idx]
        new_sym = rand(0:(num_blocks-1))

        if old_sym == new_sym
            continue;
        end

        v_orbsym[idx] = new_sym
        current_dist = evaluate_distribution_from_strings(astrs, bstrs, v_orbsym, k)
        current_variance = var(current_dist)

        if current_variance < best_variance
            best_variance = current_variance
            copyto!(best_orbsym, v_orbsym)
            copyto!(best_dist, current_dist)
        else
            v_orbsym[idx] = old_sym # 回退
        end
    end

    println("--- 寻优结束 ---")
    @printf("最终方差: %.2e | 最大块: %d | 最小块: %d\n",
        best_variance, maximum(best_dist), minimum(best_dist))

    return best_orbsym, best_dist
end


mole = Mole()
mole.name = ARGS[1]
mole.ratio = 1.0
mole.basis = ARGS[2]

build(mole)

blocks, block_map = get_sym_blocks(mole.norb, mole.nelec, mole.orbsym, 0, UInt32)

astrs = UInt32[]
bstrs = UInt32[]
for blk in blocks 
    append!(astrs, blk.astrs)
    append!(bstrs, blk.bstrs)
end

sort!(astrs)
sort!(bstrs)
unique!(astrs)
unique!(bstrs)
println(length(astrs))
println(length(bstrs))

basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)

# f = combs -> begin
#     s = UInt32(0)
#     for i in combs
#         s |= UInt32(1) << i
#     end
#     return s
# end

# astrs = f.(combinations([i for i in 0:(norb-1)], na))
# bstrs = f.(combinations([i for i in 0:(norb-1)], nb))

# best_orbsym, best_dist = optimize_v_orbsym_from_strings(astrs, bstrs, mole.norb, k_test, max_iter=1000)

# println(best_orbsym)

# # v_orbsym_optimal, dist_optimal = optimize_virtual_orbsym(norb_test, nelec_test, k_test)

# # println("\n最优虚拟标签数组 v_orbsym (长度 $(norb_test)):")
# # println(v_orbsym_optimal)

# # println("\n在 64 个虚拟块中的态数分布概览 (抽取前 8 块):")
# # println(dist_optimal[1:8])
# const norb = 8  # 设定轨道数量

# # 1. 物理对称性 (例如 D2h 点群，取值 0-7)
# # 这代表真实的物理空间对称性
# const orbsym_phys = [0, 1, 2, 3, 0, 1, 2, 3] 

# # 2. 虚拟对称性 (例如 Z2^4，取值 0-15)
# # 这是我们人为赋予的、用来把内存均匀切成 16 块的路由标签
# const orbsym_virt = [12, 5, 9, 3, 14, 7, 2, 11]

# # ==========================================
# # 开始验证
# # ==========================================

# # 1. 假设一个初始电子组态 (弦)
# # 例如：占据第 1, 3, 4, 7 个轨道 (二进制: 01001101 -> 77)
# str = astrs[10] 
# str = bstrs[10]
# # 2. 假设一个算符 (激发操作)
# # 例如：销毁第 2 轨道，创建在第 3 轨道 (翻转第 2 和第 3 个轨道)
# # 二进制: 00000110 -> 6
# ax = 6 

# # 3. 物理系统真实发生的状态跃迁：作用算符得到目标弦
# target_str = str ⊻ ax 

# println("=== 状态与算符 ===")
# println("初始弦 (str):       ", string(str, base=2, pad=norb))
# println("算符   (ax):        ", string(ax, base=2, pad=norb))
# println("目标弦 (str ⊻ ax):  ", string(target_str, base=2, pad=norb))
# println()

# # ==========================================
# # 验证 1: 物理对称性同态映射
# # ==========================================
# phys_sym_str    = get_sym(str, orbsym_phys)
# phys_sym_ax     = get_sym(ax, orbsym_phys)
# phys_sym_target = get_sym(target_str, orbsym_phys)

# println("=== 物理对称性验证 (D2h) ===")
# println("S_phys(str):              ", phys_sym_str)
# println("S_phys(ax):               ", phys_sym_ax)
# println("S_phys(str) ⊻ S_phys(ax): ", phys_sym_str ⊻ phys_sym_ax)
# println("S_phys(target_str):       ", phys_sym_target)
# println(">> 同态映射成立?          ", phys_sym_target == (phys_sym_str ⊻ phys_sym_ax))
# println()

# # ==========================================
# # 验证 2: 虚拟对称性同态映射 (网络路由引擎的核心)
# # ==========================================
# virt_sym_str    = get_sym(str, orbsym_virt)
# virt_sym_ax     = get_sym(ax, orbsym_virt)
# virt_sym_target = get_sym(target_str, orbsym_virt)

# println("=== 虚拟对称性验证 (Z2^k) ===")
# println("S_virt(str):              ", virt_sym_str)
# println("S_virt(ax):               ", virt_sym_ax)
# println("S_virt(str) ⊻ S_virt(ax): ", virt_sym_str ⊻ virt_sym_ax)
# println("S_virt(target_str):       ", virt_sym_target)
# println(">> 同态映射成立?          ", virt_sym_target == (virt_sym_str ⊻ virt_sym_ax))

# ```

# ### 预期输出与物理映射意义
# 跑完这段代码，你会看到类似这样的输出：

# ```plaintext
# === 状态与算符 ===
# 初始弦 (str):       01001101
# 算符   (ax):        00000110
# 目标弦 (str ⊻ ax):  01001011

# === 物理对称性验证 (D2h) ===
# S_phys(str):              1
# S_phys(ax):               3
# S_phys(str) ⊻ S_phys(ax): 2
# S_phys(target_str):       2
# >> 同态映射成立?          true

# === 虚拟对称性验证 (Z2^k) ===
# S_virt(str):              14
# S_virt(ax):               12
# S_virt(str) ⊻ S_virt(ax): 2
# S_virt(target_str):       2
# >> 同态映射成立?          true

# ```
# 这段输出极其精炼地展示了你的架构为什么能够跑通：

# 1. **虚拟路由层：** 网络通信模块看到 `S_virt(ax) = 12`，于是它告诉持有了当前态（`S_virt = 14`）的物理节点：“你需要去向编号为 $14 \oplus 12 = 2$ 的那个虚拟分块请求数据。” （完美的大块路由）
# 2. **底层计算层：** 当包含了目标态的数据块通过 MPI 飞过来，并喂给你底层的 `gather_contract_pure_a_batched_impl` 时，C++ 内核执行了真实的 `str ⊻ ax`。此时，系统**根本不需要验证**它的物理对称性是不是 `2`，因为数学保证了它**一定**是 `2`。底层只需要在发过来的 Chunk 里直接查绝对地址指针即可。
