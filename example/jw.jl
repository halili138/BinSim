ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 1)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

function get_random_mo_phys(norb::Int; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    # 1. 单体积分
    h1_random = randn(Float64, norb, norb)
    _1_mo = (h1_random + h1_random') / 2.0
    _1_mo -= Diagonal(abs.(randn(norb)) .* 5.0)

    # 2. 双体积分 (半正定矩阵 V_IJ)
    n_pairs = div(norb * (norb + 1), 2)
    rank = max(1, div(n_pairs, 2))
    L = randn(Float64, n_pairs, rank)
    V_IJ = L * L'

    # 生成 2D 索引映射矩阵 (norb x norb)
    p = 1:norb
    q = (1:norb)'
    idx_map = @. max(p, q) * (max(p, q) - 1) ÷ 2 + min(p, q)

    # === 直接生成 <pq|rs> 的魔法 ===
    # reshape 不会复制任何数据，只是改变我们“看”这个数组的维度

    # 让第一个复合索引占据第 1 维 (p) 和第 3 维 (r)
    idx_pr = reshape(idx_map, norb, 1, norb, 1)

    # 让第二个复合索引占据第 2 维 (q) 和第 4 维 (s)
    idx_qs = reshape(idx_map, 1, norb, 1, norb)

    # 直接生成物理学家记号下的 4D 张量！
    _2_mo_phys = getindex.(Ref(V_IJ), idx_pr, idx_qs)

    return _1_mo, _2_mo_phys
end

if abspath(PROGRAM_FILE) == @__FILE__
    for norb in 8:2:parse(Int, ARGS[1])
        _1_mo, _2_mo = get_random_mo_phys(norb, seed=42)
        _JW_hamiltonian(norb, 1.0, _1_mo, _2_mo, tol=1e-12, spin="aabb", verbose=true)
    end
end
