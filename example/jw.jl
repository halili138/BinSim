ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", 1)
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

function get_random_mo_phys(norb::Int; seed::Union{Int, Nothing}=nothing)
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


# if abspath(PROGRAM_FILE) == @__FILE__
#     norb = parse(Int, ARGS[1])
#     _1_mo, _2_mo = get_random_mo_phys(norb, seed=42)
    
#     # 在物理学家记号 <pq|rs> 下，8 重对称性表现为：
#     # <pq|rs> == <qp|sr> (同时交换 1,2 和 3,4)
#     # <pq|rs> == <rs|pq> (交换前后两对)
#     println("_2_mo shape: ", size(_2_mo))
#     println("Phys symmetry (<pq|rs> == <qp|sr>): ", isapprox(_2_mo[1,2,3,4], _2_mo[2,1,4,3]))
#     println("Phys symmetry (<pq|rs> == <rs|pq>): ", isapprox(_2_mo[1,2,3,4], _2_mo[3,4,1,2]))
    
#     @time if 0 <= norb < 32
#         int2ham_real_ui64_f64(norb, 1.0, _1_mo, _2_mo, 1e-12, true)
#     elseif 32 <= norb < 64
#         int2ham_real_ui128_f64(norb, 1.0, _1_mo, _2_mo, 1e-12, true)
#     elseif 64 <= norb < 128
#         int2ham_real_ui256_f64(norb, 1.0, _1_mo, _2_mo, 1e-12, true)
#     else
#         error("Maximum supported is (127o, 254q)")
#     end
# end

if abspath(PROGRAM_FILE) == @__FILE__
    for norb in 8:2:parse(Int, ARGS[1])
        _1_mo, _2_mo = get_random_mo_phys(norb, seed=42)        
        t = @elapsed if 0 <= norb < 32
            int2ham_real_ui64_f64(norb,  1.0, _1_mo, _2_mo, 1e-12, false)
        elseif 32 <= norb < 64
            int2ham_real_ui128_f64(norb, 1.0, _1_mo, _2_mo, 1e-12, false)
        elseif 64 <= norb < 128
            int2ham_real_ui256_f64(norb, 1.0, _1_mo, _2_mo, 1e-12, false)
        else
            error("Maximum supported is (127o, 254q)")
        end
        @printf("norb: %10s     t: %15.4f\n", norb, t)
    end
end
