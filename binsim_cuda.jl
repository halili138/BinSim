using CUDA
include("binsim.jl")
const LIB_CUSVDNETWORK = joinpath(@__DIR__, "src/lib/libcusvdnetwork.so")
const HostBinaryQubitAABB{Ti,Tv,Tg} = BinaryQubitAABB{Ti,Tv,Tg,Array{Ti,1},Array{Tv,1},Array{Tg,1}}

struct CuBinOp{Ti,Tv,Tg} 
    azs::CuArray{Ti,1,CUDA.DeviceMemory}
    bzs::CuArray{Ti,1,CUDA.DeviceMemory}
    cs::CuArray{Tv,1,CUDA.DeviceMemory}
    gs::Array{Tg,1}
end

function to_gpu(h_A::HostBinaryQubitAABB{Ti,Tv,Tg}) where {Ti,Tv,Tg}
    CuBinOp(
        CuArray{Ti,1,CUDA.DeviceMemory}(h_A.azs),
        CuArray{Ti,1,CUDA.DeviceMemory}(h_A.bzs),
        CuArray{Tv,1,CUDA.DeviceMemory}(h_A.cs),
        h_A.gs,
    )
end

mutable struct CuRoutingNetworkSVD
    ptr::Ptr{Cvoid}
    dim::Int64

    function CuRoutingNetworkSVD(
        basis::BasisManager, 
        net::RoutingNetworkSVD; 
        mode::Int=0,
    )        
        ptr = @ccall LIB_CUSVDNETWORK.create_device_svd_network(
            net.ptr::Ptr{Cvoid},
            basis.ptr::Ptr{Cvoid},
            mode::Cint,
        )::Ptr{Cvoid}
        
        ptr == C_NULL && error("Failed to create C++ CuSVDNetwork.")
        
        obj = new(ptr, basis.dim)
        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_CUSVDNETWORK.destroy_device_svd_network(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end
        
        return obj
    end
end

function hvec_svd!(
    basis::BasisManager,
    cusvdnet::CuRoutingNetworkSVD,
    A0b::CuBinOp{UInt32,Float64,Int64},
    src::CuArray{Float64,1,CUDA.DeviceMemory},
    dst::CuArray{Float64,1,CUDA.DeviceMemory},
)
    ngs = length(A0b.gs) - 1

    @ccall LIB_CUSVDNETWORK.hvec_svd_network_cuda(
        basis.ptr::Ptr{Cvoid},
        cusvdnet.ptr::Ptr{Cvoid},
        A0b.azs::CuPtr{UInt32},
        A0b.bzs::CuPtr{UInt32},
        A0b.cs::CuPtr{Cdouble},
        A0b.gs::Ptr{Int64},
        ngs::Int64,
        src::CuPtr{Cdouble},
        dst::CuPtr{Cdouble},
    )::Cvoid
end

# function tvec_svd!(
#     cusvdnet::CuRoutingNetworkSVD,
#     idx::Int64,
#     θ::Float64,
#     vec::CuArray{Float64,1,CUDA.DeviceMemory},
# )
#     @ccall LIB_CUSVDNETWORK.tvec_svd_network_cuda(
#         cusvdnet.ptr::Ptr{Cvoid},
#         (idx-1)::Int64,
#         θ::Cdouble,
#         vec::CuPtr{Cdouble},
#     )::Cvoid
# end

# function expect_svd(
#     cusvdnet::CuRoutingNetworkSVD,
#     idx::Int64,
#     θ::Float64,
#     lv::CuArray{Float64,1,CUDA.DeviceMemory},
#     rv::CuArray{Float64,1,CUDA.DeviceMemory},
# )
#     return @ccall LIB_CUSVDNETWORK.tvec_svd_network_cuda(
#         cusvdnet.ptr::Ptr{Cvoid},
#         (idx-1)::Int64,
#         θ::Cdouble,
#         lv::CuPtr{Cdouble},
#         rv::CuPtr{Cdouble},
#     )::Cdouble
# end

# function energy_objective(
#     basis::BasisManager,
#     cu_ham_net::CuRoutingNetworkSVD,
#     cu_pool_net::CuRoutingNetworkSVD, 
#     idxs::Vector{Int64},
#     x::Vector{Float64},          
#     H0b::CuBinOp, 
#     lv::CuVector{Float64},
#     rv::CuVector{Float64},
# )
#     nparas = length(x)

#     for i in 1:nparas
#         tvec_svd!(cu_pool_net, idxs[i], x[i], lv)
#     end

#     hvec_svd!(basis, cu_ham_net, H0b, lv, rv)

#     lnorm  = norm(lv) ^ 2
#     rnorm  = norm(rv) ^ 2
#     energy = real(dot(lv, rv)) / lnorm
#     δ²H    = max(0.0, rnorm / lnorm - energy ^ 2)
#     grad   = Vector{Float64}(undef, nparas)

#     for i in nparas:-1:1
#         tvec_svd!(cu_pool_net, idxs[i], -x[i], lv)
#         grad[i] = real(expect_svd(cu_pool_net, idxs[i], x[i], lv, rv)) * 2 / lnorm
#         tvec_svd!(cu_pool_net, idxs[i], -x[i], rv)
#     end

#     return energy, grad, δ²H
# end

# function davidson_cuda(
#     hvec::Function,
#     v0::Vector{Float64},
#     diags::CuArray{Float64,1,CUDA.DeviceMemory};
#     maxiter::Int=100000,
#     tol::Float64=1e-5,
#     ncv::Int=3, 
#     maxspace::Int=ncv+20,
#     verbose::Bool=true,
#     save_path::String="",
# )
#     @assert maxspace > ncv

#     N = length(v0)
    
#     d_r  = CUDA.zeros(Float64, N)
#     d_vt = CUDA.zeros(Float64, N)
    
#     xt     = Vector{Float64}(undef, N)
#     axt    = Vector{Float64}(undef, N)
#     r      = Vector{Float64}(undef, N)
#     vt     = Vector{Float64}(undef, N)
#     v_best = Vector{Float64}(undef, N)
#     heff_col_buf = Vector{Float64}(undef, maxspace)

#     V  = [Vector{Float64}(undef, N) for _ in 1:maxspace]
#     AV = [Vector{Float64}(undef, N) for _ in 1:maxspace]
    
#     V_ptrs  = [pointer(v) for v in V]
#     AV_ptrs = [pointer(av) for av in AV]

#     V_tmp  = [Vector{Float64}(undef, N) for _ in 1:ncv]
#     AV_tmp = [Vector{Float64}(undef, N) for _ in 1:ncv]

#     heff = zeros(Float64, maxspace, maxspace)

#     copyto!(V[1], v0)
#     @ccall LIB_DAVIDSON.inplace_normalize(N::Int64, V_ptrs[1]::Ptr{Cdouble})::Cdouble
    
#     copyto!(d_vt, V[1])
#     d_av0 = hvec(d_vt)       
#     copyto!(AV[1], d_av0)
#     CUDA.unsafe_free!(d_av0)
    
#     e_best = @ccall LIB_DAVIDSON.fast_dot(
#         N::Int64, V_ptrs[1]::Ptr{Cdouble}, AV_ptrs[1]::Ptr{Cdouble}
#     )::Cdouble

#     heff[1,1] = e_best
    
#     dim::Int32 = 1 
    
#     GC.@preserve V AV V_ptrs AV_ptrs xt axt r vt heff_col_buf begin
#         for iter in 1:maxiter
#             es, vs  = eigen(heff[1:dim, 1:dim])
#             min_idx = argmin(real.(es))
#             min_e   = real(es[min_idx])
#             coeffs  = vs[:, min_idx]

#             norm_r = @ccall LIB_DAVIDSON.build_ritz_and_residual(
#                 N::Int64, dim::Cint, coeffs::Ptr{Cdouble}, min_e::Cdouble,
#                 V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}},
#                 xt::Ptr{Cdouble}, axt::Ptr{Cdouble}, r::Ptr{Cdouble}
#             )::Cdouble

#             verbose && @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
#             e_best = min_e

#             if !isempty(save_path)
#                 copyto!(v_best, xt)
#                 jldopen(save_path, "w") do file
#                     file["e_best"] = e_best
#                     file["v_best"] = v_best
#                 end
#                 if norm_r < tol
#                     break
#                 end
#             else
#                 if norm_r < tol
#                     copyto!(v_best, xt)
#                     break
#                 end
#             end

#             if dim >= maxspace
#                 nsave = min(ncv, dim)
#                 p = sortperm(real.(es))
                
#                 for i in 1:nsave
#                     idx = p[i]
#                     fill!(V_tmp[i],  0.0)
#                     fill!(AV_tmp[i], 0.0)
#                     for j in 1:dim
#                         c = vs[j, idx]
#                         @. V_tmp[i]  += c * V[j]
#                         @. AV_tmp[i] += c * AV[j]
#                     end
#                 end
                
#                 for i in 1:nsave
#                     copyto!(V[i],  V_tmp[i])
#                     copyto!(AV[i], AV_tmp[i])
#                 end
                
#                 dim = nsave
                
#                 @ccall LIB_DAVIDSON.rebuild_heff(
#                     N::Int64, dim::Cint, 
#                     V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}}, 
#                     heff::Ptr{Cdouble}, maxspace::Cint
#                 )::Cvoid
#             end

#             copyto!(d_r, r) 
            
#             shift = max(norm_r * 1e-2, 1e-4)
#             @. d_vt = d_r / ifelse(abs(diags - min_e) < shift, 
#                                    copysign(shift, diags - min_e), 
#                                    diags - min_e)

#             copyto!(vt, d_vt)
#             norm_before = norm(vt) 
            
#             norm_after = @ccall LIB_DAVIDSON.mgs_orthogonalize(
#                 N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, vt::Ptr{Cdouble}
#             )::Cdouble

#             kappa = 0.717
#             if norm_after < kappa * norm_before
#                 norm_after = @ccall LIB_DAVIDSON.mgs_orthogonalize(
#                     N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, vt::Ptr{Cdouble}
#                 )::Cdouble
#             end

#             if norm_after < 1e-12
#                 println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
#                 break
#             end
            
#             @ccall LIB_DAVIDSON.inplace_normalize(N::Int64, vt::Ptr{Cdouble})::Cdouble

#             dim += 1
#             copyto!(V[dim], vt)
            
#             copyto!(d_vt, V[dim])
#             d_avt = hvec(d_vt)
#             copyto!(AV[dim], d_avt)
#             CUDA.unsafe_free!(d_avt)
            
#             @ccall LIB_DAVIDSON.compute_heff_col(
#                 N::Int64, dim::Cint, 
#                 V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs[dim]::Ptr{Cdouble}, 
#                 heff_col_buf::Ptr{Cdouble}
#             )::Cvoid
            
#             for j in 1:dim
#                 hij = heff_col_buf[j]
#                 heff[j, dim] = hij
#                 heff[dim, j] = hij
#             end
#         end
#     end

#     CUDA.unsafe_free!(d_r)
#     CUDA.unsafe_free!(d_vt)

#     println("\n")

#     return e_best, v_best
# end

