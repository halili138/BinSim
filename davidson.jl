const LIB_DIAG = joinpath(@__DIR__, "src/lib/libdiag.so")

function davidson(
    aop!::Function,
    v0::Vector{Float64},
    diags::Vector{Float64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,          
    maxspace::Int=ncv+20,
    verbose::Bool=true,
    save_path::String="",
)
    @assert maxspace > ncv

    N = length(v0)
    
    axt = Vector{Float64}(undef, N) 
    heff_col_buf = Vector{Float64}(undef, maxspace)

    V  = Vector{Vector{Float64}}()
    AV = Vector{Vector{Float64}}()
    
    V_ptrs  = Vector{Ptr{Cdouble}}(undef, maxspace)
    AV_ptrs = Vector{Ptr{Cdouble}}(undef, maxspace)

    V_tmp  = Vector{Vector{Float64}}()
    AV_tmp = Vector{Vector{Float64}}()

    heff = zeros(Float64, maxspace, maxspace)

    push!(V, v0)
    push!(AV, Vector{Float64}(undef, N))
    
    V_ptrs[1]  = pointer(V[1])
    AV_ptrs[1] = pointer(AV[1])
    
    @ccall LIB_DIAG.inplace_normalize(N::Int64, V_ptrs[1]::Ptr{Cdouble})::Cdouble
    
    aop!(V[1], AV[1])
    
    e_best = @ccall LIB_DIAG.fast_dot(N::Int64, V_ptrs[1]::Ptr{Cdouble}, AV_ptrs[1]::Ptr{Cdouble})::Cdouble
    heff[1,1] = e_best
    
    dim::Int32 = 1 
    
    GC.@preserve V AV V_tmp AV_tmp V_ptrs AV_ptrs axt diags begin
        for iter in 1:maxiter
   
            es, vs  = eigen(heff[1:dim, 1:dim])
            min_idx = argmin(real.(es))
            min_e   = real(es[min_idx])
            coeffs  = vs[:, min_idx]

            norm_r = @ccall LIB_DIAG.build_ritz_and_residual(
                N::Int64, dim::Cint, coeffs::Ptr{Cdouble}, min_e::Cdouble,
                V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}},
                axt::Ptr{Cdouble}
            )::Cdouble

            verbose && @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
            e_best = min_e

            if norm_r < tol
                xt = Vector{Float64}(undef, N)
                @ccall LIB_DIAG.build_ritz_vector(
                    N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
                    V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
                )::Cvoid
                
                if !isempty(save_path)
                    JLD2.jldopen(save_path, "w") do file
                        file["e_best"] = e_best
                        file["v_best"] = xt 
                    end
                end
                println("\nConvergence reached!")
                return e_best, xt
            end

            if dim >= maxspace
                nsave = min(ncv, dim)
                p = sortperm(real.(es))
                
                while length(V_tmp) < nsave
                    push!(V_tmp,  Vector{Float64}(undef, N))
                    push!(AV_tmp, Vector{Float64}(undef, N))
                end
                
                for i in 1:nsave
                    idx = p[i]
                    fill!(V_tmp[i],  0.0)
                    fill!(AV_tmp[i], 0.0)
                    for j in 1:dim
                        c = vs[j, idx]
                        @. V_tmp[i]  += c * V[j]
                        @. AV_tmp[i] += c * AV[j]
                    end
                end
                
                for i in 1:nsave
                    copyto!(V[i],  V_tmp[i])
                    copyto!(AV[i], AV_tmp[i])
                end
                
                dim = nsave
                
                @ccall LIB_DIAG.rebuild_heff(
                    N::Int64, dim::Cint, 
                    V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}}, 
                    heff::Ptr{Cdouble}, maxspace::Cint
                )::Cvoid
            end

            shift = max(norm_r * 1e-2, 1e-4)
            
            @ccall LIB_DIAG.apply_preconditioner_inplace(
                N::Int64, axt::Ptr{Cdouble}, diags::Ptr{Cdouble}, 
                min_e::Cdouble, shift::Cdouble
            )::Cvoid

            norm_before = norm(axt) 
            
            norm_after = @ccall LIB_DIAG.mgs_orthogonalize(
                N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
            )::Cdouble

            kappa = 0.717
            if norm_after < kappa * norm_before
                norm_after = @ccall LIB_DIAG.mgs_orthogonalize(
                    N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
                )::Cdouble
            end

            if norm_after < 1e-12
                println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
                xt = Vector{Float64}(undef, N)
                @ccall LIB_DIAG.build_ritz_vector(
                    N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
                    V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
                )::Cvoid
                return e_best, xt
            end
            
            @ccall LIB_DIAG.inplace_normalize(N::Int64, axt::Ptr{Cdouble})::Cdouble

            dim += 1
            
            if length(V) < dim
                push!(V,  Vector{Float64}(undef, N))
                push!(AV, Vector{Float64}(undef, N))
                V_ptrs[dim]  = pointer(V[end])
                AV_ptrs[dim] = pointer(AV[end])
            end

            copyto!(V[dim], axt)
            aop!(V[dim], AV[dim]) 
            
            @ccall LIB_DIAG.compute_heff_col(
                N::Int64, dim::Cint, 
                V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs[dim]::Ptr{Cdouble}, 
                heff_col_buf::Ptr{Cdouble}
            )::Cvoid
            
            for j in 1:dim
                hij = heff_col_buf[j]
                heff[j, dim] = hij
                heff[dim, j] = hij
            end
        end
    end
    
    println("\nMaximum iterations reached.")
    
    xt = Vector{Float64}(undef, N)
    es, vs = eigen(heff[1:dim, 1:dim])
    min_idx = argmin(real.(es))
    coeffs = vs[:, min_idx]
    @ccall LIB_DIAG.build_ritz_vector(
        N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
        V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
    )::Cvoid
    
    return e_best, xt
end
