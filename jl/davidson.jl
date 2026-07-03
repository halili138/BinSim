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
    comm=nothing,
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
    
    if comm !== nothing
        ln2 = MPI.Allreduce(dot(V[1], V[1]), +, comm)
        V[1] ./= sqrt(ln2)
    else
        @ccall LIB_DIAG.inplace_normalize_f64(N::Int64, V_ptrs[1]::Ptr{Cdouble})::Cdouble
    end
    
    aop!(V[1], AV[1])
    
    if comm !== nothing
        e_best = MPI.Allreduce(dot(V[1], AV[1]), +, comm)
    else
        e_best = @ccall LIB_DIAG.fast_dot_f64(N::Int64, V_ptrs[1]::Ptr{Cdouble}, AV_ptrs[1]::Ptr{Cdouble})::Cdouble
    end
    
    heff[1,1] = e_best
    
    dim::Int32 = 1 
    
    GC.@preserve V AV V_tmp AV_tmp V_ptrs AV_ptrs axt diags begin
        for iter in 1:maxiter
    
            es, vs  = eigen(heff[1:dim, 1:dim])
            min_idx = argmin(real.(es))
            min_e   = real(es[min_idx])
         
            coeffs  = vs[:, min_idx]

            if comm !== nothing
                fill!(axt, 0.0)
                for j in 1:dim
                    @. axt += coeffs[j] * (AV[j] - min_e * V[j])
                end
                norm_r = sqrt(MPI.Allreduce(dot(axt, axt), +, comm))
            else
                norm_r = @ccall LIB_DIAG.build_ritz_and_residual_f64(
                    N::Int64, dim::Cint, coeffs::Ptr{Cdouble}, min_e::Cdouble,
                    V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}},
                    axt::Ptr{Cdouble}
                )::Cdouble
            end

            verbose && (comm === nothing || MPI.Comm_rank(comm) == 0) &&
                @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
            e_best = min_e

            if norm_r < tol
                if comm !== nothing
                    xt = Vector{Float64}(undef, N)
                    fill!(xt, 0.0)
                    for j in 1:dim
                        @. xt += coeffs[j] * V[j]
                    end
                else
                    xt = Vector{Float64}(undef, N)
                    @ccall LIB_DIAG.build_ritz_vector_f64(
                        N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
                        V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
                    )::Cvoid
                end
                
                if !isempty(save_path)
                    JLD2.jldopen(save_path, "w") do file
                        file["e_best"] = e_best
                        file["v_best"] = xt 
                    end
                end
                verbose && (comm === nothing || MPI.Comm_rank(comm) == 0) && println("\nConvergence reached!")
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
                
                if comm !== nothing
                    for j in 1:dim
                        for i in 1:j
                            h = MPI.Allreduce(dot(V[i], AV[j]), +, comm)
                            heff[i, j] = h
                            heff[j, i] = h
                        end
                    end
                else
                    @ccall LIB_DIAG.rebuild_heff_f64(
                        N::Int64, dim::Cint, 
                        V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}}, 
                        heff::Ptr{Cdouble}, maxspace::Cint
                    )::Cvoid
                end
            end

            shift = max(norm_r * 1e-2, 1e-4)
            
            @ccall LIB_DIAG.apply_preconditioner_inplace_f64(
                N::Int64, axt::Ptr{Cdouble}, diags::Ptr{Cdouble}, 
                min_e::Cdouble, shift::Cdouble
            )::Cvoid

            norm_before = norm(axt) 
            if comm !== nothing
                norm_before = sqrt(MPI.Allreduce(norm_before^2, +, comm))
            end
            
            if comm !== nothing
                for j in 1:dim
                    c = MPI.Allreduce(dot(V[j], axt), +, comm)
                    @. axt -= c * V[j]
                end
                norm_after2 = MPI.Allreduce(dot(axt, axt), +, comm)
                norm_after = sqrt(norm_after2)

                kappa = 0.717
                if norm_after < kappa * norm_before
                    for j in 1:dim
                        c = MPI.Allreduce(dot(V[j], axt), +, comm)
                        @. axt -= c * V[j]
                    end
                    norm_after2 = MPI.Allreduce(dot(axt, axt), +, comm)
                    norm_after = sqrt(norm_after2)
                end
            else
                norm_after = @ccall LIB_DIAG.mgs_orthogonalize_f64(
                    N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
                )::Cdouble

                kappa = 0.717
                if norm_after < kappa * norm_before
                    norm_after = @ccall LIB_DIAG.mgs_orthogonalize_f64(
                        N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
                    )::Cdouble
                end
            end

            if norm_after < 1e-12
                (comm === nothing || MPI.Comm_rank(comm) == 0) &&
                    println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
                if comm !== nothing
                    xt = Vector{Float64}(undef, N)
                    fill!(xt, 0.0)
                    for j in 1:dim
                        @. xt += coeffs[j] * V[j]
                    end
                else
                    xt = Vector{Float64}(undef, N)
                    @ccall LIB_DIAG.build_ritz_vector_f64(
                        N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
                        V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
                    )::Cvoid
                end
                return e_best, xt
            end
            
            if comm !== nothing
                ln2 = MPI.Allreduce(dot(axt, axt), +, comm)
                axt ./= sqrt(ln2)
            else
                @ccall LIB_DIAG.inplace_normalize_f64(N::Int64, axt::Ptr{Cdouble})::Cdouble
            end

            dim += 1
            
            if length(V) < dim
                push!(V,  Vector{Float64}(undef, N))
                push!(AV, Vector{Float64}(undef, N))
                V_ptrs[dim]  = pointer(V[end])
                AV_ptrs[dim] = pointer(AV[end])
            end

            copyto!(V[dim], axt)
            aop!(V[dim], AV[dim]) 
            
            if comm !== nothing
                for j in 1:dim
                    hij = MPI.Allreduce(dot(V[j], AV[dim]), +, comm)
                    heff[j, dim] = hij
                    heff[dim, j] = hij
                end
            else
                @ccall LIB_DIAG.compute_heff_col_f64(
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
    end
    
    (comm === nothing || MPI.Comm_rank(comm) == 0) && println("\nMaximum iterations reached.")
    
    if comm !== nothing
        xt = Vector{Float64}(undef, N)
        fill!(xt, 0.0)
        for j in 1:dim
            @. xt += coeffs[j] * V[j]
        end
    else
        xt = Vector{Float64}(undef, N)
        es, vs = eigen(heff[1:dim, 1:dim])
        min_idx = argmin(real.(es))
        coeffs = vs[:, min_idx]
        @ccall LIB_DIAG.build_ritz_vector_f64(
            N::Int64, dim::Cint, coeffs::Ptr{Cdouble},
            V_ptrs::Ptr{Ptr{Cdouble}}, xt::Ptr{Cdouble}
        )::Cvoid
    end
    
    return e_best, xt
end


function davidson(
    aop!::Function,
    v0::Vector{ComplexF64},
    diags::Vector{ComplexF64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,          
    maxspace::Int=ncv+20,
    verbose::Bool=true,
    save_path::String="",
    comm=nothing,
)
    @assert maxspace > ncv

    N = length(v0)
    
    axt = Vector{ComplexF64}(undef, N) 
    heff_col_buf = Vector{ComplexF64}(undef, maxspace)

    V  = Vector{Vector{ComplexF64}}()
    AV = Vector{Vector{ComplexF64}}()
    
    V_ptrs  = Vector{Ptr{ComplexF64}}(undef, maxspace)
    AV_ptrs = Vector{Ptr{ComplexF64}}(undef, maxspace)

    V_tmp  = Vector{Vector{ComplexF64}}()
    AV_tmp = Vector{Vector{ComplexF64}}()

    heff = zeros(ComplexF64, maxspace, maxspace)

    push!(V, v0)
    push!(AV, Vector{ComplexF64}(undef, N))
    
    V_ptrs[1]  = pointer(V[1])
    AV_ptrs[1] = pointer(AV[1])
    
    if comm !== nothing
        ln2 = MPI.Allreduce(real(dot(V[1], V[1])), +, comm)
        V[1] ./= sqrt(ln2)
    else
        @ccall LIB_DIAG.inplace_normalize_c64(N::Int64, V_ptrs[1]::Ptr{ComplexF64})::Cdouble
    end
    
    aop!(V[1], AV[1])
    
    if comm !== nothing
        e_best = MPI.Allreduce(dot(V[1], AV[1]), +, comm)
    else
        e_best = @ccall LIB_DIAG.fast_dot_c64(N::Int64, V_ptrs[1]::Ptr{ComplexF64}, AV_ptrs[1]::Ptr{ComplexF64})::ComplexF64
    end
    heff[1,1] = e_best
    
    dim::Int32 = 1 
    
    GC.@preserve V AV V_tmp AV_tmp V_ptrs AV_ptrs axt diags begin
        for iter in 1:maxiter
    
            es, vs  = eigen(heff[1:dim, 1:dim])
            min_idx = argmin(real.(es))
            min_e   = real(es[min_idx])
            coeffs  = vs[:, min_idx]

            if comm !== nothing
                fill!(axt, 0.0)
                for j in 1:dim
                    @. axt += coeffs[j] * (AV[j] - min_e * V[j])
                end
                norm_r2 = real(MPI.Allreduce(dot(axt, axt), +, comm))
                norm_r = sqrt(max(0.0, norm_r2))
            else
                norm_r = @ccall LIB_DIAG.build_ritz_and_residual_c64(
                    N::Int64, dim::Cint, coeffs::Ptr{ComplexF64}, min_e::Cdouble,
                    V_ptrs::Ptr{Ptr{ComplexF64}}, AV_ptrs::Ptr{Ptr{ComplexF64}},
                    axt::Ptr{ComplexF64}
                )::Cdouble
            end

            verbose && (comm === nothing || MPI.Comm_rank(comm) == 0) &&
                @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
            
            e_best = min_e

            if norm_r < tol
                if comm !== nothing
                    xt = Vector{ComplexF64}(undef, N)
                    fill!(xt, 0.0)
                    for j in 1:dim
                        @. xt += coeffs[j] * V[j]
                    end
                else
                    xt = Vector{ComplexF64}(undef, N)
                    @ccall LIB_DIAG.build_ritz_vector_c64(
                        N::Int64, dim::Cint, coeffs::Ptr{ComplexF64},
                        V_ptrs::Ptr{Ptr{ComplexF64}}, xt::Ptr{ComplexF64}
                    )::Cvoid
                end
                
                if !isempty(save_path)
                    JLD2.jldopen(save_path, "w") do file
                        file["e_best"] = e_best
                        file["v_best"] = xt 
                    end
                end
                verbose && (comm === nothing || MPI.Comm_rank(comm) == 0) && println("\nConvergence reached!")
                return e_best, xt
            end

            if dim >= maxspace
                nsave = min(ncv, dim)
                p = sortperm(real.(es))
                
                while length(V_tmp) < nsave
                    push!(V_tmp,  Vector{ComplexF64}(undef, N))
                    push!(AV_tmp, Vector{ComplexF64}(undef, N))
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
                
                if comm !== nothing
                    for j in 1:dim
                        for i in 1:j
                            h = MPI.Allreduce(dot(V[i], AV[j]), +, comm)
                            heff[i, j] = h
                            heff[j, i] = conj(h)
                        end
                    end
                else
                    @ccall LIB_DIAG.rebuild_heff_c64(
                        N::Int64, dim::Cint, 
                        V_ptrs::Ptr{Ptr{ComplexF64}}, AV_ptrs::Ptr{Ptr{ComplexF64}}, 
                        heff::Ptr{ComplexF64}, maxspace::Cint
                    )::Cvoid
                end
            end

            shift = max(norm_r * 1e-2, 1e-4)
            
            @ccall LIB_DIAG.apply_preconditioner_inplace_c64(
                N::Int64, axt::Ptr{ComplexF64}, diags::Ptr{ComplexF64}, 
                min_e::Cdouble, shift::Cdouble
            )::Cvoid

            lnsq = real(dot(axt, axt))
            norm_before = comm !== nothing ? sqrt(MPI.Allreduce(lnsq, +, comm)) : sqrt(lnsq)
            
            if comm !== nothing
                for j in 1:dim
                    c = MPI.Allreduce(dot(V[j], axt), +, comm)
                    @. axt -= c * V[j]
                end
                norm_after2 = real(MPI.Allreduce(dot(axt, axt), +, comm))
                norm_after = sqrt(max(0.0, norm_after2))

                kappa = 0.717
                if norm_after < kappa * norm_before
                    for j in 1:dim
                        c = MPI.Allreduce(dot(V[j], axt), +, comm)
                        @. axt -= c * V[j]
                    end
                    norm_after2 = real(MPI.Allreduce(dot(axt, axt), +, comm))
                    norm_after = sqrt(max(0.0, norm_after2))
                end
            else
                norm_after = @ccall LIB_DIAG.mgs_orthogonalize_c64(
                    N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{ComplexF64}}, axt::Ptr{ComplexF64}
                )::Cdouble

                kappa = 0.717
                if norm_after < kappa * norm_before
                    norm_after = @ccall LIB_DIAG.mgs_orthogonalize_c64(
                        N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{ComplexF64}}, axt::Ptr{ComplexF64}
                    )::Cdouble
                end
            end

            if norm_after < 1e-12
                (comm === nothing || MPI.Comm_rank(comm) == 0) &&
                    println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
                if comm !== nothing
                    xt = Vector{ComplexF64}(undef, N)
                    fill!(xt, 0.0)
                    for j in 1:dim
                        @. xt += coeffs[j] * V[j]
                    end
                else
                    xt = Vector{ComplexF64}(undef, N)
                    @ccall LIB_DIAG.build_ritz_vector_c64(
                        N::Int64, dim::Cint, coeffs::Ptr{ComplexF64},
                        V_ptrs::Ptr{Ptr{ComplexF64}}, xt::Ptr{ComplexF64}
                    )::Cvoid
                end
                return e_best, xt
            end
            
            if comm !== nothing
                ln2 = real(MPI.Allreduce(dot(axt, axt), +, comm))
                axt ./= sqrt(max(0.0, ln2))
            else
                @ccall LIB_DIAG.inplace_normalize_c64(N::Int64, axt::Ptr{ComplexF64})::Cdouble
            end

            dim += 1
            
            if length(V) < dim
                push!(V,  Vector{ComplexF64}(undef, N))
                push!(AV, Vector{ComplexF64}(undef, N))
                V_ptrs[dim]  = pointer(V[end])
                AV_ptrs[dim] = pointer(AV[end])
            end

            copyto!(V[dim], axt)
            aop!(V[dim], AV[dim]) 
            
            if comm !== nothing
                for j in 1:dim
                    hij = MPI.Allreduce(dot(V[j], AV[dim]), +, comm)
                    heff[j, dim] = hij
                    heff[dim, j] = conj(hij)
                end
            else
                @ccall LIB_DIAG.compute_heff_col_c64(
                    N::Int64, dim::Cint, 
                    V_ptrs::Ptr{Ptr{ComplexF64}}, AV_ptrs[dim]::Ptr{ComplexF64}, 
                    heff_col_buf::Ptr{ComplexF64}
                )::Cvoid
                
                for j in 1:dim
                    hij = heff_col_buf[j]
                    heff[j, dim] = hij
                    heff[dim, j] = conj(hij)
                end
            end
        end
    end
    
    (comm === nothing || MPI.Comm_rank(comm) == 0) && println("\nMaximum iterations reached.")
    
    if comm !== nothing
        xt = Vector{ComplexF64}(undef, N)
        fill!(xt, 0.0)
        for j in 1:dim
            @. xt += coeffs[j] * V[j]
        end
    else
        xt = Vector{ComplexF64}(undef, N)
        es, vs = eigen(heff[1:dim, 1:dim])
        min_idx = argmin(real.(es))
        coeffs = vs[:, min_idx]
        @ccall LIB_DIAG.build_ritz_vector_c64(
            N::Int64, dim::Cint, coeffs::Ptr{ComplexF64},
            V_ptrs::Ptr{Ptr{ComplexF64}}, xt::Ptr{ComplexF64}
        )::Cvoid
    end
    
    return e_best, xt
end


function davidson2(
    aop!::Function,
    v0::Vector{Float64},
    diags::Vector{Float64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,          
    maxspace::Int=ncv+20,
    verbose::Bool=true,
    save_path::String="",
    comm=nothing,
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
    
    if comm !== nothing
        ln2 = dot(V[1], V[1])
        V[1] ./= sqrt(ln2)
    else
        @ccall LIB_DIAG.inplace_normalize_f64(N::Int64, V_ptrs[1]::Ptr{Cdouble})::Cdouble
    end
    
    aop!(V[1], AV[1])
    
    if comm !== nothing
        e_best = dot(V[1], AV[1])
    else
        e_best = @ccall LIB_DIAG.fast_dot_f64(N::Int64, V_ptrs[1]::Ptr{Cdouble}, AV_ptrs[1]::Ptr{Cdouble})::Cdouble
    end
    
    heff[1,1] = e_best
    dim::Int32 = 1 
    
    GC.@preserve V AV V_tmp AV_tmp V_ptrs AV_ptrs axt diags begin
        for iter in 1:maxiter
    
            es, vs  = eigen(heff[1:dim, 1:dim])
            min_idx = argmin(real.(es))
            min_e   = real(es[min_idx])
            coeffs  = vs[:, min_idx]

            if comm !== nothing
                fill!(axt, 0.0)
                for j in 1:dim
                    @. axt += coeffs[j] * (AV[j] - min_e * V[j])
                end
                norm_r = sqrt(dot(axt, axt))
            else
                norm_r = @ccall LIB_DIAG.build_ritz_and_residual_f64(
                    N::Int64, dim::Cint, coeffs::Ptr{Cdouble}, min_e::Cdouble,
                    V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}},
                    axt::Ptr{Cdouble}
                )::Cdouble
            end

            verbose && comm === nothing &&
                @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
            e_best = min_e

            # ---------------------------------------------------------
            # 退出点 1: 正常收敛
            # ---------------------------------------------------------
            if norm_r < tol
                if !isempty(save_path)
                    JLD2.jldopen(save_path, "w") do file
                        file["e_best"] = e_best
                        # 斩断: 不再提取和保存 xt
                    end
                end
                verbose && comm === nothing && println("\nConvergence reached!")
                return e_best
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
                
                if comm !== nothing
                    for j in 1:dim
                        for i in 1:j
                            h = dot(V[i], AV[j])
                            heff[i, j] = h
                            heff[j, i] = h
                        end
                    end
                else
                    @ccall LIB_DIAG.rebuild_heff_f64(
                        N::Int64, dim::Cint, 
                        V_ptrs::Ptr{Ptr{Cdouble}}, AV_ptrs::Ptr{Ptr{Cdouble}}, 
                        heff::Ptr{Cdouble}, maxspace::Cint
                    )::Cvoid
                end
            end

            shift = max(norm_r * 1e-2, 1e-4)
            
            @ccall LIB_DIAG.apply_preconditioner_inplace_f64(
                N::Int64, axt::Ptr{Cdouble}, diags::Ptr{Cdouble}, 
                min_e::Cdouble, shift::Cdouble
            )::Cvoid

            norm_before = norm(axt) 
            if comm !== nothing
                norm_before = sqrt(norm_before ^ 2)
            end
            
            if comm !== nothing
                for j in 1:dim
                    c = dot(V[j], axt)
                    @. axt -= c * V[j]
                end
                norm_after2 = dot(axt, axt)
                norm_after = sqrt(norm_after2)

                kappa = 0.717
                if norm_after < kappa * norm_before
                    for j in 1:dim
                        c = dot(V[j], axt)
                        @. axt -= c * V[j]
                    end
                    norm_after2 = dot(axt, axt)
                    norm_after = sqrt(norm_after2)
                end
            else
                norm_after = @ccall LIB_DIAG.mgs_orthogonalize_f64(
                    N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
                )::Cdouble

                kappa = 0.717
                if norm_after < kappa * norm_before
                    norm_after = @ccall LIB_DIAG.mgs_orthogonalize_f64(
                        N::Int64, dim::Cint, V_ptrs::Ptr{Ptr{Cdouble}}, axt::Ptr{Cdouble}
                    )::Cdouble
                end
            end

            # ---------------------------------------------------------
            # 退出点 2: 线性相关击穿
            # ---------------------------------------------------------
            if norm_after < 1e-12
                comm === nothing && println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
                return e_best
            end
            
            if comm !== nothing
                ln2 = dot(axt, axt)
                axt ./= sqrt(ln2)
            else
                @ccall LIB_DIAG.inplace_normalize_f64(N::Int64, axt::Ptr{Cdouble})::Cdouble
            end

            dim += 1
            
            if length(V) < dim
                push!(V,  Vector{Float64}(undef, N))
                push!(AV, Vector{Float64}(undef, N))
                V_ptrs[dim]  = pointer(V[end])
                AV_ptrs[dim] = pointer(AV[end])
            end

            copyto!(V[dim], axt)
            aop!(V[dim], AV[dim]) 
            
            if comm !== nothing
                for j in 1:dim
                    hij = dot(V[j], AV[dim])
                    heff[j, dim] = hij
                    heff[dim, j] = hij
                end
            else
                @ccall LIB_DIAG.compute_heff_col_f64(
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
    end
    
    # ---------------------------------------------------------
    # 退出点 3: 达到最大迭代次数
    # ---------------------------------------------------------
    comm === nothing  && println("\nMaximum iterations reached.")
    return e_best
end
