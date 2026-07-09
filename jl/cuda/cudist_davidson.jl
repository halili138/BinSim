function _local_diags_for_gmap(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,TK,TV},
    gmap::GlobalMemMap,
    comm::MPI.Comm,
) where {Ti,Tv,TK,TV}
    funcs = OTF_Functions(basis, ham, BinaryQubitAABB{Ti,Tv,TK,TV}[]; info_print=false, time_print=false)
    diag_full = get_diags(basis, funcs.ham, Float64)
    dims = MPI.Allgather(Int64(gmap.local_dim), comm)
    offset = sum(dims[1:gmap.mpi_rank]; init=Int64(0))
    return diag_full[offset + 1 : offset + gmap.local_dim]
end

function davidson_cuda_distributed(
    funcs::CuDistributedFunctions{ModeNVLink},
    v0::CuVector{Float64},
    diags::CuVector{Float64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,
    maxspace::Int=ncv+20,
    verbose::Bool=true,
    save_path::String="",
)
    @assert maxspace > ncv

    comm = funcs.comm
    rank = funcs.rank
    N = length(v0)

    axt = CUDA.zeros(Float64, N)
    V = CuVector{Float64}[]
    AV = CuVector{Float64}[]
    V_tmp = CuVector{Float64}[]
    AV_tmp = CuVector{Float64}[]
    heff = zeros(Float64, maxspace, maxspace)

    push!(V, copy(v0))
    funcs.normalize(V[1])
    push!(AV, funcs.zeros())
    funcs.hvec(V[1], AV[1])
    e_best = funcs.inner(V[1], AV[1])
    heff[1, 1] = e_best
    dim = 1

    coeffs = ones(Float64, 1)
    for iter in 1:maxiter
        es, vs = eigen(heff[1:dim, 1:dim])
        min_idx = argmin(real.(es))
        min_e = real(es[min_idx])
        coeffs = real.(vs[:, min_idx])

        fill!(axt, 0.0)
        for j in 1:dim
            @. axt += coeffs[j] * (AV[j] - min_e * V[j])
        end
        norm_r = sqrt(funcs.inner(axt, axt))
        verbose && rank == 0 && @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
        e_best = min_e

        if norm_r < tol
            xt = funcs.zeros()
            fill!(xt, 0.0)
            for j in 1:dim
                @. xt += coeffs[j] * V[j]
            end
            funcs.normalize(xt)
            if !isempty(save_path) && rank == 0
                JLD2.jldopen(save_path, "w") do file
                    file["e_best"] = e_best
                    file["v_best_local"] = Array(xt)
                end
            end
            rank == 0 && println("\nConvergence reached!")
            return e_best, xt
        end

        if dim >= maxspace
            nsave = min(ncv, dim)
            p = sortperm(real.(es))
            while length(V_tmp) < nsave
                push!(V_tmp, funcs.zeros())
                push!(AV_tmp, funcs.zeros())
            end
            for i in 1:nsave
                idx = p[i]
                fill!(V_tmp[i], 0.0)
                fill!(AV_tmp[i], 0.0)
                for j in 1:dim
                    c = real(vs[j, idx])
                    @. V_tmp[i] += c * V[j]
                    @. AV_tmp[i] += c * AV[j]
                end
            end
            for i in 1:nsave
                copyto!(V[i], V_tmp[i])
                copyto!(AV[i], AV_tmp[i])
            end
            dim = nsave
            fill!(heff, 0.0)
            for j in 1:dim, i in 1:j
                h = funcs.inner(V[i], AV[j])
                heff[i, j] = h
                heff[j, i] = h
            end
        end

        shift = max(norm_r * 1e-2, 1e-4)
        @. axt = axt / (min_e - diags + shift)
        norm_before = sqrt(funcs.inner(axt, axt))
        for pass in 1:2
            for j in 1:dim
                c = funcs.inner(V[j], axt)
                @. axt -= c * V[j]
            end
            norm_after = sqrt(funcs.inner(axt, axt))
            (pass == 2 || norm_after >= 0.717 * norm_before) && break
        end
        norm_after = sqrt(funcs.inner(axt, axt))
        if norm_after < 1e-12
            rank == 0 && println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
            xt = funcs.zeros()
            fill!(xt, 0.0)
            for j in 1:dim
                @. xt += coeffs[j] * V[j]
            end
            funcs.normalize(xt)
            return e_best, xt
        end
        axt ./= norm_after

        dim += 1
        if length(V) < dim
            push!(V, funcs.zeros())
            push!(AV, funcs.zeros())
        end
        copyto!(V[dim], axt)
        funcs.hvec(V[dim], AV[dim])
        for j in 1:dim
            hij = funcs.inner(V[j], AV[dim])
            heff[j, dim] = hij
            heff[dim, j] = hij
        end
    end

    rank == 0 && println("\nMaximum iterations reached.")
    xt = funcs.zeros()
    fill!(xt, 0.0)
    for j in 1:dim
        @. xt += coeffs[j] * V[j]
    end
    funcs.normalize(xt)
    return e_best, xt
end


function davidson_cuda_distributed_host_memory(
    funcs::CuDistributedFunctions{ModeNVLink},
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

    comm = funcs.comm
    rank = funcs.rank
    N = length(v0)

    axt = Vector{Float64}(undef, N)
    d_v = funcs.zeros()
    d_av = funcs.zeros()
    V = Vector{Float64}[]
    AV = Vector{Float64}[]
    V_tmp = Vector{Float64}[]
    AV_tmp = Vector{Float64}[]
    heff = zeros(Float64, maxspace, maxspace)

    host_inner(lv::Vector{Float64}, rv::Vector{Float64}) = MPI.Allreduce(dot(lv, rv), +, comm)
    function host_normalize!(v::Vector{Float64})
        nrm = sqrt(host_inner(v, v))
        v ./= nrm
        return v
    end
    function hvec_from_host!(src::Vector{Float64}, dst::Vector{Float64})
        copyto!(d_v, src)
        funcs.hvec(d_v, d_av)
        copyto!(dst, Array(d_av))
        return dst
    end

    push!(V, copy(v0))
    host_normalize!(V[1])
    push!(AV, Vector{Float64}(undef, N))
    hvec_from_host!(V[1], AV[1])
    e_best = host_inner(V[1], AV[1])
    heff[1, 1] = e_best
    dim = 1

    coeffs = ones(Float64, 1)
    for iter in 1:maxiter
        es, vs = eigen(heff[1:dim, 1:dim])
        min_idx = argmin(real.(es))
        min_e = real(es[min_idx])
        coeffs = real.(vs[:, min_idx])

        fill!(axt, 0.0)
        for j in 1:dim
            @. axt += coeffs[j] * (AV[j] - min_e * V[j])
        end
        norm_r = sqrt(host_inner(axt, axt))
        verbose && rank == 0 && @printf("  Step %03d:  residual: %.4e,  e: %.14f\n", iter, norm_r, min_e)
        e_best = min_e

        if norm_r < tol
            xt = zeros(Float64, N)
            for j in 1:dim
                @. xt += coeffs[j] * V[j]
            end
            host_normalize!(xt)
            if !isempty(save_path) && rank == 0
                JLD2.jldopen(save_path, "w") do file
                    file["e_best"] = e_best
                    file["v_best_local"] = xt
                end
            end
            rank == 0 && println("\nConvergence reached!")
            return e_best, xt
        end

        if dim >= maxspace
            nsave = min(ncv, dim)
            p = sortperm(real.(es))
            while length(V_tmp) < nsave
                push!(V_tmp, Vector{Float64}(undef, N))
                push!(AV_tmp, Vector{Float64}(undef, N))
            end
            for i in 1:nsave
                idx = p[i]
                fill!(V_tmp[i], 0.0)
                fill!(AV_tmp[i], 0.0)
                for j in 1:dim
                    c = real(vs[j, idx])
                    @. V_tmp[i] += c * V[j]
                    @. AV_tmp[i] += c * AV[j]
                end
            end
            for i in 1:nsave
                copyto!(V[i], V_tmp[i])
                copyto!(AV[i], AV_tmp[i])
            end
            dim = nsave
            fill!(heff, 0.0)
            for j in 1:dim, i in 1:j
                h = host_inner(V[i], AV[j])
                heff[i, j] = h
                heff[j, i] = h
            end
        end

        shift = max(norm_r * 1e-2, 1e-4)
        @. axt = axt / (min_e - diags + shift)
        norm_before = sqrt(host_inner(axt, axt))
        for pass in 1:2
            for j in 1:dim
                c = host_inner(V[j], axt)
                @. axt -= c * V[j]
            end
            norm_after = sqrt(host_inner(axt, axt))
            (pass == 2 || norm_after >= 0.717 * norm_before) && break
        end
        norm_after = sqrt(host_inner(axt, axt))
        if norm_after < 1e-12
            rank == 0 && println("Breakdown: new vector lies in the current subspace (Linear Dependent)")
            xt = zeros(Float64, N)
            for j in 1:dim
                @. xt += coeffs[j] * V[j]
            end
            host_normalize!(xt)
            return e_best, xt
        end
        axt ./= norm_after

        dim += 1
        if length(V) < dim
            push!(V, Vector{Float64}(undef, N))
            push!(AV, Vector{Float64}(undef, N))
        end
        copyto!(V[dim], axt)
        hvec_from_host!(V[dim], AV[dim])
        for j in 1:dim
            hij = host_inner(V[j], AV[dim])
            heff[j, dim] = hij
            heff[dim, j] = hij
        end
    end

    rank == 0 && println("\nMaximum iterations reached.")
    xt = zeros(Float64, N)
    for j in 1:dim
        @. xt += coeffs[j] * V[j]
    end
    host_normalize!(xt)
    return e_best, xt
end

function run_fci_host_memory(
    funcs::CuDistributedFunctions{ModeNVLink},
    basis::BasisManager,
    ham::BinaryQubitAABB,
    v0::Vector{Float64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,
    maxspace::Int=ncv+20,
    save_path::String="",
    verbose::Bool=true,
)
    funcs.rank == 0 && print("Generating local Diag elements vector for CUDA NVLink host-memory Davidson ... ")
    time_ops = @elapsed h_diags = _local_diags_for_gmap(basis, ham, GlobalMemMap(basis, funcs.comm), funcs.comm)
    funcs.rank == 0 && @printf("Done in %.4f seconds\n", time_ops)

    funcs.rank == 0 && println("Solving CUDA NVLink FCI with Davidson (V/AV resident in host memory) ... ")
    result = @timed davidson_cuda_distributed_host_memory(
        funcs, v0, h_diags;
        maxiter=maxiter, tol=tol, ncv=ncv, maxspace=maxspace,
        verbose=verbose, save_path=save_path,
    )
    funcs.rank == 0 && @printf("Done in %.4f seconds\n\n", result.time)
    return result.value
end

function run_fci_host_memory(
    funcs::CuDistributedFunctions{ModeNVLink},
    basis::BasisManager,
    ham::BinaryQubitAABB,
    v0::CuVector{Float64};
    kwargs...,
)
    return run_fci_host_memory(funcs, basis, ham, Array(v0); kwargs...)
end

function run_fci(
    funcs::CuDistributedFunctions{ModeNVLink},
    basis::BasisManager,
    ham::BinaryQubitAABB,
    v0::CuVector{Float64};
    maxiter::Int=100000,
    tol::Float64=1e-5,
    ncv::Int=3,
    maxspace::Int=ncv+20,
    save_path::String="",
    verbose::Bool=true,
)
    funcs.rank == 0 && print("Generating local Diag elements vector for CUDA NVLink Davidson ... ")
    time_ops = @elapsed h_diags = _local_diags_for_gmap(basis, ham, GlobalMemMap(basis, funcs.comm), funcs.comm)
    funcs.rank == 0 && @printf("Done in %.4f seconds\n", time_ops)
    d_diags = CuArray(h_diags)

    funcs.rank == 0 && println("Solving CUDA NVLink FCI with Davidson (V/AV resident in GPU memory) ... ")
    result = @timed davidson_cuda_distributed(
        funcs, v0, d_diags;
        maxiter=maxiter, tol=tol, ncv=ncv, maxspace=maxspace,
        verbose=verbose, save_path=save_path,
    )
    funcs.rank == 0 && @printf("Done in %.4f seconds\n\n", result.time)
    return result.value
end
