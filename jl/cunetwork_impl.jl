
function sync_device!()
    @ccall LIB_CUOTF.sync_device_cuda()::Cvoid
end

mutable struct CuBasisManager
    ptr::Ptr{Cvoid}
end

function CuBasisManager()
    return CuBasisManager(C_NULL)
end

function CuBasisManager(basis::BasisManager)
    ptr = @ccall LIB_CUOTF.build_basisdev_f64(basis.ptr::Ptr{Cvoid})::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create C++ CU_BASIS.")

    obj = CuBasisManager(ptr)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_CUOTF.destroy_basisdev_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

mutable struct CuOTF
    ptr::Ptr{Cvoid}
end

function CuOTF()
    return CuOTF(C_NULL)
end

function CuOTF(otf::OTF)
    ptr = @ccall LIB_CUOTF.build_networkdev_f64(otf.ptr::Ptr{Cvoid})::Ptr{Cvoid}

    ptr == C_NULL && error("Failed to create C++ CU_OTF_NET.")

    obj = CuOTF(ptr)

    finalizer(obj) do o
        if o.ptr != C_NULL
            @ccall LIB_CUOTF.destroy_networkdev_f64(o.ptr::Ptr{Cvoid})::Cvoid
            o.ptr = C_NULL
        end
    end

    return obj
end

function hvec_cuda!(basis::CuBasisManager, otf::CuOTF, src::T1, dst::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
    @ccall LIB_CUOTF.hvec_cuda(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        src::CuPtr{Cdouble}, dst::CuPtr{Cdouble},
    )::Cvoid
end

function expm_cuda!(basis::CuBasisManager, otf::CuOTF, idx::Int64, θ::Float64, vec::T) where {Tv,T<:AbstractArray{Tv,1}}
    @ccall LIB_CUOTF.expm_cuda(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx-1)::Int64, θ::Cdouble, vec::CuPtr{Cdouble},
    )::Cvoid
end

function grad_cuda(basis::CuBasisManager, otf::CuOTF, idx::Int64, θ::Float64, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
    return @ccall LIB_CUOTF.grad_cuda(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble
end

function backgrad_cuda!(basis::CuBasisManager, otf::CuOTF, idx::Int64, θ::Float64, lv::T1, rv::T2) where {Tv,T1<:AbstractArray{Tv,1},T2<:AbstractArray{Tv,1}}
    return @ccall LIB_CUOTF.backgrad_cuda(
        basis.ptr::Ptr{Cvoid}, otf.ptr::Ptr{Cvoid},
        (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble
end

struct CuOTF_Functions
    hvec::Function
    expm::Function
    tvec::Function
    grad::Function
    backgrad::Function
    backtran::Function
    batchexpm::Function
    batchgrad::Function
    batchtran::Function
    basis::CuBasisManager
    ham::CuOTF
    pool::CuOTF
end

function CuOTF_Functions(basis::BasisManager, ham::OTF, pool::OTF; info_print::Bool=true, time_print::Bool=false)
    f_hvec      = (v, Hv)               -> nothing
    f_expm      = (idx, θ, v)           -> nothing
    f_tvec      = (idx, lv, rv)         -> nothing
    f_grad      = (idx, θ, lv, rv)      -> nothing
    f_backgrad  = (idx, θ, lv, rv)      -> nothing
    f_backtran  = (idx, θ, lv, rv, tlv) -> nothing
    f_batchexpm = (idx, θ, mat, N, j)   -> nothing
    f_batchgrad = (lv, rv, grads, x)    -> nothing
    f_batchtran = (lv, rv, trans)       -> nothing
    cu_basis    = CuBasisManager(basis)
    cu_ham_otf  = CuOTF(C_NULL)
    cu_pool_otf = CuOTF(C_NULL)

    if ham.ptr != C_NULL
        info_print && print("Uploading Ham OTF to device... ")
        time_ops = @elapsed cu_ham_otf = CuOTF(ham)
        info_print && @printf("Done in %.4f seconds\n", time_ops)
        if time_print
            f_hvec = (v, Hv) -> @printf("hvec time %.6f seconds",
                    @elapsed begin
                        hvec_cuda!(cu_basis, cu_ham_otf, v, Hv)
                        sync_device!()
                    end
                )
        else
            f_hvec = (v, Hv) -> hvec_cuda!(cu_basis, cu_ham_otf, v, Hv)
        end
    end
    if pool.ptr != C_NULL
        info_print && print("Uploading Pool OTF to device ... ")
        time_ops = @elapsed cu_pool_otf = CuOTF(pool)
        info_print && @printf("Done in %.4f seconds\n", time_ops)

        f_expm = (idx, θ, v) -> expm_cuda!(cu_basis, cu_pool_otf, idx, θ, v)
        # f_tvec = (idx, lv, rv) -> tvec_svd!(basis, pool_otf, idx, lv, rv)
        f_grad = (idx, θ, lv, rv) -> return grad_cuda(cu_basis, cu_pool_otf, idx, θ, lv, rv)
        f_backgrad = (idx, θ, lv, rv) -> return backgrad_cuda!(cu_basis, cu_pool_otf, idx, θ, lv, rv)
        # f_backtran = (idx, θ, lv, rv, tlv) -> return back_tran_svd!(basis, pool_otf, idx, θ, lv, rv, tlv)
        # f_batchexpm = (idx, θ, mat, N, j) -> batch_expm_svd!(basis, pool_otf, idx, θ, mat, N, j)
        # f_batchgrad = (lv, rv, grads, x) -> return batch_grad_svd(basis, pool_otf, x, lv, rv, grads)
        # f_batchtran = (lv, rv, trans) -> return batch_tran_svd(basis, pool_otf, lv, rv, trans)
    end

    return CuOTF_Functions(
        f_hvec, f_expm, f_tvec, f_grad,
        f_backgrad, f_backtran,
        f_batchexpm, f_batchgrad, f_batchtran,
        cu_basis, cu_ham_otf, cu_pool_otf)
end
