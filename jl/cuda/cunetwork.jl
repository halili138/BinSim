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

struct CuOTF_Functions
    hvec::Function
    get_diags::Function
    expm::Function
    tvec::Function
    grad::Function
    backgrad::Function
    backtran::Function
    batchexpm::Function
    batchgrad::Function
    batchtran::Function
    expm_2d::Function
    grad_2d::Function
    backgrad_2d::Function
    basis::CuBasisManager
    ham::CuOTF
    pool::CuOTF
end

function CuOTF_Functions(basis::BasisManager, ham::OTF, pool::OTF; info_print::Bool=true, time_print::Bool=false)
    f_hvec       = (v, Hv)               -> nothing
    f_get_diags  = dv                    -> nothing
    f_expm       = (idx, θ, v)           -> nothing
    f_tvec       = (idx, lv, rv)         -> nothing
    f_grad       = (idx, θ, lv, rv)      -> nothing
    f_backgrad   = (idx, θ, lv, rv)      -> nothing
    f_backtran   = (idx, θ, lv, rv, tlv) -> nothing
    f_batchexpm  = (idx, θ, mat, N, j)   -> nothing
    f_batchgrad  = (lv, rv, grads, x)    -> nothing
    f_batchtran  = (lv, rv, trans)       -> nothing
    f_expm_2d    = (idx, θ, v)           -> nothing
    f_grad_2d    = (idx, θ, lv, rv)      -> nothing
    f_backgrad_2d= (idx, θ, lv, rv)      -> nothing
    cu_basis     = CuBasisManager(basis)
    cu_ham_otf   = CuOTF(C_NULL)
    cu_pool_otf  = CuOTF(C_NULL)

    if ham.ptr != C_NULL
        info_print && print("Uploading Ham OTF to device... ")
        time_ops = @elapsed cu_ham_otf = CuOTF(ham)
        info_print && @printf("Done in %.4f seconds\n", time_ops)
    end

    if pool.ptr != C_NULL
        info_print && print("Uploading Pool OTF to device ... ")
        time_ops = @elapsed cu_pool_otf = CuOTF(pool)
        info_print && @printf("Done in %.4f seconds\n", time_ops)
    end
    
    if time_print
        f_hvec = (v, Hv) -> begin
            time_ops = @elapsed @ccall LIB_CUOTF.hvec_cuda(
                cu_basis.ptr::Ptr{Cvoid}, cu_ham_otf.ptr::Ptr{Cvoid}, v::CuPtr{Cdouble}, Hv::CuPtr{Cdouble}
            )::Cvoid
            sync_device!()
            @printf("hvec time %.6f seconds", time_ops)
        end
    else
        f_hvec = (v, Hv) -> @ccall LIB_CUOTF.hvec_cuda(
            cu_basis.ptr::Ptr{Cvoid}, cu_ham_otf.ptr::Ptr{Cvoid}, v::CuPtr{Cdouble}, Hv::CuPtr{Cdouble},
        )::Cvoid
    end


    f_get_diags = dv -> @ccall LIB_CUOTF.get_diags_elements_cuda(
        cu_basis.ptr::Ptr{Cvoid}, cu_ham_otf.ptr::Ptr{Cvoid}, dv::CuPtr{Cdouble},
    )::Cvoid


    f_expm = (idx, θ, v) -> @ccall LIB_CUOTF.expm_cuda(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, v::CuPtr{Cdouble},
    )::Cvoid

    f_grad = (idx, θ, lv, rv) -> return @ccall LIB_CUOTF.grad_cuda(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble

    f_backgrad = (idx, θ, lv, rv) -> return @ccall LIB_CUOTF.backgrad_cuda(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble

    f_batchgrad = (lv, rv, grads, x) -> @ccall LIB_CUOTF.batchgrad_cuda(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, x::CuPtr{Cdouble}, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble}, grads::CuPtr{Cdouble},
    )::Cvoid

    f_expm_2d = (idx, θ, v) -> @ccall LIB_CUOTF.expm_cuda_2d(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, v::CuPtr{Cdouble},
    )::Cvoid

    f_grad_2d = (idx, θ, lv, rv) -> return @ccall LIB_CUOTF.grad_cuda_2d(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble

    f_backgrad_2d = (idx, θ, lv, rv) -> return @ccall LIB_CUOTF.backgrad_cuda_2d(
        cu_basis.ptr::Ptr{Cvoid}, cu_pool_otf.ptr::Ptr{Cvoid}, (idx-1)::Int64, θ::Cdouble, lv::CuPtr{Cdouble}, rv::CuPtr{Cdouble},
    )::Cdouble

    return CuOTF_Functions(
        f_hvec, f_get_diags, 
        f_expm, f_tvec, f_grad,
        f_backgrad, f_backtran,
        f_batchexpm, f_batchgrad, f_batchtran,
        f_expm_2d, f_grad_2d, f_backgrad_2d,
        cu_basis, cu_ham_otf, cu_pool_otf
    )
end
