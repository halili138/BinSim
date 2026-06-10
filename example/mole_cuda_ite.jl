include("../binsim.jl")
using CUDA

const LIB_CUOTF = joinpath(@__DIR__, "../src/lib/libcuotf.so")

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
            f_hvec = (v, Hv) -> @printf(
                "hvec time %.6f seconds", 
                @elapsed hvec_cuda!(cu_basis, cu_ham_otf, v, Hv))
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
        # f_grad = (idx, θ, lv, rv) -> return grad_svd(basis, pool_otf, idx, θ, lv, rv)
        # f_backgrad = (idx, θ, lv, rv) -> return back_grad_svd!(basis, pool_otf, idx, θ, lv, rv)
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

function sync_device!()
    @ccall LIB_CUOTF.sync_device_cuda()::Cvoid
end

function run_euler_ite_cuda(
    basis::BasisManager,
    ham::BinaryQubitAABB{Ti,Tv,K,V},
    v0::Vector{Tv},
    e_scale::Float64;
    dτ::Float64=0.1,
    max_step::Int64=5000,
    tol::Float64=1e-10,
) where {Ti,Tv,K,V}
    otf = OTF(basis, ham)
    cubasis = CuBasisManager(basis)
    cuotf = CuOTF(otf)
    v = CuArray{Tv,1,CUDA.DeviceMemory}(v0)
    w = CUDA.zeros(Tv, basis.dim)

    E_hist = Float64[]
    dH_hist = Float64[]

    step = 0
    while step <= max_step
        step += 1
        @time begin
            hvec_cuda!(cubasis, cuotf, v, w)
            ln = norm(v)^2
        end
        rn = norm(w)^2
        E = real(dot(v, w)) / ln
        dH = max(0.0, rn / ln - E^2)
        push!(E_hist, E)
        push!(dH_hist, dH)

        dE = step > 1 ? abs(E_hist[end] - E_hist[end-1]) : abs(E_hist[end])

        @printf("  Step %04d  E %.14f  Err %.3e  dE %.3e  δ²H %.3e\n",
            step, E, abs(E - e_scale), dE, dH)

        dE < tol && break

        @. v -= dτ * w
        normalize!(v)
    end

    println("\n  Converged at step $step")
    return E_hist[end]
end

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name = ARGS[1]
    mole.ratio = parse(Float64, ARGS[2])
    mole.basis = ARGS[3]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    ham   = JW_hamiltonian(mole)
    orbs  = Orbitals(); kernel(mole, orbs, generalize=false)
    pool  = FEB(orbs)

    # funcs    = OTF_Functions(basis, ham, pool)
    ham_otf  = OTF(basis, ham)
    pool_otf = OTF(basis, pool)
    cu_funcs = CuOTF_Functions(basis, ham_otf, pool_otf)
    # mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))


    # run_euler_ite_cuda(basis, ham, get_hf(basis, mole.nelec), mole.e_scale,
    #     dτ=0.1, max_step=5, tol=1e-8)

    # nparas = length(pool)
    # amps   = rand(Float64, nparas)
    # idxs   = [i for i in 1:nparas]
    # lv     = CUDA.rand(Float64,  basis.dim)
    # rv     = CUDA.zeros(Float64, basis.dim)

    # @time begin
    #     for i in 1:nparas
    #         cu_funcs.expm(idxs[i], amps[i], lv)
    #     end

    #     cu_funcs.hvec(lv, rv)

    #     lnorm  = norm(lv) ^ 2
    #     rnorm  = norm(rv) ^ 2
    #     energy = real(dot(lv, rv)) / lnorm
    # end

    nparas = length(pool)
    amps   = rand(Float64, nparas)
    idxs   = [i for i in 1:nparas]
    lv     = CUDA.rand(Float64, basis.dim)

    @time begin
        for i in 1:nparas
            cu_funcs.expm(idxs[i], amps[i], lv)
        end
        sync_device!() 
    end
end
