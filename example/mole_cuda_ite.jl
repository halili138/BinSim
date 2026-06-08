include("../binsim.jl")
using CUDA

const LIB_CUOTF = joinpath(@__DIR__, "../src/lib/libcuotf.so")

mutable struct CuBasisManager
    ptr::Ptr{Cvoid}

    function CuBasisManager(basis::BasisManager)
        ptr = @ccall LIB_CUOTF.build_basisdev_f64(
            basis.ptr::Ptr{Cvoid},
        )::Ptr{Cvoid}

        ptr == C_NULL && error("Failed to create C++ CU_BASIS.")

        obj = new(ptr)

        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_CUOTF.destroy_basisdev_f64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end
end

mutable struct CuOTF
    ptr::Ptr{Cvoid}

    function CuOTF(otf::OTF)
        ptr = @ccall LIB_CUOTF.build_networkdev_f64(
            otf.ptr::Ptr{Cvoid},
        )::Ptr{Cvoid}

        ptr == C_NULL && error("Failed to create C++ CU_OTF_NET.")

        obj = new(ptr)

        finalizer(obj) do o
            if o.ptr != C_NULL
                @ccall LIB_CUOTF.destroy_networkdev_f64(o.ptr::Ptr{Cvoid})::Cvoid
                o.ptr = C_NULL
            end
        end

        return obj
    end

end

function hvec_otf!(basis::CuBasisManager, otf::CuOTF, src::T, dst::T) where {T<:AbstractArray{Float64,1}}
    @ccall LIB_CUOTF.hvec_cuda(
        basis.ptr::Ptr{Cvoid},
        otf.ptr::Ptr{Cvoid},
        src::CuPtr{Cdouble},
        dst::CuPtr{Cdouble},
    )::Cvoid
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
            hvec_otf!(cubasis, cuotf, v, w)
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
    ham = JW_hamiltonian(mole)
    # mole.e_scale, _ = run_fci(basis, ham, get_hf(basis, mole.nelec))
    run_euler_ite_cuda(basis, ham, get_hf(basis, mole.nelec), mole.e_scale,
        dτ=0.1, max_step=1000, tol=1e-8)
end
