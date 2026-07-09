ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "2")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include("../jl/binsim.jl")

using Random
using Test

function _pair_index0(p::Int, q::Int)
    a = max(p, q)
    b = min(p, q)
    return a * (a + 1) ÷ 2 + b
end

function _random_real_molecular_integrals(norb::Int; seed::Int=20260709)
    rng = MersenneTwister(seed + norb)

    one_body = randn(rng, norb, norb)
    one_body = 0.5 .* (one_body .+ transpose(one_body))

    two_body = zeros(Float64, norb, norb, norb, norb)
    for p in 0:norb-1, q in 0:norb-1, r in 0:norb-1, s in 0:norb-1
        _pair_index0(p, q) <= _pair_index0(r, s) || continue
        value = randn(rng)
        for (pp, qq, rr, ss) in (
            (p, q, r, s),
            (q, p, s, r),
            (r, s, p, q),
            (s, r, q, p),
        )
            two_body[pp+1, qq+1, rr+1, ss+1] = value
        end
    end

    return one_body, two_body
end

function _canonical_hamiltonian(H::BinaryQubitAABB)
    idx = sortperm(collect(zip(H.axs, H.bxs, H.azs, H.bzs)))
    return H.axs[idx], H.bxs[idx], H.azs[idx], H.bzs[idx], H.cs[idx]
end

@testset "real molecular Hamiltonian path" begin
    for norb in 1:4
        one_body, two_body = _random_real_molecular_integrals(norb)
        mole = Mole()
        mole.norb = norb
        mole.energy_nuc = 0.125
        mole.one_body_mo = one_body
        mole.two_body_mo = two_body

        Hgeneric = _canonical_hamiltonian(JW_hamiltonian(mole; spin="aabb"))
        Hreal = _canonical_hamiltonian(JW_hamiltonian_real(mole; spin="aabb"))

        @test Hreal[1] == Hgeneric[1]
        @test Hreal[2] == Hgeneric[2]
        @test Hreal[3] == Hgeneric[3]
        @test Hreal[4] == Hgeneric[4]
        @test Hreal[5] ≈ Hgeneric[5] atol=1e-8 rtol=1e-8
    end
end
