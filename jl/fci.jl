function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}, v0::Vector{Tv}) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=true)
    diags = zero(v0)
    funcs.get_diags(diags)

    println("Solving FCI with davidson ... ")
    @time e_fci, v_fci = davidson(funcs.hvec, v0, diags, tol=1e-5)
    println("")

    return e_fci, v_fci
end


function run_fci(basis::BasisManager, ham::BinaryQubitAABB{Ti,Tv,K,V}; k::Int=1) where {Ti,Tv,K,V}
    funcs = OTF_Functions(basis, ham, typeof(ham)[], time_print=false)
    hvec_map = LinearMap{Tv}(
        (dst, src) -> funcs.hvec(src, dst),
        basis.dim,
        ismutating=true,
        ishermitian=true
    )

    print("Running Arpack directly on C++ OTF Network...")
    time_ops = @elapsed λ_aggs, ϕ_aggs = eigs(hvec_map, nev=k, which=:SR)
    @printf("Done in %.4f seconds\n", time_ops)

    λ_aggs = real.(λ_aggs)
    df_states = [i == 1 ? "000 (GS)" : @sprintf("%03d", i - 1) for i in 1:k]
    df_energies = [@sprintf("%.14f", e) for e in λ_aggs]

    df_step = DataFrame(
        "State" => df_states,
        "f (Energy)" => df_energies,
    )

    println("-------------------------------")
    show(stdout, df_step, summary=false, eltypes=false, show_row_number=false)
    println("\n")

    return λ_aggs, ϕ_aggs
end
