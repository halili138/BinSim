include("../binsim.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = ARGS[1]
    mole.ratio = 1.0
    mole.basis = ARGS[2]

    build(mole)

    basis = BasisManager(mole.norb, mole.nelec, mole.orbsym)
    psi_space = basis.dim * 8 / (1 << 30)
    @printf("Num symmetry allowed elements: %d    %.4f GB\n\n", basis.dim, psi_space)

    ham   = JW_hamiltonian(mole)
    hf    = get_hf(basis, mole.nelec)
    diags = get_diags(basis, ham)

    ret = @timed otf = OTF(basis, ham)
    println("Successifully Generate OTF in $(ret.time) seconds\n")

    aop! = (src, dst) -> begin
        cpu_start = CPUtime_us()
        t2 = @timed hvec_otf!(basis, otf, src, dst)
        cpu_time = (CPUtime_us() - cpu_start) / 1e6
        @printf("hvec-time process: %.4f  wall: %.4f seconds ", cpu_time, t2.time)
    end

    @time davidson(aop!, hf, diags)
end

