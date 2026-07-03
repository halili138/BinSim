# Cleanup SCI printf output

## Problem
Three separate `@printf` lines print confusing cartesian product numbers (`Expand X→Y`, `merged=M`) mixed with the meaningful `new_sel` and `E` values.

## Changes (3 edits in `jl/sci_bitstr.jl`)

### Edit 1: Remove `Expand ... new_sel ...` print (line 499)

```
# BEFORE:
        nsel = length(sel_v)
        verbose && @printf("[%d] Expand %d→%d  new_sel=%d  ", iter, basis.dim, tgt.dim, nsel)

        if nsel == 0

# AFTER:
        nsel = length(sel_v)

        if nsel == 0
```

### Edit 2: Remove `merged=...` print (line 510)

```
# BEFORE:
        new_basis = SciBasisManagerBitstr(new_a, new_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
        verbose && @printf("merged=%d  ", new_basis.dim)

# AFTER:
        new_basis = SciBasisManagerBitstr(new_a, new_b, mole.norb, total_sym, mole.orbsym, na, nb; sorted=true, num_irreps=num_irreps)
```

### Edit 3: Combine `E=... err=...` into one line with new_sel and dim (line 526)

```
# BEFORE:
        verbose && @printf("E=%.10f  err=%.1e\n", E, abs(E - mole.e_scale))

# AFTER:
        verbose && @printf("[%d] new_sel=%d  dim=%d  E=%.10f  err=%.1e\n",
            iter, nsel, new_basis.dim, E, abs(E - mole.e_scale))
```

## Output format after change

```
[1] new_sel=1227  dim=32042  E=-109.0928342043  err=1.3e-02
[2] new_sel=85267  dim=1593217  E=-109.1052789024  err=9.5e-05
[3] new_sel=16034  dim=2378281  E=-109.1053558474  err=1.8e-05
[4] new_sel=84     dim=2407996  E=-109.1053562517  err=1.8e-05
[5] No new states, done.
```

| Field | Meaning |
|-------|---------|
| `new_sel=N` | Newly selected determinant pairs this iteration |
| `dim=D` | Working basis dimension (Cartesian product of string union, used for Davidson) |
| `E=...` | Variational energy |
| `err=...` | |E - reference|
