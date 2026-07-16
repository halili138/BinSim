# SCI Nosym — Full Implementation Plan

## Files to create

1. `src/include/sci_basis_nosym.hpp` — SciBasisManagerNosym + create/destroy/remap/get_diags
2. `src/include/sci_select_nosym.hpp` — ForwardSharedNosym + select + contract_hvec
3. `src/extern/sci_otf_nosym.cpp` — C wrappers
4. `jl/sci_nosym.jl` — Julia bindings + run_sci_nosym
5. `example/test_sci_nosym.jl` — test
6. Update `src/Makefile` — add libsci_otf_nosym.so target
7. Update `jl/binsim.jl` — add LIB_SCI_NOSYM path

## Key differences from symm version

| Aspect | Symm (`sci_select.hpp`) | Nosym |
|--------|------------------------|-------|
| ForwardShared | 2D `(group, block)` CSR | 1D `(group)` CSR |
| build_idx_map | `str → (local, blk)` | `str → idx` |
| precompute_shared_chunk | buckets `(g, blk)` | buckets `(g)` only |
| psi gid | `blk.offset + a_local*blk.num_b + b_local` | `a*num_b + b` |
| same-block guard | `src_a_blk == src_b_blk` | deleted |
| select_pass params | `src_blocks`, `old_X_num_blocks` | `num_b` only |
| contract_hvec | per-block gather_* | flat gather_*_nosym |
| remap_wavefunction | sym lookup + block offset | flat lookup `a*num_b+b` |
| get_diags | per-block double for | flat double for |

## sci_basis_nosym.hpp structure

```
SciBasisManagerNosym<Ti>:
  Ti *all_astrs, *all_bstrs
  int64 num_a, num_b, dim, norb
  map<Ti,int> a_idx_map, b_idx_map   // str → global position

create_sci_basis_manager_nosym(astrs, na, bstrs, nb, norb)
destroy_sci_basis_manager_nosym(basis)
remap_wavefunction_nosym(old, old_psi, new, new_psi)  // flat lookup
get_diags_elements_sci_nosym(basis, net, diags)       // flat double for
```

## sci_select_nosym.hpp structure

```
sqnorm, sci_eps_check              (copied)
precompute_phase_select            (copied)

ForwardSharedNosym<Tv>:
  int64 ngs
  vector<int64> offsets[ngs+1]      // per-group
  vector<int> dst_idxs              // flat
  vector<int> src_idxs              // flat (no src_blk_idxs!)
  vector<Tv> phase0, phase1         // flat
  range(int64 group) const -> pair<int64,int64>

build_idx_map_nosym(basis, IsAlpha) -> map<Ti,int>  // str → idx only
build_old2new_link_chunk           (unchanged)
precompute_shared_chunk_nosym      (simplified: no block bucketing)
precompute_diag_phases             (unchanged)

select_pass_a_nosym(new_α, n_new_α, old_β, n_old_β, new_β, n_new_β,
                    old_a_idx_map, old_b_idx_map, num_b,
                    all_groups, src_psi,
                    pa_diag, pb_diag_old, pb_diag_new, diag_rank,
                    E_var, eps, out_p1, out_p3)

select_pass_b_nosym(new_β, n_new_β, old_α, n_old_α,
                    old_b_idx_map, old_a_idx_map, num_b,
                    all_groups, src_psi,
                    pb_diag, pa_diag_old, diag_rank,
                    E_var, eps, out_p2)

// Contract infrastructure (flat, no blocks)
gather_diag_nosym, gather_pure_a_nosym,
gather_pure_b_nosym, gather_mixed_nosym
dispatch_contract_chunks_nosym
contract_hvec_sci_nosym
```

## SciBasisManagerNosym interface for contract_hvec_sci_nosym

The gather_*_nosym functions take flat arrays + idx_maps directly, not a SciBasisManager pointer. This avoids the need for a compatibility layer.

gather_pure_a_nosym signature:
```
(astrs, num_a, bstrs, num_b, a_idx_map, b_idx_map,
 groups, num_groups, src_vec, dst_vec)
```

## C wrapper (sci_otf_nosym.cpp)

Same pattern as sci_otf.cpp, but function names changed:
- sci_select_nosym_f64 → calls select_pass_a_nosym/b_nosym
- remap_wavefunction_sci_nosym_f64 → calls remap_wavefunction_nosym
- get_diags_elements_sci_nosym_f64 → calls get_diags_elements_sci_nosym
- hvec_sci_nosym_f64 → calls contract_hvec_sci_nosym
- create/destroy_sci_basis_manager_nosym_f64

## Julia (jl/sci_nosym.jl)

SciBasisManagerNosym struct + constructor
sci_select_nosym! wrapper
run_sci_nosym(mole; max_iter, eps, verbose) pipeline

## Test (example/test_sci_nosym.jl)

Same as test_sci_bitstr.jl, calls run_sci_nosym instead.

## ~ lines per file

sci_basis_nosym.hpp:   ~140
sci_select_nosym.hpp:  ~520
sci_otf_nosym.cpp:     ~120
jl/sci_nosym.jl:       ~140
test_sci_nosym.jl:     ~20
