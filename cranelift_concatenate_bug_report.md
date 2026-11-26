# Cranelift backend panics on `lower_concatenate_nd` for multi-entity (`el.six_dof`) worlds

**SDK version:** v0.20.0
**Platform:** WSL2 Ubuntu 24.04 (Windows 11 host)
**Example:** a custom multi-drone extension of a six-DOF quadrotor sim (search-and-rescue style), structurally similar to `examples/n-body` in that it spawns N `el.Body` entities sharing one composed system, but — unlike n-body — the system is a longer pipeline: several `@el.map` guidance/control stages piped into `el.six_dof(..., integrator=el.Integrator.SemiImplicit) | sensors_imu | telemetry`.

## Summary

Compiling a world with more than one entity sharing this kind of system (six_dof + a downstream sensor/telemetry stage) panics inside the Cranelift backend's `lower_concatenate_nd`, with a plain `N_DRONES=1` vs `N_DRONES=3` toggle in the Python repro as the only difference between a clean compile and a hard crash.

## Repro

```bash
N_DRONES=1 elodin run main.py   # compiles and runs fine
N_DRONES=3 elodin run main.py   # panics during compile, before the sim ever ticks
```

World-building succeeds completely in both cases (all entities/components/cameras register, confirmed via the `convert name "..." to ID "..."` log lines) — the panic happens later, when `world.run(system(), ...)` triggers the actual Cranelift compile.

## Observed panic

```
sim [elodin-cranelift] fold: 10172 -> 9348 instr (-824, -8.1%)
sim
sim thread '<unnamed>' (3171) panicked at libs/cranelift-mlir/src/lower.rs:9268:38:
sim index out of bounds: the len is 1 but the index is 1
sim note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
sim Traceback (most recent call last):
sim   File ".../main.py", line 411, in <module>
sim     world.run(system(), ...)
sim   File ".../site-packages/elodin/__init__.py", line 702, in run
sim     addr = super().run(system, ...)
sim RuntimeError: cranelift backend: Cranelift compile failed: panic compiling inner_295: index out of bounds: the len is 1 but the index is 1
Error:   x failed to build sim Some(1)
```

## Root cause (traced against the `v0.20.0` tag)

`libs/cranelift-mlir/src/lower.rs`, `lower_concatenate_nd`:

```rust
fn lower_concatenate_nd(
    parts: &[(Vec<Value>, TensorType)],
    dim: usize,
    out_ty: &TensorType,
) -> TensorVals {
    ...
    for flat_out in 0..n {
        ...
        let local_dim_idx = concat_idx - offset_in_dim;
        let (ref _shape, ref strides) = part_infos[part_idx];
        let mut src_flat = 0;
        for d in 0..rank {
            let idx = if d == dim {
                local_dim_idx
            } else {
                out_indices[d]   // <-- uses the OUTPUT's index for every non-concat dim
            };
            src_flat += idx * strides[d];
        }

        result.push(parts[part_idx].0[src_flat]);   // <-- panics here (line 9268, col 38)
    }
    result
}
```

For every dimension other than the concat dimension, this reuses `out_indices[d]` — the index into the *output* tensor — directly against each operand's own strides, with no check that a given operand was actually materialized at the output's full extent along that dimension. That's only sound if every operand already spans the output's full size in every non-concat dimension.

The panic message (`len is 1 but index is 1`) is consistent with one concatenated operand having stayed at size 1 along a dimension where the *output* has size > 1 (e.g., a per-entity batch dimension of 3) — i.e. an operand that should have been broadcast out to match the batch dimension before reaching this concatenate, but wasn't. `src_flat` ends up computed as if that operand already had 3 elements along that axis, and indexing element 1 of a 1-element buffer is exactly this crash.

This is consistent with why `N_DRONES=1` never hits it: with a single entity there's no real multi-element batch dimension for anything to under-broadcast against, so this code path is either not exercised or degenerate (effectively always index 0). With `N_DRONES=3`, something in the compiled six_dof+sensor+telemetry chain produces a concatenate where one operand didn't get broadcast across that batch dimension first.

### Ruled out

We initially suspected this was caused by our own Python-side system composition — specifically a helper that piped one `el.System` object into itself 3 times (`out_sys = out_sys | system`, looped) to build a 3x inner fast-loop. We rewrote it to build three fully independent `System` objects instead (fresh `el.six_dof(...)`, fresh effectors, fresh everything, no object reuse at all) and the exact same panic recurred, verbatim, down to the same symbol name and byte offset. That rules out Python-level object-identity/self-composition as the cause — this is purely a backend lowering issue triggered by entity count, not by how the system pipeline is built in Python.

We also confirmed `examples/n-body` (which spawns multiple `el.Body` entities and runs `el.six_dof` across all of them) compiles and runs fine — so multi-entity `six_dof` is not broken in general. The difference is n-body's system is a single simple composition (`el.six_dof(sys=gravity) | truth_track`) with no downstream sensor/telemetry stage; ours chains six_dof into additional `@el.map` stages, which is apparently what produces the problematic concatenate.

## Update: checked the actual dumped IR, ruled out a StableHLO-level shape bug

Setting `ELODIN_CRANELIFT_DEBUG_DIR=/tmp/elodin_debug` before the failing run (per `debug.rs`'s own doc comment — this gates every diagnostic dump in the crate off one env var, with zero effect on the emitted machine code when unset) dumps the exact `stablehlo.mlir` being compiled before the panic. We wrote a small script (attached, `find_bad_concat.py`) to parse every `stablehlo.concatenate` in that dump and check operand rank, operand shape vs. output shape in every non-concat dimension, and that concat-dim sizes sum correctly. Result:

```
scanned 316 concatenate ops in /tmp/elodin_debug/stablehlo.mlir (0 lines failed to parse)
found 0 with shape inconsistencies
```

So the StableHLO IR itself is fully well-typed — every one of the 316 concatenate ops in the compiled module has consistent, correctly-shaped operands (the entity-batch dimension is already `3` everywhere it should be, e.g. `tensor<3x1xf64>` operands concatenating along `dim=1` into `tensor<3x4xf64>`). This rules out a naive "forgot to broadcast before concatenating" bug at the Python/JAX/StableHLO level entirely.

That means the length-1-vs-index-1 mismatch is introduced purely inside `cranelift-mlir`'s own IR→Cranelift value materialization, for at least one operand feeding some concatenate: its declared `TensorType` is correct (consistent with everything above), but the actual `Vec<Value>` buffer backing it at the point `lower_concatenate_nd` reads `parts[part_idx].0` has fewer entries than `ty.num_elements()` implies. Two places in `lower.rs`/`const_fold.rs` look like plausible sites for this kind of value/type-count divergence, though we were not able to fully confirm either without rebuilding the compiler with added instrumentation:

- `const_fold.rs`'s `try_fold` for `Instruction::BroadcastInDim` folds a scalar/splat source into `ConstantValue::DenseSplat(v, result_ty)` (the `"broadcast_scalar"` rule — the fold log for this exact failing compile shows `broadcast_scalar=507`, i.e. 507 such folds happened in this one compile). Any later consumer that materializes a `DenseSplat` needs to always produce `result_ty.num_elements()` actual values, not just however many are convenient.
- `lower.rs`'s `LaneRepr::PackedF64` (SIMD-packed F64X2 chunks + optional odd-`n` tail) and its unpacking via `get_vals()`/`unpack_in()`. We traced through `unpack_in`'s logic by hand for `n=3` (matching our 3-drone batch dimension) and it appears to reconstruct the full 3-element array correctly (`chunks.len()*2` + tail, then `truncate(n)`), so this specific path looks sound on inspection — but it's the other main place in this file where a tensor's "value count" and "declared element count" are tracked somewhat independently, and is worth a closer look by someone with the full picture of how `LaneRepr` variants flow through this file.

## Suggested next step for a fix

The cheapest way to pin this down exactly: add a debug assertion at the top of the `flat_out` loop body in `lower_concatenate_nd` (`lower.rs:9210`) checking `parts[part_idx].0.len() == part_infos[part_idx].0.iter().product::<usize>()` (i.e. the part's actual value count vs. what its own recorded shape implies), with an `eprintln!` of `part_idx` and both lengths on mismatch. That would immediately identify which upstream instruction produced the short buffer, rather than failing downstream at an opaque flat index. We don't have a Rust build of the compiler set up to add this ourselves, but it should be a five-minute change for someone who does.

## Repro artifacts

Both scripts used above are available on request:
- `find_bad_concat.py` — scans a dumped `stablehlo.mlir` for operand/shape inconsistencies in `stablehlo.concatenate` ops (found none in this failing compile, which is itself useful signal — see above).
- The `sim.py` rewrite that ruled out the self-composition hypothesis (three independently-constructed `el.six_dof | sensors.imu | telemetry.telemetry` instances chained instead of one instance self-piped three times via a loop) — same panic, same symbol name, same byte offset, with zero object reuse anywhere in the chain.
