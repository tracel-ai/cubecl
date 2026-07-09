# Two CPU shared-memory bugs in cubecl

cubecl rev `d9960a85e1d6b57a9ae11ca9cd18a2246afb0530`. Self-contained (only depends
on `cubecl` with the `cpu` feature). Two separate, minimal repros — one per bug.

## `cargo run --release --bin current_shared_sum` — the *current* problem

Surfaced by updating cubek to the latest cubecl rev. The CPU backend rejects any
launch with `cube_dim.num_elems() > max_units_per_cube` (= CPU thread count) but
**swallows the error** — the kernel never runs and the output is left untouched.
A legal cube writes 42; a cube one unit larger returns the untouched sentinel with
no error reported.

Hits cubek's `shared_sum` (cube_dim 32×8 = 256 units) → `reduce_shared::test_shared_sum`
returns 0.

Backend: `cubecl-cpu/src/compute/stream.rs` (the `MaxUnitPerCube` early return).

## `cargo run --release --bin old_interpolate` — the *old* interpolate problem

Pre-existing; the interpolate crate already works around it by refusing the
shared-memory strategy on CPU (`cubek-interpolate/src/lib.rs`). When a launch
dispatches **more than one cube**, all cubes share a single shared-memory buffer.
`sync_cube` only synchronizes within a cube, so concurrent cubes race and corrupt
each other. Each cube fills its own buffer with a cube-specific value and sums it;
the sums come back wrong and **change from run to run**. Single-cube shared memory
is correct — the bug needs `cube_count > 1`.

Hits cubek's interpolate `*_shared_memory_*` tests (each dispatches ≥ 2 cubes, one
per tile/batch).

Backend: `cubecl-cpu/src/compiler/mlir_data.rs` `MlirData::new` allocates the shared
buffer once per launch into one `Arc<SharedMlirData>` that the threadpool clones to
every unit of every cube — there is no per-cube allocation.

## Are they the same bug?

No.

| | `current_shared_sum` | `old_interpolate` |
|---|---|---|
| Trigger | cube_dim > max_units_per_cube | legal cube, but cube_count > 1 |
| Failure | launch silently dropped, whole output = 0 | launch runs, cubes race on one buffer |
| Determinism | deterministic (always 0) | nondeterministic, partial corruption |
| Backend site | `stream.rs` (swallowed `MaxUnitPerCube`) | `mlir_data.rs` (shared buffer not per-cube) |
