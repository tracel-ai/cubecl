# `feat/environment` review fixes

Tracking file for the code review of this branch. Delete before merge.

Status: `[ ]` todo, `[~]` in progress, `[x]` done, `[-]` deferred (with reason).

## Blockers

- [x] **B1** `cubecl-runtime/src/config/autotune.rs:1`, `compilation.rs:1` — orphaned `#[cfg(std_io)]` now gates the logger import, breaking `--no-default-features`.
- [x] **B2** `cubecl-runtime/src/config/environment.rs:3,27` — ungated `use super::cache::CacheConfig` and `path` field against a `#[cfg(std_io)]` module.
- [x] **B3** `cubecl-environment` — `cache` feature activates `dep:rusqlite`, declared only for non-wasm, while `pub mod sqlite` is gated on the bare feature. Needs a `native_cache` build.rs alias.

## Correctness

- [x] **C1** `bundle/export.rs:58,240` — `prepare_output` validates any existing output as SQLite, so re-exporting a flat bundle over itself always fails.
- [x] **C2** `persistence/sqlite.rs:151` — exports are WAL-mode, so a bundle installed in a read-only directory silently imports zero entries. Fixed on both sides: read-only opens fall back to the `immutable=1` URI, and `export_sqlite` calls the new `Database::finalize_for_shipping` (`journal_mode=DELETE`) before publishing.
- [x] **C3** `persistence/store.rs:247` — an undecodable stored value is reported `Stored` and can never be replaced; permanent recompile for `BlobStore` keys.
- [x] **C4** `environment.rs:94` — `path()` takes two independent snapshots of name and root, tearing across a concurrent reconfigure.
- [x] **C5** `config/mod.rs:181` — `panic!` on a TOML parse error; `CompilationConfig::cache` was retyped, so a stale `cubecl.toml` aborts the process.
- [x] **C6** `bundle/export.rs:191` — `ATTACH DATABASE` gets a lossily-converted path; on non-UTF-8 paths SQLite creates an empty db and the export silently copies nothing.
- [x] **C7** `cubecl-cpu/src/runtime.rs:20,43` — mixes `std::sync::Arc` and `cubecl_environment::sync::Arc`.
- [x] **C8** `cubecl-hip/src/compute/context.rs:71` — arch fingerprint strips the target-feature suffix; `cubecl-metal/src/compute/context.rs:65` has no fingerprint at all.

## Elegance

- [x] **E1** `sync/base.rs:52,91,110` — `lock()`/`read()`/`write()` return `Result<_, String>` documented as always `Ok`. Return the guard; delete ~43 `.unwrap()`/`.expect()` sites.
- [x] **E2** `environment.rs:75-161` — `set_root`/`root`/`path`/`file_name` gated on the `cache` feature, so an unrelated public API appears and disappears via feature unification.
- [x] **E3** `persistence/storage.rs:49` — `Storage::insert` has no error channel, but `KvStore::insert` returns `Result`. Three-state result.
- [x] **E4** `persistence/storage.rs:49` — `insert` takes `&[u8]` although callers hold `Bytes`; memory backends copy twice.
- [x] **E5** Backend `Cargo.toml`s — five different declaration shapes for `cubecl-environment`; Metal enables `cache` unconditionally, bypassing the platform gate.
- [x] **E7** `tune/tune_cache.rs:230` — `sync_persistent` is a permanent no-op on native; comments claim the opposite.
- [x] **E8** `sync/base.rs:131` — `SyncOnceCell` forces `T: Debug` on its whole API for one internal `.unwrap()`.
- [x] **E9** `bundle/import.rs:41` — one immediate transaction per entry. Add `Storage::insert_many`.
- [x] **E10** `bundle/embedded.rs:279` — `namespace(index)` walks from the start each call; `namespaces()`/`summary()`/`import` are quadratic.
- [x] **E12** `bundle/mod.rs:50` — `manifest` is gated behind `cache`, the very feature the flat format exists to avoid, so wasm/no-std gets opaque metadata bytes with no type to parse them into and none of the schema guards.
- [x] **E11** Dead code sweep: `cubecl-macros` env dep, `thread.rs` `ThreadId`, `stream/id.rs` `swap`, `hashbrown` in cubecl-runtime, `cfg-if` in cubecl-common, non-dev `serde_json`, ~~`sqlite.rs:228` `let _ = existing;`~~ (done), `config/mod.rs:3` stray doc line.

## Smaller

- [x] **S1** `bundle/flat.rs:110` — 4 GiB overflow reported as `InvalidManifest` with wrong advice. Add `BundleError::TooLarge`.
- [x] **S2** `xtask/src/commands/bundle.rs:195,236` — swallows the real bundle error. Sniff `MAGIC` to pick the format.
- [x] **S3** `bundle/embedded.rs:129` — `validate_entries` never checks the index is sorted, the one invariant lookups rely on.
- [x] **S4** `bundle/embedded.rs:83` — doc example calls `bundle::install`, which does not exist.
- [x] **S5** `persistence/store.rs:53` — `StoreError` has no `Display`/`Error` impl and carries unused `Serialize` bounds.
- [x] **S6** `persistence/storage.rs:51` — `scan` runs the visitor under the connection mutex; document the re-entrancy ban.
- [x] **S7** `tune/tune_cache.rs:272` — `KeyOutOfSync` folded into `DuplicatedKey`, so a benign multi-process race warns with full payloads.
- [x] **S8** `future/channel.rs:10` — `pub use oneshot;` with no wasm guard, unlike `block_on`/`read_sync`.
- [x] **S9** `persistence/root.rs:28,59` — `unwrap`/`expect` next to a `Target` arm hardened against the same failures.
- [x] **S10** `bundle/export.rs:143,224` — the prefix predicate written twice; `""` means everything for flat and nothing for SQLite.
- [x] **S11** `bundle/export.rs` — a failed export leaves a file that wedges every later export. Write to `.tmp` and rename.
- [x] **S12** `cubecl-cuda`/`cubecl-hip` `compute/context.rs` — still `std::collections::HashMap` for the hot `module_names` lookup.
- [x] **S13** `cubecl-wgpu/src/compute/poll.rs` — half-migrated to the thread shim, which adds no portability here.
- [x] **S14** `persistence/store.rs:164` — `get` cannot ingest an async load, forcing the `sync_persistent` workaround on callers.

## Tests

- [x] **T1** flat re-export over an existing flat bundle.
- [x] **T2** `EmbeddedBundle::from_static`.
- [x] **T3** multi-root dedup and namespace filtering for `BundleFormat::Flat`.
- [x] **T4** opening a bundle from a read-only directory.

## Deferred

- [-] **E1b** Make poison recovery opt-in rather than the only behaviour (`sync/base.rs`). Real concern for memory-pool state, but it is a separate design call with workspace-wide reach. Not folded into this branch.
- [-] **E6** `cubecl-common` is now vestigial and depends on the layer below it. Dissolving it into `cubecl-environment` is a follow-up restructure, not a review fix.
