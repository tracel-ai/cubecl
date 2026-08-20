# LLVM Compiler

Pliron-based LLVM compiler for CubeCL. It takes a `KernelDefinition`, runs it
through the lowering and optimization passes down to the LLVM dialect, converts
that to LLVM IR, and JIT-compiles the result with ORC/LLJIT. The compiled kernel
is handed back as a `PlironEngine`, which a runtime calls into.

LLVM is vendored through `tracel-llvm-bundler` by this crate's `build.rs`, so
there is no system LLVM to install.

## Layout

| Module             | What it does                                                                  |
| ------------------ | ----------------------------------------------------------------------------- |
| `shared`           | `PlironCompiler`, the `Compiler` impl and the pass pipeline                   |
| `shared::to_llvm`  | Lowering of each CubeCL operation to the LLVM dialect                         |
| `shared::polyfill` | Operations with no direct LLVM equivalent: math, ordered atomics, `sync_cube` |
| `shared::metadata` | The entry-point ABI: buffer/metadata tables and shared-memory slots           |
| `shared::jit`      | LLVM IR conversion, the `default<O3>` pipeline and LLJIT                      |

## Debugging the compiler

When a kernel miscompiles or you are modifying the lowering passes, set
`CUBECL_DEBUG_PLIRON` to a directory to dump every intermediate representation
the compiler goes through. Each kernel writes a subfolder named after itself:

```bash
CUBECL_DEBUG_PLIRON=./debug cargo test -p cubecl-cpu
```

Use any binary, test, or example that launches a kernel. Per kernel you get:

| File                    | What it is                               | How to inspect                  |
| ----------------------- | ---------------------------------------- | ------------------------------- |
| `N-after-<pass>.plir`   | Pliron IR after each pass, in order      | text; diff consecutive files    |
| `llvm.ll`               | LLVM IR straight out of the conversion   | text                            |
| `llvm.opt.ll`           | LLVM IR after the `default<O3>` pipeline | text                            |

Diffing `N-after-<pass>.plir` against `N+1-after-<next>.plir` is usually the
fastest way to find the pass that broke a kernel. If both `.plir` and `llvm.ll`
look right, the bug is in LLVM.

### The `pliron-dump` feature

```bash
cargo test -p cubecl-cpu --features cubecl-llvm/pliron-dump
```

The feature is re-exported as `cubecl-cpu/pliron-dump` and `cubecl/pliron-dump`
for crates that depend on those instead.
