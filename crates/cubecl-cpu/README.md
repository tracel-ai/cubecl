# CPU Runtime

Pliron-based JIT runtime for CubeCL. It lowers CubeCL IR to LLVM, and executes the result in-process through `LLJIT`, using SIMD where the host CPU supports it. LLVM is vendored through `tracel-llvm-bundler`, so there is no system LLVM to install.

## Setup

Add `cubecl` with the `cpu` feature:

```toml
[dependencies]
cubecl = { version = "*", features = ["cpu"] }
```

## Debugging the compiler

When a kernel miscompiles or you are modifying the lowering passes, you can dump every intermediate representation the compiler goes through.

Setting `CUBECL_DEBUG_PLIRON` auto-enables the `pliron-dump` feature at build time
(see `build.rs`), so a single command is enough — no extra `--features` flag:

```bash
CUBECL_DEBUG_PLIRON=./debug cargo test -p cubecl-cpu
```

Use any binary, test, or example that launches a CPU kernel. Each kernel writes
a subfolder named after its kernel name under the directory you chose. The
following artifacts are produced per kernel:

| File                     | What it is                       | How to inspect                          |
| ------------------------ | -------------------------------- | --------------------------------------- |
| `cubecl.ir.txt`          | Raw CubeCL IR                    | text                                    |
| `cubecl-opt.ir.txt`      | IR after the optimization passes | text                                    |
| `cubecl-opt.ir.dot`      | Control-flow graph of the kernel | `dot -Tsvg cubecl-opt.ir.dot > cfg.svg` |
| `<module>/N_<pass>.mlir` | MLIR after each lowering pass    | text trace where lowering diverges      |
| `mlir_output.so`         | Final compiled shared object     | `llvm-objdump -d mlir_output.so`        |

To enable the dump feature explicitly without the env var (for example from
another crate that depends on `cubecl-cpu`):

```bash
cargo run --features cpu,cubecl-cpu/mlir-dump
```

## Troubleshooting

- **Segfaults during execution.** Kernel invocation is `unsafe` and a bad pointer or shape will
  segfault rather than return an error. Dump the IR and inspect the per-pass
  `.plir` files to find where the generated code diverges from what you expect.
