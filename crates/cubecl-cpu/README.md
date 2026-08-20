# CPU Runtime

CPU runtime for CubeCL. It executes kernels in-process on a worker thread pool,
using SIMD where the host CPU supports it. Compilation is delegated to
[`cubecl-llvm`](../cubecl-llvm), which JIT-compiles each kernel through LLVM;
this crate owns the memory management, scheduling and execution around it.

## Setup

Add `cubecl` with the `cpu` feature:

```toml
[dependencies]
cubecl = { version = "*", features = ["cpu"] }
```

## Layout

| Module                | What it does                                                        |
| --------------------- | ------------------------------------------------------------------- |
| `compute::server`     | The `ComputeServer` impl: allocation, launches and reads            |
| `compute::threadpool` | Worker threads, and the per-unit dispatch of a launch               |
| `compute::affinity`   | Core topology and thread pinning, per platform                      |
| `runtime`             | `CpuRuntime`, device properties and the supported type/atomic table |

## Tuning

| Variable                | Effect                                                           |
| ----------------------- | ---------------------------------------------------------------- |
| `CUBECL_CPU_STACK_SIZE` | Worker thread stack size, in bytes                               |
| `CUBECL_CPU_STACK_MB`   | Same, in MiB; used only when `CUBECL_CPU_STACK_SIZE` is unset    |

This is the stack of the worker threads the JIT'd kernels run on, so a kernel's
own stack frame comes out of it. Both variables are floored at 16 MiB, and the
default is 64 MiB.

## Debugging

To dump the IR a kernel goes through on its way to machine code, see
[Debugging the compiler](../cubecl-llvm/README.md#debugging-the-compiler) in
`cubecl-llvm`.

## Troubleshooting

- **Segfaults during execution.** Kernel invocation is `unsafe`, so a bad
  pointer or shape segfaults rather than returning an error. Dump the IR and
  inspect the per-pass `.plir` files to find where the generated code diverges
  from what you expect.
- **Stack overflow in a worker.** Raise `CUBECL_CPU_STACK_MB`. Shared memory is
  reserved from the memory pool rather than the stack, so this points at a large
  kernel stack frame.
