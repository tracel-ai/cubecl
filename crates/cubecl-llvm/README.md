# LLVM Compiler

Pliron-based LLVM compiler for CubeCL. It takes a `KernelDefinition`, runs it
through the lowering and optimization passes down to the LLVM dialect, and
converts that to LLVM IR. Where it goes from there is the target it was built
for:

| Target               | Produces                                                        |
| -------------------- | --------------------------------------------------------------- |
| `LlvmTarget::Cpu`    | a `PlironEngine`, JIT-compiled with ORC/LLJIT                    |
| `LlvmTarget::AmdGpu` | an `AmdGpuModule`, a linked code object for `hipModuleLoadData`  |

Either one arrives as a `PlironArtifact`, which a runtime calls into.

LLVM is vendored through `tracel-llvm-bundler` by this crate's `build.rs`, so
there is no system LLVM to install.

## CPU-only builds and the `amdgpu` feature

`amdgpu` controls native AMDGPU code generation, the C++ shims, and LLD linking.
It remains enabled by default for direct `cubecl-llvm` users. `cubecl-hip` enables
it explicitly; `cubecl-cpu` uses LLVM without its default features, including
when the CPU crate's own defaults are enabled. CPU builds therefore need only
the host LLVM target, such as AArch64 on Apple silicon.

For direct CPU-only use:

```toml
cubecl-llvm = { version = "=0.11.0-pre.3", default-features = false, features = ["std"] }
```

The public modules, `LlvmTarget` variants, `PlironOptions::arch`, `AmdGpuModule`,
and `PlironArtifact` variants remain available in either configuration. Existing
struct literals and exhaustive matches continue to compile. With `amdgpu`
disabled, requesting AMDGPU compilation or object linking returns an error
explaining which feature to enable, before changing IR or creating files.
Device-library linking remains a no-op when no libraries are needed. The legacy
unsafe `lower_printf_to_hostcall` helper returns a boolean, so it cannot report
an error: without the feature, an actual printf rewrite panics before modifying
the module; modules without printf calls remain a no-op.

Cargo combines dependency features. A program that also enables HIP or depends
on `cubecl-llvm` with its defaults needs an LLVM bundle containing AMDGPU as well
as the host target. Disabling `amdgpu` changes native AMDGPU availability for
existing direct users of `default-features = false`; those users must opt in if
they need AMDGPU code generation.

## Layout

| Module             | What it does                                                                  |
| ------------------ | ----------------------------------------------------------------------------- |
| `shared`           | `PlironCompiler`, the `Compiler` impl and the pass pipeline                   |
| `shared::to_llvm`  | Lowering of each CubeCL operation to the LLVM dialect                         |
| `shared::polyfill` | Operations with no direct LLVM equivalent: math, complex, `sync_cube`         |
| `shared::metadata` | The entry-point ABI: the metadata table and the target's argument layout      |
| `cpu`              | The CPU target: grid emulation, the pointer-table ABI, shared memory          |
| `cpu::jit`         | LLVM IR conversion, the `default<O3>` pipeline and LLJIT                      |
| `amdgpu`           | The AMDGPU target: hardware builtins, plane, matrix, LDS, the kernarg ABI     |
| `amdgpu::codegen`  | LLVM IR to an AMD code object, through the AMDGPU backend and LLD             |

The AMDGPU target reaches three parts of LLVM that have no C API: LLD's ELF
driver, the bitcode linker's `--only-needed` mode, and the AMDGPU `printf`
emitter. With `amdgpu` enabled, `build.rs` compiles `amdgpu/cpp_shims/` alongside the crate to wrap
them.

## Debugging the compiler

When a kernel miscompiles or you are modifying the lowering passes, set
`CUBECL_DEBUG_PLIRON` to a directory to dump every intermediate representation
the compiler goes through. Each kernel writes a subfolder named after itself:

```bash
CUBECL_DEBUG_PLIRON=./debug cargo test -p cubecl-cpu
```

Use any binary, test, or example that launches a kernel. Per kernel you get:

| File                    | What it is                                | Target |
| ----------------------- | ----------------------------------------- | ------ |
| `N-after-<pass>.plir`   | Pliron IR after each pass, in order       | both   |
| `llvm.ll`               | LLVM IR straight out of the conversion    | CPU    |
| `llvm.opt.ll`           | LLVM IR after the `default<O3>` pipeline  | CPU    |
| `amdgpu.ll`             | LLVM IR stamped for the HSA target        | AMDGPU |
| `amdgpu.s`              | AMDGPU assembly, HSA metadata included    | AMDGPU |

Diffing `N-after-<pass>.plir` against `N+1-after-<next>.plir` is usually the
fastest way to find the pass that broke a kernel. If both the `.plir` and the
LLVM IR look right, the bug is in LLVM. On AMDGPU, `amdgpu.s` is the last stop
before the hardware: it carries the `amdhsa.kernels` metadata the loader reads,
so a launch the driver rejects usually disagrees with something there.

### The `pliron-dump` feature

```bash
cargo test -p cubecl-cpu --features cubecl-llvm/pliron-dump
```

The feature is re-exported as `cubecl-cpu/pliron-dump`,
`cubecl-hip/pliron-dump` and `cubecl/pliron-dump` for crates that depend on
those instead.
