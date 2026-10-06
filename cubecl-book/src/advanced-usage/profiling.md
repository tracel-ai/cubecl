# Profiling Kernels

CubeCL gives kernel debug data to the usual debuggers and profilers: gdb, lldb, `perf`, `samply`
and `cargo flamegraph`. It uses only open formats: DWARF, the GDB JIT interface, the perf map,
`#line` directives and SPIR-V debug data. CubeCL adds no profiler of its own.

The `cpu` runtime gives kernel source lines to gdb and lldb (see [Debuggers](#debuggers)), and
kernel names to `perf` and `samply`. The `cuda` runtime gives source lines to the NVIDIA tools
(see [CUDA](#cuda)).

## Enable Debug Data

Kernels get debug data from the same setting as your host code: the `debug` key of the cargo
profile. The `dev` profile sets it. For a release build, add a profile:

```toml
[profile.profiling]
inherits = "release"
debug = "line-tables-only"
```

Then each kernel has line tables. Line tables do not change the machine code. They add only
compile time.

In gdb, each inlined `#[cube]` function is a frame of its own, so a backtrace shows the call chain
inside a kernel. `perf` and `samply` show a kernel as one frame.

If you override the profile for one package, put the same override on `cubecl-runtime`. CubeCL
reads the `debug` value of `cubecl-runtime`.

To remove the debug data from a build that has it, set `CUBECL_DEBUG_INFO=none` (see
[Configuration](./config.md)).

## Source Files

The kernel debug data has the path of each source file that `file!()` gives. This path is relative
to the directory where cargo compiled the crate, usually the workspace root. The binary keeps only
the relative path, so `--remap-path-prefix` and cargo `trim-paths` apply to it.

When a kernel compiles, CubeCL looks for each file. It searches `CUBECL_SOURCE_ROOT`, then the
working directory and its parents. It puts the first directory that has the file into the kernel
debug data. Then a debugger or a profiler opens the file from any directory. With `Full` debug
data, the file must also have the text that the kernel was compiled from.

This applies to the LLVM compilers (CPU, CUDA and HIP), to SPIR-V, and to the `#line` directives
of the C++ compilers.

If you run the binary outside of its source tree, set `CUBECL_SOURCE_ROOT` to the workspace root.

If the source tree is not on the computer, a kernel with `Full` debug data can write its source
files into a directory that you select:

```sh
CUBECL_SOURCE_CACHE=~/.cache/cubecl/sources ./app
```

CubeCL writes the files only when it does not find the source tree, and only for kernels that have
the source text. Each set of files goes into a subdirectory with a name from the MD5 of the files.
CubeCL does not remove the files. A kernel with line tables only keeps the relative path.

## Profiler Symbol Files

A profiler finds JIT code only through symbol files. These files stay after the process stops, so
you must ask for them at run time: `CUBECL_JIT_SYMBOLS=perf` (or `perfmap`) writes the **perf
map** (`/tmp/perf-<pid>.map`). It gives the name and the address range of each kernel.
`perf report`, `perf script` and `samply` read it with no extra step.

If `CUBECL_JIT_SYMBOLS` is not set, CubeCL reads `DOTNET_PerfMapEnabled`, the .NET variable for
the same file: `1` and `3` give the perf map.

A kernel without debug data writes no symbols.

`cargo flamegraph` uses the perf map. It names each kernel, but it gives no source lines in the
kernel. `perf` cannot unwind JIT code with DWARF, so record with frame pointers (see
[Frame Pointers](#frame-pointers)):

```sh
RUSTFLAGS="-C force-frame-pointers=yes" CUBECL_JIT_SYMBOLS=perf \
    cargo flamegraph --profile profiling -c "record -F 997 --call-graph fp -g"
```

`samply` reads the perf map too:

```sh
CUBECL_JIT_SYMBOLS=perf samply record ./target/profiling/app
```

## Frame Pointers

`perf --call-graph fp` walks the stack with frame pointers. Build with
`RUSTFLAGS="-C force-frame-pointers=yes"`. Then the JIT kernels keep frame pointers too.

## Debuggers

gdb and lldb see each kernel through the GDB JIT interface. This is automatic when the kernel has
debug data, and it writes no files. A breakpoint in a `#[cube]` function or an interrupt shows the
kernel frames with their source lines.

## CUDA

The `cuda` runtime with the LLVM compiler gives the PTX of each kernel a `.loc` directive for each
source line. CubeCL loads the PTX with `CU_JIT_GENERATE_LINE_INFO`. Thus the machine code (SASS)
has the line table. Each inlined `#[cube]` function is an inline frame with its name. The
optimization level does not change.

The PTX cannot contain the source text, because `ptxas` does not accept it. Thus the tools read the
source from the file (see [Source Files](#source-files)).

The NVRTC compiler (C++) gives the source lines through `#line` directives, with the path from the
same search, but no inline frames.

The example `profiling_kernels` has two kernels with nested `#[cube]` functions. Use it to try the
commands below. The `dev` profile gives line tables:

```sh
cargo build -p profiling_kernels --no-default-features --features cuda
```

Nsight Compute shows the source lines of each kernel. `--import-source yes` copies the source files
into the report, so the report shows the source on a different computer:

```sh
ncu --import-source yes --set full -o kernels ./target/debug/profiling_kernels
ncu --import kernels.ncu-rep --page source --print-source cuda,sass
```

The metrics of a line in an inlined `#[cube]` function include all of its call sites. The Inline
Functions table of the Nsight Compute user interface gives each call site.

cuda-gdb stops in an inlined `#[cube]` function. It shows the function name and the call site, for
example `main.rs:11 in square_third inlined from main.rs:17`. Do not compile with `-G`: CubeCL does
not need it, and it changes the machine code. Set the breakpoint after the first kernel launch.
Before the launch, cuda-gdb puts a breakpoint on a `#[cube]` line into the host code that the macro
generates.

```sh
cuda-gdb -ex 'set cuda break_on_launch application' -ex run \
    -ex 'break main.rs:11' -ex continue ./target/debug/profiling_kernels
```

If a tool does not find the source, run the program from the workspace, or set
`CUBECL_SOURCE_ROOT` (see [Source Files](#source-files)).

## Worker Threads

The `cpu` runtime runs kernels on its worker threads. Thus a kernel stack does not start at the
host function that launched the kernel. To connect them, build with the `tracing` feature: each
launch is a `tracing` span with the host call site.
