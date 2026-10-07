<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tracel-ai/cubecl/main/assets/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/tracel-ai/cubecl/main/assets/logo-light.svg" alt="CubeCL" width="400px"/>
</picture>

<br />
<br />

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/KSBSPhAUCc)
[![Current Crates.io Version](https://img.shields.io/crates/v/cubecl.svg)](https://crates.io/crates/cubecl)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/cubecl)](https://crates.io/crates/cubecl)
[![Test Status](https://github.com/tracel-ai/cubecl/actions/workflows/ci.yml/badge.svg)](https://github.com/tracel-ai/cubecl/actions/workflows/ci.yml)
[![Documentation](https://docs.rs/cubecl/badge.svg)](https://docs.rs/cubecl)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
<br />
[![NVIDIA](https://img.shields.io/badge/nvidia-cuda-82b432)](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-cuda)
[![AMD](https://img.shields.io/badge/amd-rocm-c22b23)](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-hip)
[![Apple](https://img.shields.io/badge/apple-metal-a2aaad)](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-wgpu)
[![WGPU](https://img.shields.io/badge/cross_platform-wgpu-008855)](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-wgpu)
[![CPU](https://img.shields.io/badge/cpu-llvm-4a6fa5)](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-cpu)

---

**One codebase. More hardware choices.**
<br/>
Write your compute kernels in Rust. CubeCL translates them for different GPUs and CPUs, so you can reuse the same logic across hardware backends.
<br/>

</div>

## TL;DR

CubeCL is a Rust language extension, a Just-in-Time compiler, and a set of runtimes for writing high-performance compute kernels.
A single `#[cube]` Rust function is compiled on demand and runs on CUDA, HIP, Metal, Vulkan, WebGPU, or CPU, while still using the best instructions each platform offers.
The programming model is low-level by design, so that a single kernel can reach peak performance on every backend it targets.

## Motivation

A `#[cube]` function is written in regular Rust, not in a separate shader language.
That means it is type-checked, borrow-checked, composable, and testable, and you do not have to context-switch into another language or build shader sources by string concatenation at runtime.
A `#[cube]` trait lets one kernel skeleton take different operations.
`#[derive(CubeType)]` turns a struct or an enum into a kernel type.
A `match` on a comptime enum leaves a single arm in the compiled kernel.
Generics are specialized per type, so `F: Float` costs nothing at runtime.
Many mistakes surface as Rust compile errors, and the CPU runtime lets `cargo test` run your kernels on your development machine without a GPU.

Performance is not given up in exchange.
Comptime specializes kernels at compile time, autotune searches the configuration space at first run, and kernels can take tensor core paths on hardware that has them.

The same properties make CubeCL a reasonable foundation for scientific computing in Rust.
It sits between low-level wrappers like `wgpu` and `cudarc` and high-level frameworks like [Burn](https://burn.dev), so that kernel libraries can be written once and reused across the ecosystem.

## How it works

CubeCL describes hardware as four levels of parallelism.
Each level is defined by how its members work together, and each runtime maps it onto its own hardware.

| Level         | Shares through             | CUDA / HIP                | WebGPU             | Metal               | CPU                                  |
| ------------- | -------------------------- | ------------------------- | ------------------ | ------------------- | ------------------------------------ |
| **Unit**      | its own registers          | thread                    | invocation         | thread              | a task on one core                   |
| **Plane**     | lockstep instructions      | warp                      | subgroup           | SIMD-group          | 1 unit                               |
| **Cube**      | shared memory, `sync_cube` | block                     | workgroup          | threadgroup         | the cores side by side               |
| **Hypercube** | global memory              | several blocks            | several workgroups | several threadgroups | a loop each core runs over the cubes |

There is no hypercube builtin. A kernel uses `CUBE_POS` and `CUBE_COUNT` to split one job over several cubes, which combine their results in global memory.

On CPU, parallelism comes from the cube's units, spread over the cores, and cubes run as a loop.
On GPU, parallelism comes from the number of cubes.

<details>
<summary>Topology equivalence with CUDA, WebGPU, and Metal 👇</summary>
<br />

<div align="center">
<img src="https://raw.githubusercontent.com/tracel-ai/cubecl/main/assets/cubecl.drawio.svg" width="100%"/>
</div>

<br />

Topology variables are constants within a kernel entry point, so CubeCL uses Rust constant syntax with capital letters.
Often when writing kernels we do not care about the relative position of a unit within a cube along each axis, only its position in general.
Each kind of variable therefore has its own axis-independent variable, which is often not present in other languages.

| CubeCL             | CUDA                       | WebGPU                 | Metal                            |
| ------------------ | -------------------------- | ---------------------- | -------------------------------- |
| CUBE_COUNT         | N/A                        | N/A                    | N/A                              |
| CUBE_COUNT_X       | gridDim.x                  | num_workgroups.x       | threadgroups_per_grid.x          |
| CUBE_COUNT_Y       | gridDim.y                  | num_workgroups.y       | threadgroups_per_grid.y          |
| CUBE_COUNT_Z       | gridDim.z                  | num_workgroups.z       | threadgroups_per_grid.z          |
| CUBE_POS           | N/A                        | N/A                    | N/A                              |
| CUBE_POS_X         | blockIdx.x                 | workgroup_id.x         | threadgroup_position_in_grid.x   |
| CUBE_POS_Y         | blockIdx.y                 | workgroup_id.y         | threadgroup_position_in_grid.y   |
| CUBE_POS_Z         | blockIdx.z                 | workgroup_id.z         | threadgroup_position_in_grid.z   |
| CUBE_DIM           | N/A                        | N/A                    | N/A                              |
| CUBE_DIM_X         | blockDim.x                 | workgroup_size.x       | threads_per_threadgroup.x        |
| CUBE_DIM_Y         | blockDim.y                 | workgroup_size.y       | threads_per_threadgroup.y        |
| CUBE_DIM_Z         | blockDim.z                 | workgroup_size.z       | threads_per_threadgroup.z        |
| CUBE_CLUSTER_DIM   | N/A                        | N/A                    | N/A                              |
| CUBE_CLUSTER_DIM_X | cluster.dim_blocks().x     | N/A                    | N/A                              |
| CUBE_CLUSTER_DIM_Y | cluster.dim_blocks().y     | N/A                    | N/A                              |
| CUBE_CLUSTER_DIM_Z | cluster.dim_blocks().z     | N/A                    | N/A                              |
| CUBE_POS_CLUSTER   | cluster.block_rank()       | N/A                    | N/A                              |
| CUBE_POS_CLUSTER_X | cluster.block_index().x    | N/A                    | N/A                              |
| CUBE_POS_CLUSTER_Y | cluster.block_index().y    | N/A                    | N/A                              |
| CUBE_POS_CLUSTER_Z | cluster.block_index().z    | N/A                    | N/A                              |
| UNIT_POS           | N/A                        | local_invocation_index | thread_index_in_threadgroup      |
| UNIT_POS_X         | threadIdx.x                | local_invocation_id.x  | thread_position_in_threadgroup.x |
| UNIT_POS_Y         | threadIdx.y                | local_invocation_id.y  | thread_position_in_threadgroup.y |
| UNIT_POS_Z         | threadIdx.z                | local_invocation_id.z  | thread_position_in_threadgroup.z |
| PLANE_POS          | N/A                        | subgroup_id            | simdgroup_index_in_threadgroup   |
| PLANE_DIM          | warpSize                   | subgroup_size          | threads_per_simdgroup            |
| UNIT_POS_PLANE     | N/A                        | subgroup_invocation_id | thread_index_in_simdgroup        |
| ABSOLUTE_POS       | N/A                        | N/A                    | N/A                              |
| ABSOLUTE_POS_X     | N/A                        | global_id.x            | thread_position_in_grid.x        |
| ABSOLUTE_POS_Y     | N/A                        | global_id.y            | thread_position_in_grid.y        |
| ABSOLUTE_POS_Z     | N/A                        | global_id.z            | thread_position_in_grid.z        |

The cluster variables follow CUDA thread block clusters, available on NVIDIA Hopper and newer with the `cuda-cpp` compiler.

</details>

### Example

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn sigmoid<N: Size>(input: &[Vector<f32, N>], output: &mut [Vector<f32, N>]) {
    let x = input[ABSOLUTE_POS];
    output[ABSOLUTE_POS] = Vector::one() / (Vector::one() + Vector::exp(-x));
}

sigmoid::launch(
    &client,
    CubeCount::Static(1, 1, 1), // how many cubes
    CubeDim::new_1d(64),        // units per cube
    4,                          // vector size: elements per unit per instruction
    unsafe { BufferArg::from_raw_parts(input, 256) },
    unsafe { BufferArg::from_raw_parts(output, 256) },
);
```

The launch chooses how many cubes run (`CubeCount`), how many units each cube has (`CubeDim`), and how many elements each unit handles per instruction (the vector size `N`).

The hardware groups each cube's units into **planes** (warps on CUDA, subgroups on WebGPU, SIMD-groups on Metal).
A plane's units run in lockstep and can trade values directly, for example with `plane_sum`.
A plane is 32 units on NVIDIA, 32 or 64 on AMD, and 1 on CPU, so kernels read `PLANE_DIM` instead of assuming a width.

Good CubeCL kernels adapt to these values rather than hardcoding them.
Writing `warpSize == 32` is a common CUDA antipattern, because it breaks on AMD where the warp size is 64, and it is meaningless on CPU where a plane is a single unit.
This is also part of why CPU is a first-class target rather than something bolted on afterwards.
And it is why autotune is so useful, because it is essentially searching over comptime-resolved choices along these levels.

For a complete runnable example with device setup and buffers, see [`examples/gelu`](https://github.com/tracel-ai/cubecl/tree/main/examples/gelu).

### From Rust to IR to many backends

CubeCL uses Rust's proc macro system in a slightly unusual way. The `#[cube]` macro does not lower a function to an intermediate representation directly.
Instead, it rewrites the function into a new Rust function that is semantically similar to the original, but whose job is to build the IR when called.

```
#[cube] fn ──▶ expanded Rust ──▶ CubeCL IR (pliron)
                                     │
       ┌───────────────┬─────────────┼─────────────┐
       ▼               ▼             ▼             ▼
      LLVM            MSL          SPIR-V         WGSL
   ┌────┼────┐      (Metal)       (Vulkan)      (WebGPU)
   ▼    ▼    ▼
  PTX AMDGPU CPU
```

CubeCL's IR and compiler passes are written in Rust with [pliron](https://github.com/pliron-org/pliron).
For the LLVM targets, the build fetches a prebuilt LLVM, so you don't need to install it yourself.

A few useful properties fall out of this design.
Comptime is essentially Rust running at expansion time, so `comptime!` blocks are ordinary Rust whose results are baked into the IR as constants.
Vectorization is a type-level parameter, which means each backend can lower it to the best SIMD instruction available.

## Special features

### Automatic vectorization

High-performance kernels should rely on SIMD instructions whenever possible, but doing so by hand can quickly get complicated.
With CubeCL, you specify a vectorization factor when launching a kernel, and inside the kernel you keep using a single type, `Vector<F, N>`, whose width `N` is picked at launch.
Number literals broadcast automatically (`x * 2.0`), and a scalar value becomes a vector with `Vector::new`.
The runtime lowers this to the platform's native SIMD instructions.
When the algorithm itself depends on the vectorization factor, you can read it at comptime without any runtime cost.

How much vectorization buys you depends on the target.
The CPU runtime tends to benefit from larger vectorization factors thanks to AVX-512, while GPUs mostly see their gains saturate around 128 bits, which matches the width of a global memory load.

### Comptime

Comptime is a way to modify the compiler IR at the time a kernel is first compiled.
It enables lots of optimizations and flexibility without having to write many separate variants of the same kernel.

| Use case                       | Why it matters                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instruction Specialization** | Not all instructions are available on all hardware, so when a specialized one exists, you want to enable it with a simple if statement.                       |
| **Adaptive Parallelism**       | You can pass the device's properties, such as its plane width range, along with the cube dimensions and the vector size, as comptime values to specialize the kernel without hardcoding warp sizes or block dimensions. |
| **Loop Unrolling**             | You may want multiple flavors of the same kernel, with loop unrolling for only a certain range of values. This can be configured easily with comptime.        |
| **Shape Specialization**       | For deep learning kernels it is often crucial to rely on different kernels for different input sizes. You can do this by passing shape as comptime values.    |
| **Compile Time Calculation**   | You can compute a constant using Rust at host time and inject it into a kernel during its compilation, so it does not need to be recalculated on each launch. |

### Autotuning

Autotuning drastically simplifies kernel selection by running small benchmarks at runtime to figure out the best kernel and configuration to use on the current hardware.
Combined with comptime, it can even search over genuinely different specializations rather than only over runtime parameters.
Sometimes the results are surprising, which is part of why this is so useful for portability.

Even if these benchmarks add some overhead the first time, the results get cached on the device and reused on later runs.
When you have control over the deployment target, you can save a warm environment with `cubecl::environment::bundle()` and ship it with your binary, which removes the cold-start cost entirely.
A dry run compiles and tunes every launch without executing it, so producing that bundle is cheap.

## Ecosystem

CubeCL is one part of a small ecosystem with deliberately separated concerns, so you can go as low as you need.

- **Write your own kernels.** The [CubeCL book](https://burn.dev/books/cubecl/) takes you from setup to your first kernel, then through the core features. This repository contains the language, the JIT compiler, the IR, and the per-platform runtimes.
- **Build on [CubeK](https://github.com/tracel-ai/cubek).** Ready-made kernels for matrix multiplication, attention, and reductions, plus tiling APIs to write your own. It was split out from CubeCL so that kernel work and language work can move at their own cadences, since the kind of pull requests and the rate of updates are very different between the two.
- **Train and run models with [Burn](https://burn.dev).** Burn uses CubeCL to run on GPUs and CPUs. It is the deep learning framework that drove CubeCL's design, and its production proving ground.

## Supported platforms

| Platform | Compiler       | Runtime |
| -------- | -------------- | ------- |
| NVIDIA   | LLVM → PTX     | CUDA    |
| AMD      | LLVM → AMDGPU  | HIP     |
| Apple    | MSL            | wgpu    |
| Vulkan   | SPIR-V         | wgpu    |
| WebGPU   | WGSL           | wgpu    |
| CPU      | LLVM → native  | cpu     |

`cuda-cpp`, `hip-cpp` and `metal-native` are available as alternatives.

Not all platforms support the same features.
For example, tensor cores are not available with the WGSL compiler (plain WebGPU) or on CPU.
When you try to use an instruction that is not available on the current platform, you will get a compilation error at runtime.
The launch function is responsible for dispatching the right specialization based on the device properties.

## Getting started

Add CubeCL to your Cargo.toml, and turn on the runtimes your binary links.

```toml
[dependencies]
cubecl = { version = "*", features = ["cuda"] }   # or "hip", "metal", "vulkan", "webgpu", "cpu"
```

A kernel library turns none of them on.
It compiles against no runtime at all, and stays free to run on any of them.

To start building, read the [CubeCL book](https://burn.dev/books/cubecl/).
You can also browse runnable kernels in the [`examples/`](https://github.com/tracel-ai/cubecl/tree/main/examples) directory, and the per-crate API on [docs.rs](https://docs.rs/cubecl).
For real-world kernels built on CubeCL, see [CubeK](https://github.com/tracel-ai/cubek).

## Status

CubeCL is currently in alpha.
It is used in production by [Burn](https://burn.dev), which is also the framework that drove its design, but the public API is still evolving and you should expect breaking changes between minor versions.
We recommend pinning a version if you depend on it directly.

## History

CubeCL started as a WebGPU-only backend for Burn.
As we optimized it, we realized that we needed an intermediate representation that could be optimized and then compiled to WGSL.
Having an IR made it easy to support another compilation target, so we added a CUDA runtime.
Writing kernels directly in that IR was not easy though, so we created a Rust frontend using the [syn](https://github.com/dtolnay/syn) crate.
Navigating the differences between CUDA and WebGPU while trying to leverage both platforms forced us to come up with general concepts that worked everywhere.
That is how CubeCL was born.

## Compared to other projects

CubeCL is a low-level GPU/CPU programming language similar to CUDA, and it aims to reach peak performance on every backend it supports.
The tradeoff compared to tile DSLs is not performance but kernel complexity, because the programmer has to handle device properties explicitly and specialize the kernel for them.
CubeCL relies on Rust's type system to manage that complexity, which is already good at expressing abstractions, rather than shipping a separate language frontend.
If you'd rather write at the tile level, [CubeK](https://github.com/tracel-ai/cubek) provides tiling APIs built on CubeCL, plus ready-made matmul, attention and reduction kernels.

Kernels are compiled just-in-time, not ahead-of-time.
Only the variants you actually launch are generated, so you do not end up with a precompiled library that has to cover every combination of shape, hardware target, and instruction set, which can easily run into gigabytes for AoT projects like hand-written CUDA.
A compilation cache stores the results between runs, and you can ship a warm cache with your binary when you know the deployment target, so the cold-start cost is paid once at build time.
Tile DSLs like Triton and TileLang use a similar JIT model.

CubeCL is also a runtime, not only a language.
The per-platform runtimes handle compilation, dispatch, autotune caching, and memory, and they are not bound to the `#[cube]` frontend.
A kernel produced by a separate DSL like cuTile lowers to PTX, and the CubeCL CUDA runtime executes PTX through `cudarc`, so external kernels can plug directly into the runtime without going through the CubeCL IR.

| Project                                              | Language | Model       | Targets                              | Notes                             |
| ---------------------------------------------------- | -------- | ----------- | ------------------------------------ | --------------------------------- |
| **CubeCL**                                           | Rust     | Cube / SIMT | NVIDIA, AMD, Apple, Vulkan, Web, CPU | low-level, also a runtime         |
| [cuda-oxide](https://github.com/NVlabs/cuda-oxide)   | Rust     | SIMT        | NVIDIA                               | rustc codegen backend to PTX      |
| [cuTile Rust](https://github.com/NVlabs/cutile-rs)   | Rust     | Tile        | NVIDIA                               | tile DSL via NVIDIA Tile IR       |
| [Triton](https://github.com/openai/triton)           | Python   | Tile        | NVIDIA, AMD                          | mature, large kernel ecosystem    |
| [TileLang](https://github.com/tile-ai/tilelang)      | Python   | Tile (TVM)  | NVIDIA, AMD                          | peak-performance AI kernels       |
| CUDA C++                                             | C++      | SIMT        | NVIDIA                               | similar low-level paradigm        |
| WGSL / WebGPU                                        | WGSL     | SIMT        | Cross-platform                       | used as a CubeCL backend via wgpu |

## Community

We are a small team also building [Burn](https://burn.dev), so questions, ideas, and contributions are all very welcome.
The easiest place to reach us is on [Discord](https://discord.gg/KSBSPhAUCc).
Porting algorithms is one of the most useful things you can contribute, more than you would imagine.

## License

CubeCL is dual licensed under either [MIT](https://github.com/tracel-ai/cubecl/blob/main/LICENSE-MIT) or [Apache 2.0](https://github.com/tracel-ai/cubecl/blob/main/LICENSE-APACHE), at your option.
