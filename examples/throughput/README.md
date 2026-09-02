# throughput

Small benchmarks that measure a device's **peak throughput** for a few workloads:

| example           | measures                                                     |
| ----------------- | ------------------------------------------------------------ |
| `compute_direct`  | peak arithmetic throughput, per float type the device supports |
| `compute_cmma`    | peak cooperative-matrix throughput, per accumulator width      |
| `memory`          | peak memory (copy) bandwidth                                   |
| `memory_read`     | peak read-only streaming bandwidth                             |
| `memory_write`    | peak write-only streaming bandwidth                            |
| `memory_curve`    | bandwidth against working set size, for each access pattern    |
| `launch_overhead` | the fixed cost of a single kernel launch                       |
| `all`             | every single-size probe above, as one table                    |

## Running

Pick a backend with a feature flag:

```sh
cargo run --release -p throughput --features wgpu --example all
cargo run --release -p throughput --features cuda --example memory
cargo run --release -p throughput --features cuda --example compute_cmma
cargo run --release -p throughput --features wgpu --example launch_overhead
```

Always use `--release`; a debug build measures the wrong thing.

Example output:

```
Peak throughput — cuda / NVIDIA GeForce RTX 4070 Ti SUPER
  compute-direct f32                         46.0146 TOPS/s
  compute-direct f16                         48.5997 TOPS/s
  compute-direct bf16                        48.5997 TOPS/s
  compute-cmma   f16→f16 32×8×16            192.7744 TOPS/s
  compute-cmma   f16→f32 32×8×16             96.3201 TOPS/s
  memory         copy    1 GiB            601.4902 Gbytes/s
  memory         read    512 MiB          660.2920 Gbytes/s
  memory         write   512 MiB          748.0780 Gbytes/s
  launch                                     3.502µs/launch
```

A row reads `unsupported` where the device implements no such thing: a
cooperative matrix on a card without tensor hardware, a type it cannot compute
in. A row that measured nothing reads `N/A` instead — the two are different
answers.

## Reading the numbers

The two accumulator rows are separate because consumer parts halve their tensor
rate for f32 accumulation, and that is the rate a matmul runs on.

Memory counts the bytes a kernel asks for, not the traffic that reaches DRAM: an
ordinary store also pays read-for-ownership where the hardware does, so on a CPU
the write and copy figures land near half the bus while read lands near all of
it. Comparing any of them against a vendor bus figure needs that in mind.

`memory` reports one working set. `memory_curve` reports the whole sweep, and a
kernel moving far less than the top of it cannot reach the top of it.

## Backends

| feature        | runtime        | notes                        |
| -------------- | -------------- | ---------------------------- |
| `wgpu`         | `WgpuRuntime`  | WGSL                         |
| `vulkan`       | `WgpuRuntime`  | SPIR-V                       |
| `metal`        | `WgpuRuntime`  | MSL (via wgpu)               |
| `webgpu`       | `WgpuRuntime`  | WebGPU                       |
| `cuda`         | `CudaRuntime`  | NVIDIA                       |
| `hip` / `rocm` | `HipRuntime`   | AMD                          |
| `cpu`          | `CpuRuntime`   | CPU (MLIR)                   |
| `metal-native` | `MetalRuntime` | Apple (native Metal backend) |

The `wgpu`, `vulkan`, `metal` and `webgpu` features all run on `WgpuRuntime`;
the feature only selects which compiler/adapter is used. Enabling more than one
runtime backend at once simply runs the benchmark on each.

## Caching

Peak throughput is cached per device, so repeated runs return the first
measured value instantly. Toggle it with `CUBECL_THROUGHPUT_CACHE`:

```sh
# force a fresh measurement (ignore and overwrite the cache)
CUBECL_THROUGHPUT_CACHE=off cargo run --release -p throughput --example all --features wgpu
```

Accepted values: `on` / `1` / `true` to enable (the default), `off` / `0` /
`false` to disable.
