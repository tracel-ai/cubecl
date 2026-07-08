# throughput

Small benchmarks that measure a device's **peak throughput** for a few workloads:

| example          | measures                                          |
| ---------------- | ------------------------------------------------- |
| `compute_direct` | peak arithmetic throughput (non-CMMA, f32)        |
| `compute_cmma`   | peak tensor-core (CMMA) throughput (f16 → f32)    |
| `memory`         | peak memory (copy) bandwidth                      |
| `all`            | runs all of the above and prints them as a table  |

## Running

Pick a backend with a feature flag:

```sh
cargo run --release -p throughput --features wgpu --example all
cargo run --release -p throughput --features cuda --example memory
cargo run --release -p throughput --features cuda --example compute_cmma
```

Always use `--release`; a debug build measures the wrong thing.

Example output:

```
Peak throughput — wgpu<wgsl>
  compute-direct f32                          6.4877 TOPS/s
  compute-cmma   f16→f16 16×16×16               unsupported
  memory         f32                      131.5563 Gbytes/s
```

CMMA needs a tensor-core backend. On backends without it (e.g. WGSL) it prints
`unsupported` and is skipped.

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
