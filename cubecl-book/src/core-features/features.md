# Querying hardware features

Some features and datatypes are only supported on some hardware or some backends. They can be
queried with:

```rust, ignore
client.properties().feature_enabled(feature)
```

Also see [`Feature`](https://docs.rs/cubecl/latest/cubecl/enum.Feature.html).

## Overview

### Features

Also requires device support

| Feature            | CUDA | ROCm | WGPU (WGSL) | WGPU (SPIR-V) |
| ------------------ | ---- | ---- | ----------- | ------------- |
| Plane              | ✔️   | ✔️   | ✔️          | ✔️            |
| CMMA               | ✔️   | ✔️   | ❌          | ✔️            |
| Tensor Accelerator | ✔️   | ❌   | ❌          | ❌            |
| Block scaled MMA   | ✔️   | ❌   | ❌          | ❌            |

### Datatypes

`flex32` is implementation dependent. Allows using f16 for MMA on all platforms, and reduced
precision for most operations in Vulkan. `f64` not supported for all operations

| Type   | CUDA   | ROCm | WGPU (WGSL) | WGPU (SPIR-V) |
| ------ | ------ | ---- | ----------- | ------------- |
| u8     | ✔️     | ✔️   | ❌          | ✔️            |
| u16    | ✔️     | ✔️   | ❌          | ✔️            |
| u32    | ✔️     | ✔️   | ✔️          | ✔️            |
| u64    | ✔️     | ✔️   | ❌          | ✔️            |
| i8     | ✔️     | ✔️   | ❌          | ✔️            |
| i16    | ✔️     | ✔️   | ❌          | ✔️            |
| i32    | ✔️     | ✔️   | ✔️          | ✔️            |
| i64    | ✔️     | ✔️   | ❌          | ✔️            |
| fp4    | ✔️[^1] | ❌   | ❌          | ❌            |
| fp8    | ✔️[^1] | ❌   | ❌          | ✔️[^1]        |
| f16    | ✔️     | ✔️   | ❌          | ✔️            |
| bf16   | ✔️     | ✔️   | ❌          | ❔[^2]        |
| flex32 | ❔     | ❔   | ❔          | ✔️            |
| tf32   | ✔️     | ❌   | ❌          | ❌            |
| f32    | ✔️     | ✔️   | ✔️          | ✔️            |
| f64    | ❔     | ❔   | ❌          | ❔            |
| bool   | ✔️     | ✔️   | ✔️          | ✔️            |

## Datatype Details

### Flex32

Relaxed precision 32-bit float. Minimum range and precision is equivalent to `f16`, but may be
higher. Defaults to `f32` when relaxed precision isn't supported.

### Tensor-Float32

19-bit CUDA-only type that should only be used as a CMMA matrix type. May be able to reinterpret
from `f32`, but officially undefined. Use `Cast::cast_from` to safely convert.

## Feature Details

### Plane

Plane level operations, i.e.
[`plane_sum`](https://docs.rs/cubecl/latest/cubecl/frontend/fn.plane_sum.html),
[`plane_elect`](https://docs.rs/cubecl/latest/cubecl/frontend/fn.plane_elect.html).

### Cooperative Matrix Multiply-Add (CMMA)

Plane-level cooperative matrix multiply-add operations. Maps to `wmma` in CUDA and
`CooperativeMatrixMultiply` in SPIR-V. Features are registered for each size and datatype that is
supported by the hardware. For supported functions, see
[`cmma`](https://docs.rs/cubecl/latest/cubecl/frontend/cmma/index.html).

### Tensor accelerator

Async tensor loading using the TMA accelerator available on Blackwell cards.

### Block scaled MMA

Plane-level cooperative matrix multiply-add operations, with built-in block scaling. Available on
Blackwell cards.

---

[^1]: fp8/fp6/fp4 types are supported only for conversion and MMA

<!-- -->

[^2]: bf16 is only supported for conversion, CMMA and, on some platforms, dot product
