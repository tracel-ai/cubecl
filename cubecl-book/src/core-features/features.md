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

| Feature | CUDA | ROCm | WGPU (WGSL) | WGPU (SPIR-V) |
| ------- | ---- | ---- | ----------- | ------------- |
| Plane   | ✔️   | ✔️   | ✔️          | ✔️            |
| CMMA    | ✔️   | ✔️   | ❌          | ✔️            |

### Datatypes

`flex32` represented as `f32` everywhere except SPIR-V, with no reduced precision. `f64` not
supported for all operations

| Type   | CUDA | ROCm | WGPU (WGSL) | WGPU (SPIR-V) |
| ------ | ---- | ---- | ----------- | ------------- |
| u8     | ✔️   | ✔️   | ❌          | ✔️            |
| u16    | ✔️   | ✔️   | ❌          | ✔️            |
| u32    | ✔️   | ✔️   | ✔️          | ✔️            |
| u64    | ✔️   | ✔️   | ❌          | ✔️            |
| i8     | ✔️   | ✔️   | ❌          | ✔️            |
| i16    | ✔️   | ✔️   | ❌          | ✔️            |
| i32    | ✔️   | ✔️   | ✔️          | ✔️            |
| i64    | ✔️   | ✔️   | ❌          | ✔️            |
| f16    | ✔️   | ✔️   | ❌          | ✔️            |
| bf16   | ✔️   | ✔️   | ❌          | ❌            |
| flex32 | ❔   | ❔   | ❔          | ✔️            |
| tf32   | ✔️   | ❌   | ❌          | ❌            |
| f32    | ✔️   | ✔️   | ✔️          | ✔️            |
| f64    | ❔   | ❔   | ❌          | ❔            |
| bool   | ✔️   | ✔️   | ✔️          | ✔️            |

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
