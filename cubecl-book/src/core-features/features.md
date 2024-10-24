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
| Subcube | ✔️   | ✔️   | ✔️          | ✔️            |
| CMMA    | ✔️   | ✔️   | ❌          | ✔️            |

### Datatypes

| Type | CUDA | ROCm | WGPU (WGSL) | WGPU (SPIR-V) |
| ---- | ---- | ---- | ----------- | ------------- |
| u8   | ❌   | ❌   | ❌          | ❌            |
| u16  | ❌   | ❌   | ❌          | ❌            |
| u32  | ✔️   | ✔️   | ✔️          | ✔️            |
| u64  | ❌   | ❌   | ❌          | ❌            |
| i8   | ❌   | ❌   | ❌          | ❌            |
| i16  | ❌   | ❌   | ❌          | ❌            |
| i32  | ✔️   | ✔️   | ✔️          | ✔️            |
| i64  | ❌   | ❌   | ❌          | ✔️            |
| f16  | ✔️   | ✔️   | ❌          | ✔️            |
| bf16 | ✔️   | ✔️   | ❌          | ❌            |
| f32  | ✔️   | ✔️   | ✔️          | ✔️            |
| f64  | ❌   | ❌   | ❌          | ✔️            |
| bool | ✔️   | ✔️   | ✔️          | ✔️            |

## Subcube

Subcube level operations, i.e.
[`subcube_sum`](https://docs.rs/cubecl/latest/cubecl/frontend/fn.subcube_sum.html),
[`subcube_elect`](https://docs.rs/cubecl/latest/cubecl/frontend/fn.subcube_elect.html).

## Cooperative Matrix Multiply-Add (CMMA)

Subcube-level cooperative matrix multiply-add operations. Maps to `wmma` in CUDA and
`CooperativeMatrixMultiply` in SPIR-V. Features are registered for each size and datatype that is
supported by the hardware. For supported functions, see
[`cmma`](https://docs.rs/cubecl/latest/cubecl/frontend/cmma/index.html).
