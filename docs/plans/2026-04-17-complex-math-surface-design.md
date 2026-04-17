# Complex CUDA Math Surface Follow-up — Design

## Overview

Close the remaining gap in CubeCL's CUDA complex-number support by making the complex math contract explicit and testable for downstream runtimes.

This follow-up keeps interleaved `Complex<f32>` / `Complex<f64>` as the primary model, keeps invalid operations centrally rejected, and focuses on the CUDA backend only.

## Scope

This design covers:

- `abs`
- `exp`
- `log`
- `sin`
- `cos`
- `sqrt`
- `tanh`
- `powf`
- runtime tests for `Complex32` and `Complex64`

This design does not cover:

- WGPU complex support
- CPU complex support
- GEMM or linalg integration
- HIP support

## API Contract

### `abs(complex)` returns a real scalar

`abs(complex)` should return the magnitude, not a complex value with zero imaginary part.

Concrete return types:

- `num_complex::Complex<f32> -> f32`
- `num_complex::Complex<f64> -> f64`
- `Vector<Complex<f32>, N> -> Vector<f32, N>`
- `Vector<Complex<f64>, N> -> Vector<f64, N>`

This aligns `abs(complex)` with downstream expectations and makes the result-type policy explicit rather than implicit.

### `real_val()` / `imag_val()` remain real-valued

The existing `real_val()` and `imag_val()` methods already return the underlying floating-point scalar. That policy remains unchanged.

### `tanh(complex)` and `powf(complex, complex)` become first-class supported ops

These operations should work through the same frontend/IR/CUDA pipeline as the already-supported complex arithmetic and unary math surface.

## Frontend Design

### 1. Generalize `Abs`

Today `Abs` is modeled as a same-type unary trait. That is correct for real scalars and integers, but incorrect for complex values.

The `Abs` trait should be reshaped to return an associated output type based on `CubePrimitive::WithScalar<_>`:

- real scalars still return themselves
- integers still return themselves
- complex returns its corresponding floating scalar
- vectors inherit the correct result through `WithScalar`

This keeps `Abs` uniform at the trait level without introducing a complex-only escape hatch.

### 2. Use fixed-output expansion for complex `abs`

No new IR opcode is needed. The existing `Arithmetic::Abs` operation is still emitted.

The frontend decides the output type:

- real / int `abs` continues using same-type unary expansion
- complex `abs` uses `unary_expand_fixed_output(...)` with the matching float element type

This mirrors the already-established approach for `Real` and `Imag`.

### 3. Extend complex math trait coverage

The frontend currently allows several unary complex ops but does not fully expose the downstream-required surface.

Required additions:

- add complex types to `Tanh`
- add complex types to `Powf`

`Complex` itself remains an independent trait rather than implementing `Float` or `Numeric`.

## IR Design

No new IR instruction is required for this issue.

The relevant existing operations are already sufficient:

- `Arithmetic::Abs`
- `Arithmetic::Exp`
- `Arithmetic::Log`
- `Arithmetic::Sin`
- `Arithmetic::Cos`
- `Arithmetic::Sqrt`
- `Arithmetic::Tanh`
- `Arithmetic::Powf`
- `Operator::Real`
- `Operator::Imag`

The main change is the frontend result-type policy for `Abs`, not the IR vocabulary.

## CUDA Backend Design

### 1. Stay on `cuComplex.h`

The branch already switched from `thrust::complex` to `cuComplex.h` for NVRTC compatibility. This design keeps that direction.

### 2. Add explicit complex math helpers

The CUDA backend should not rely on generic `exp(z)` / `pow(z, w)` calls accidentally working for `cuFloatComplex` / `cuDoubleComplex`.

Instead, `cuda/dialect.rs` should define explicit inline helpers for:

- `cubecl_abs`
- `cubecl_exp`
- `cubecl_log`
- `cubecl_sin`
- `cubecl_cos`
- `cubecl_sqrt`
- `cubecl_tanh`
- `cubecl_powf`

These helpers should be overloaded for `cuFloatComplex` and `cuDoubleComplex`.

Representative formulas:

- `abs(z) = hypot(re, im)`
- `exp(x + iy) = exp(x) * (cos(y) + i sin(y))`
- `log(z) = log(|z|) + i atan2(im, re)`
- `sin(x + iy) = sin(x) cosh(y) + i cos(x) sinh(y)`
- `cos(x + iy) = cos(x) cosh(y) - i sin(x) sinh(y)`
- `sqrt(z)` via polar half-angle formula or equivalent numerically stable branch
- `tanh(z) = sinh(2x)/(cosh(2x)+cos(2y)) + i sin(2y)/(cosh(2x)+cos(2y))`
- `pow(z, w) = exp(w * log(z))`

### 3. Route shared unary/binary formatting through explicit complex branches

`shared/unary.rs` and `shared/binary.rs` should emit the explicit helper calls whenever the element type is `CF32` or `CF64`.

This makes the generated CUDA source deterministic and backend-owned instead of depending on incidental overload resolution.

### 4. Keep explicit unsupported behavior centralized

The following remain intentionally unsupported for complex values:

- ordering comparisons
- ordering-based `min` / `max`
- bitwise ops
- integer-only saturating ops
- `MulHi`
- `Remainder`

## Testing Strategy

### Runtime tests

Extend `crates/cubecl-core/src/runtime_tests/complex.rs` with focused CUDA runtime tests for:

- `abs` with real-valued outputs
- `exp`
- `log`
- `sin`
- `cos`
- `sqrt`
- `tanh`
- `powf`

Each op should be tested for:

- `Complex<f32>`
- `Complex<f64>`
- nontrivial values with nonzero real and imaginary parts
- branch-sensitive values for `log`, `sqrt`, and `powf`

Reference values should come from `num_complex`.

### Comparison policy

Complex-valued outputs should be compared with per-component tolerances.

Real-valued outputs from `abs` should be compared as floats.

## Validation Notes

On this machine, `cargo test -p cubecl-cuda test_complex` currently fails at launch time with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, so local verification for this issue should be split into:

- compile-level verification locally
- runtime validation on CI or a CUDA environment with a compatible driver/toolchain

## Decisions

- Keep `cuComplex.h` as the CUDA representation for NVRTC compatibility.
- Make `abs(complex)` return a real scalar, not a zero-imaginary complex.
- Do not add new IR opcodes for this issue.
- Implement the remaining complex math surface explicitly in CUDA helper code instead of relying on ambient overloads.
