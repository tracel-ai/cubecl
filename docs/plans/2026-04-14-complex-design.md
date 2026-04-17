# Interleaved Complex Number Support for CubeCL — Design

## Overview

Add interleaved `Complex<f32>` / `Complex<f64>` as first-class types in CubeCL IR, with CUDA and WGPU backend support.

## Trait Hierarchy

Independent `Complex` trait, parallel to `Float` and `Int`:

```
CubePrimitive
  ├── Numeric (Add/Sub/Mul/Div/Neg/Abs/Remainder + num_traits)
  │     ├── Float (Exp/Log/Sin/Cos/Sqrt/Tan/Powf/Ceil/Floor/...)
  │     └── Int (Bitwise/Saturating/...)
  └── Complex (Add/Sub/Mul/Div/Neg/Abs + Exp/Log/Sin/Cos/Sqrt/Powf + Conj/Real/Imag)
```

Complex does NOT implement Numeric, Float, or Int. Invalid ops (Remainder, Ceil/Floor/Round/Trunc, Bitwise, ordering comparisons) are excluded by type system.

## Design Sections

### 1. IR Type System (`cubecl-ir`)

**File:** `crates/cubecl-ir/src/type.rs`

Add to `ElemType`:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum ComplexKind {
    C32,  // Complex<f32>
    C64,  // Complex<f64>
}

pub enum ElemType {
    Float(FloatKind),
    Int(IntKind),
    UInt(UIntKind),
    Bool,
    Complex(ComplexKind),  // NEW
}
```

`size()` / `size_bits()`: return 2x base float size (C32 = 8 bytes/64 bits, C64 = 16 bytes/128 bits).

Classification methods:
- `is_complex()`, `as_complex()`

No changes to `StorageType` or `Type` — `Scalar(ElemType::Complex(C32))` works as-is.

**~30 lines changed.**

### 2. CUDA Backend (`cubecl-cpp` / `cubecl-cuda`)

**Key enabler:** `thrust::complex<T>` provides operator overloading for `+`, `-`, `*`, `/`, `==`, `!=`, plus `thrust::abs`, `thrust::exp`, `thrust::log`, `thrust::sqrt`, `thrust::conj`, etc.

**Element type mapping:**

| File | Change |
|------|--------|
| `shared/element.rs` | Add `CF32`, `CF64` variants to `Elem<D>` |
| `cuda/dialect.rs` `compile_elem()` | `CF32` → `"thrust::complex<float>"`, `CF64` → `"thrust::complex<double>"` |
| `cuda/dialect.rs` `compile_includes()` | Conditional `#include <thrust/complex.h>` via `flags.elem_complex` |
| `shared/base.rs` | `ElemType::Complex(C32/C64)` → `Elem::CF32/CF64` mapping, set `flags.elem_complex` |

**Binary ops:** No changes needed. Existing `operator!(Add, "+")` etc. work via thrust overloading.

**Unary ops:** Most work as-is via `function!(Exp, "exp")` pattern with thrust namespace functions. Half-support disabled for complex types.

**New custom ops:**

| Op | Generated code | Location |
|----|----------------|----------|
| Conj | `thrust::conj(input)` | `unary.rs`, ~1 line |
| Real (extract) | `(input).real()` | New |
| Imag (extract) | `(input).imag()` | New |
| IsNan | `thrust::isnan(re) \|\| thrust::isnan(im)` | Custom |
| IsInf | `thrust::isinf(re) \|\| thrust::isinf(im)` | Custom |

**Rejected ops (panic in backend dispatch):**
- All bitwise ops
- Ordering comparisons (`<`, `<=`, `>`, `>=`)
- `Ceil`, `Floor`, `Round`, `Trunc`, `%`, `MulHi`, Saturating ops

**~65 lines changed.**

### 3. Frontend (`cubecl-core` / `cubecl-macros`)

**New file:** `crates/cubecl-core/src/frontend/element/complex.rs`

```rust
macro_rules! impl_complex {
    ($primitive:ty, $kind:ident) => {
        impl CubeType for $primitive { type ExpandType = NativeExpand<Self>; }
        impl Scalar for $primitive {}
        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = Const<1>;
            type WithScalar<S: Scalar> = S;
            fn as_type_native() -> Option<Type> {
                Some(StorageType::Scalar(ElemType::Complex(ComplexKind::$kind)).into())
            }
        }
        impl Complex for $primitive {}
    };
}

impl_complex!(num_complex::Complex<f32>, C32);
impl_complex!(num_complex::Complex<f64>, C64);
```

**Complex trait:**

```rust
pub trait Complex: CubePrimitive {
    fn conj(self) -> Self { unexpanded!() }
    fn real(self) -> Self::Scalar { unexpanded!() }
    fn imag(self) -> Self::Scalar { unexpanded!() }
}
```

**Arithmetic expand:** New `impl_complex_binop!` macro (same body as `impl_core_binop!` but without `Numeric` bound). Calls `binary_expand()` — identical IR emission.

**Transcendental expand:** `impl_complex_unary_func!` macro (same body as `impl_unary_func!`). Calls `unary_expand()` — identical IR emission.

**`CubeElement` (`pod.rs`):** `unsafe impl bytemuck::Pod` for `num_complex::Complex<f32/f64>` (they are `#[repr(C)]` with two f32/f64).

**Dependency:** Add `num-complex` to `cubecl-core/Cargo.toml`.

**~163 lines changed.**

### 4. Validation

**Primary mechanism: Rust type system.**

- `Complex` does not implement `Float` → Ceil/Floor/Round/Trunc/Sin/Cos caller sites reject
- `Complex` does not implement `Int` → Bitwise/Saturating caller sites reject
- `Complex` does not implement `Numeric` → Remainder (%) caller sites reject
- `Complex` does not implement `PartialOrd` → `<`, `>`, `<=`, `>=` reject

**Secondary mechanism: Backend dispatch panic.**

For IR instructions constructed directly (low-level paths), `base.rs` dispatch panics on invalid ops for complex types.

**~20 lines changed.**

## Scope Summary

| Scope | Lines | Difficulty |
|-------|-------|------------|
| IR type system | ~30 | Low |
| CUDA backend | ~65 | Low |
| Frontend (Complex trait + macros) | ~163 | Low–Medium |
| Validation | ~20 | Low |
| **Total Phase 1** | **~278** | **Low–Medium** |

Phase 2 (WGPU) and Phase 3 (CPU) are deferred.

## Operation Compatibility Matrix

| Category | Works (same IR, thrust overloading) | Needs custom impl | Not applicable |
|----------|--------------------------------------|-------------------|----------------|
| Binary arithmetic | `+`, `-`, `*`, `/` | `pow` | `%`, `mulhi`, saturating |
| Unary math | `abs`, `exp`, `log`, `sqrt`, `sin`, `cos` | `isnan`, `isinf` | `ceil`, `floor`, `round`, `trunc` |
| Comparison | `==`, `!=` | — | `<`, `<=`, `>`, `>=` |
| Bitwise | — | — | All |
| Structural | All (element-type agnostic) | — | — |
| Reduction | `sum`, `prod` | — | `max`, `min` |
| Complex-specific | — | `conj`, `real`, `imag` | — |

## Decisions Log

- **Interleaved (not split)**: Required for cuSOLVER/cuBLAS interop, cache locality, SIMD
- **Independent Complex trait (not Numeric extension)**: Clean separation, invalid ops excluded by type system, minimal code duplication (macros reuse `binary_expand`/`unary_expand`)
- **HIP deferred**: No CI, no local test hardware
- **WGPU deferred to Phase 2**: WGSL has no native complex type; CUDA is the priority
