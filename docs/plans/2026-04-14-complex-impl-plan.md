# Complex Number Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add interleaved `Complex<f32>` / `Complex<f64>` as first-class types in CubeCL IR with CUDA backend support.

**Architecture:** New `ElemType::Complex(ComplexKind)` variant in IR, independent `Complex` trait in frontend (parallel to Float/Int), `thrust::complex<T>` mapping in CUDA backend. Tests use the `testgen_*` macro pattern executed via `cargo test -p cubecl-cuda`.

**Tech Stack:** CubeCL IR, cubecl-cpp (CUDA codegen), cubecl-core (frontend), cubecl-cuda (runtime tests), `thrust::complex`, `num_complex`.

**Design doc:** `docs/plans/2026-04-14-complex-design.md`

---

### Task 1: Add `ComplexKind` and `ElemType::Complex` to IR

**Files:**
- Modify: `crates/cubecl-ir/src/type.rs:56-67` (add `ComplexKind` enum, add variant to `ElemType`)
- Modify: `crates/cubecl-ir/src/type.rs:152-203` (add `size()` / `size_bits()` arms)

**Step 1: Add `ComplexKind` enum before `ElemType`**

In `crates/cubecl-ir/src/type.rs`, after the `UIntKind` enum (line 56), add:

```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum ComplexKind {
    C32,
    C64,
}
```

Then add variant to `ElemType` (after `Bool` at line 66):

```rust
pub enum ElemType {
    Float(FloatKind),
    Int(IntKind),
    UInt(UIntKind),
    Bool,
    Complex(ComplexKind),
}
```

Note: `ElemType` already derives `From` via `derive_more::From`. Since we added a new variant, `From<ComplexKind>` will be auto-derived. Verify this compiles.

**Step 2: Add `size()` arm**

In the `size()` method (line 180, after `ElemType::Bool`), add:

```rust
            ElemType::Complex(kind) => match kind {
                ComplexKind::C32 => core::mem::size_of::<f32>() * 2,
                ComplexKind::C64 => core::mem::size_of::<f64>() * 2,
            },
```

**Step 3: Add `size_bits()` arm**

In `size_bits()` (line 201, after the existing match arms), the current code has:
```rust
            ElemType::Int(_) | ElemType::UInt(_) | ElemType::Bool => self.size() * 8,
```
Add a new arm:
```rust
            ElemType::Complex(_) => self.size() * 8,
```

Or alternatively, change the existing arm to include Complex:
```rust
            ElemType::Int(_) | ElemType::UInt(_) | ElemType::Bool | ElemType::Complex(_) => self.size() * 8,
```

**Step 4: Add classification methods**

After `is_bool()` (line 228-230), add:

```rust
    pub fn is_complex(&self) -> bool {
        matches!(self, ElemType::Complex(_))
    }

    pub fn as_complex(&self) -> Option<ComplexKind> {
        match self {
            ElemType::Complex(kind) => Some(*kind),
            _ => None,
        }
    }
```

**Step 5: Check compilation**

Run: `cargo check -p cubecl-ir`
Expected: Compile success. May have warnings about unused `ComplexKind` — that's fine.

**Step 6: Commit**

```bash
git add crates/cubecl-ir/src/type.rs
git commit -m "feat: add ComplexKind and ElemType::Complex to IR"
```

---

### Task 2: Register Complex types in CUDA backend (cpp layer)

**Files:**
- Modify: `crates/cubecl-cpp/src/shared/element.rs:11-38` (add `CF32`, `CF64` to `Elem<D>`)
- Modify: `crates/cubecl-cpp/src/shared/element.rs:159-188` (add `ident()` arms)
- Modify: `crates/cubecl-cpp/src/shared/base.rs:73-95` (add `elem_complex` flag to `Flags`)
- Modify: `crates/cubecl-cpp/src/shared/base.rs:2012-2066` (add IR→Elem mapping)

**Step 1: Add `CF32`, `CF64` to `Elem<D>` enum**

In `crates/cubecl-cpp/src/shared/element.rs`, add before `Bool` (line 34):

```rust
    CF32,
    CF64,
```

**Step 2: Add `ident()` arms**

In the `ident()` method (around line 159-188), add matching arms:

```rust
            Elem::CF32 => "cf32",
            Elem::CF64 => "cf64",
```

**Step 3: Add `elem_complex` flag to `Flags`**

In `crates/cubecl-cpp/src/shared/base.rs`, add to `Flags` struct (after line 80):

```rust
    pub elem_complex: bool,
```

Initialize it in the `Flags` default/constructor (search for where other `elem_*` flags are initialized, likely `false`).

**Step 4: Add IR→Elem mapping in `compile_elem()`**

In `crates/cubecl-cpp/src/shared/base.rs`, in the `compile_elem` method that maps `gpu::ElemType` to `Elem<D>` (around line 2064, after `gpu::ElemType::Bool => Elem::Bool`), add:

```rust
            gpu::ElemType::Complex(kind) => {
                self.flags.elem_complex = true;
                match kind {
                    gpu::ComplexKind::C32 => Elem::CF32,
                    gpu::ComplexKind::C64 => Elem::CF64,
                }
            }
```

**Step 5: Check compilation**

Run: `cargo check -p cubecl-cpp`
Expected: May fail until CUDA dialect's `compile_elem` is updated (Task 3). If it fails, continue to Task 3.

**Step 6: Commit**

```bash
git add crates/cubecl-cpp/src/shared/element.rs crates/cubecl-cpp/src/shared/base.rs
git commit -m "feat: add CF32/CF64 to cpp backend element types"
```

---

### Task 3: Add CUDA dialect codegen for Complex types

**Files:**
- Modify: `crates/cubecl-cpp/src/cuda/dialect.rs` (add `compile_elem` arms + include)

**Step 1: Add type name mapping in `compile_elem()`**

In `crates/cubecl-cpp/src/cuda/dialect.rs`, find the `compile_elem` method (around line 255-309). In the non-`words` branch, add before `Bool`:

```rust
            Elem::CF32 => f.write_str("thrust::complex<float>"),
            Elem::CF64 => f.write_str("thrust::complex<double>"),
```

**Step 2: Add `#include <thrust/complex.h>`**

In the same file, find `compile_includes()` (around line 37-80). Add after the existing includes:

```rust
        if flags.elem_complex {
            f.write_str("#include <thrust/complex.h>\n")?;
        }
```

**Step 3: Add word-size type mapping (if needed)**

In the `words` branch of `compile_elem()`, complex types don't have native vector types. For now, skip or map to the same type. If there's no match for `CF32`/`CF64` in the words branch, it may fall through — verify this doesn't cause issues.

**Step 4: Check compilation**

Run: `cargo check -p cubecl-cpp`
Expected: Compile success.

**Step 5: Commit**

```bash
git add crates/cubecl-cpp/src/cuda/dialect.rs
git commit -m "feat: add CUDA dialect codegen for complex types"
```

---

### Task 4: Add Complex element support in CUDA runtime

**Files:**
- Modify: `crates/cubecl-cuda/src/runtime.rs` (register complex types as supported)

**Step 1: Check how types are registered**

Look at `crates/cubecl-cpp/src/shared/base.rs` function `register_supported_types()` (around line 2091). This is likely called during runtime initialization. Add:

```rust
        gpu::ElemType::Complex(gpu::ComplexKind::C32),
        gpu::ElemType::Complex(gpu::ComplexKind::C64),
```

to the `supported_types` array.

**Step 2: Check compilation**

Run: `cargo check -p cubecl-cuda`
Expected: Compile success.

**Step 3: Commit**

```bash
git add crates/cubecl-cuda/ crates/cubecl-cpp/src/shared/base.rs
git commit -m "feat: register complex types in CUDA runtime"
```

---

### Task 5: Add Complex frontend trait and CubePrimitive impl

**Files:**
- Create: `crates/cubecl-core/src/frontend/element/complex.rs`
- Modify: `crates/cubecl-core/src/frontend/element/mod.rs` (add `mod complex; pub use complex::*;`)
- Modify: `crates/cubecl-core/Cargo.toml` (add `num-complex` dependency)

**Step 1: Add `num-complex` dependency**

In `crates/cubecl-core/Cargo.toml`, add to `[dependencies]`:

```toml
num-complex = { workspace = true }
```

Check workspace `Cargo.toml` to see if `num-complex` is already in workspace dependencies. If not, add it there:

```toml
num-complex = "0.4"
```

**Step 2: Create `complex.rs`**

Create `crates/cubecl-core/src/frontend/element/complex.rs`:

```rust
use core::ops::{Add, Div, Mul, Sub};

use crate::{
    ir::{Arithmetic, ComplexKind, ElemType, Scope, StorageType, Type},
    prelude::{
        unexpanded, Assign, AssignExpand, CubePrimitive, CubePrimitiveExpand, CubeType,
        IntoRuntime, NativeAssign, NativeExpand, Scalar,
    },
    unsafe_ignore_fmt,
};
use cubecl_ir::ConstantValue;

pub trait Complex:
    CubePrimitive
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    fn conj(self) -> Self {
        unexpanded!()
    }

    fn real_val(self) -> Self::Scalar {
        unexpanded!()
    }

    fn imag_val(self) -> Self::Scalar {
        unexpanded!()
    }
}

macro_rules! impl_complex {
    ($primitive:ty, $kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = NativeExpand<$primitive>;
        }

        impl Scalar for $primitive {}

        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = crate::prelude::Const<1>;
            type WithScalar<S: Scalar> = S;

            fn as_type_native() -> Option<Type> {
                Some(StorageType::Scalar(ElemType::Complex(ComplexKind::$kind)).into())
            }

            fn from_const_value(_value: ConstantValue) -> Self {
                unimplemented!("Complex constants not yet supported")
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl NativeAssign for $primitive {}

        impl crate::prelude::IntoMut for $primitive {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl Complex for $primitive {}
    };
}

impl_complex!(num_complex::Complex<f32>, C32);
impl_complex!(num_complex::Complex<f64>, C64);
```

**Step 3: Register module**

In `crates/cubecl-core/src/frontend/element/mod.rs`, add:

```rust
mod complex;
pub use complex::*;
```

**Step 4: Check compilation**

Run: `cargo check -p cubecl-core`
Expected: May fail because `core::ops::Neg` is referenced but not imported. Also may fail because the `#[cube]` macro's `normalize_kernel_ty` doesn't know about Complex bounds. Fix compilation errors iteratively.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-core/src/frontend/element/mod.rs crates/cubecl-core/Cargo.toml
git commit -m "feat: add Complex trait and CubePrimitive impl for Complex32/64"
```

---

### Task 6: Add Complex arithmetic expand macros and operations

**Files:**
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs` (add expand macros)
- Modify: `crates/cubecl-core/src/frontend/operation/unary.rs` (add Complex types to applicable `impl_unary_func!` invocations)

**Step 1: Add expand macros in `complex.rs`**

The key insight: `impl_core_binop!` requires `T: Add<Output=T> + CubePrimitive` but NOT `Numeric`. Let's check if `Add`, `Sub`, `Mul`, `Div` are already blanket-implemented for `num_complex::Complex<f32>`. Yes, `num_complex::Complex<T>` implements `Add`, `Sub`, `Mul`, `Div` when `T: Clone + Num`.

So we just need the expand side. Add to `complex.rs`:

```rust
use core::ops::Neg;

macro_rules! impl_complex_binop {
    ($trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            pub trait [<Cube $trait>]: $trait<Output = Self> + CubePrimitive + CubeType<ExpandType: [<$trait Expand>]> + Sized {
                fn [<__expand_ $method>](
                    scope: &mut Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Self>,
                ) -> NativeExpand<Self> {
                    lhs.[<__expand_ $method _method>](scope, rhs)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self) -> Self;
            }

            impl<T: $trait<Output = T> + CubePrimitive> [<Cube $trait>] for T {}
            impl<T: $trait<Output = T> + CubePrimitive> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self) -> Self {
                    crate::frontend::operation::base::binary_expand(scope, self.into(), rhs.into(), $op).into()
                }
            }
        }
    };
}

impl_complex_binop!(Add, add, Arithmetic::Add);
impl_complex_binop!(Sub, sub, Arithmetic::Sub);
impl_complex_binop!(Mul, mul, Arithmetic::Mul);
impl_complex_binop!(Div, div, Arithmetic::Div);
```

For Neg:
```rust
pub trait CubeNeg: Neg<Output = Self> + CubePrimitive + Sized {
    fn __expand_neg(scope: &mut Scope, x: NativeExpand<Self>) -> NativeExpand<Self>;
}

impl<T: Neg<Output = T> + CubePrimitive> CubeNeg for T {}

impl<T: Neg<Output = T> + CubePrimitive> crate::frontend::operation::unary::neg::NegExpand
    for NativeExpand<T>
{
    // This won't work directly since neg::expand is a free function
}
```

Actually, let's look at how `neg` works. The `neg::expand` function at `unary.rs:27-33` takes `E: CubePrimitive` — no `Numeric` bound. So `neg` should work out of the box for Complex types!

Similarly, the `impl_unary_func!` macro takes a list of types. We can add Complex types to the relevant invocations. Let's verify which ones Complex needs:

Complex needs: `Abs`, `Exp`, `Log`, `Sin`, `Cos`, `Sqrt`, `Powf`
Complex does NOT need: `Ceil`, `Floor`, `Round`, `Trunc`, `Erf`, `Tan`, `Tanh`, etc.

**Step 2: Add Complex types to `impl_unary_func!` invocations**

In `crates/cubecl-core/src/frontend/operation/unary.rs`, add `num_complex::Complex<f32>, num_complex::Complex<f64>` to these invocations:

- `impl_unary_func!(Abs, abs, ...)` (line 161)
- `impl_unary_func!(Exp, exp, ...)` (line 186)
- `impl_unary_func!(Log, ln, ...)` (line 197)
- `impl_unary_func!(Cos, cos, ...)` (line 209)
- `impl_unary_func!(Sin, sin, ...)` (line 210)
- `impl_unary_func!(Sqrt, sqrt, ...)` (line 334)

For `Powf`, check how it's defined — it may require `Float` trait. If so, we'll need a separate Complex-specific pow expand.

**Step 3: Check compilation**

Run: `cargo check -p cubecl-core`
Expected: Compile success after adding the import for `num_complex` at the top of `unary.rs`.

**Step 4: Commit**

```bash
git add crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-core/src/frontend/operation/unary.rs
git commit -m "feat: add Complex arithmetic and transcendental expand ops"
```

---

### Task 7: Add Complex-specific operations (Conj, Real, Imag) — Frontend

**Files:**
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs` (add expand methods)
- Modify: `crates/cubecl-ir/src/arithmetic.rs` or `operator.rs` (add IR ops if needed)

**Step 1: Check if IR has Conj operator**

Search `crates/cubecl-ir/src/arithmetic.rs` for `Conj`. If it doesn't exist, add it to the `Arithmetic` enum:

```rust
Conj,
```

Also check if `Operator` has `Real`/`Imag` or if we should use a different mechanism (e.g., lowering via metadata or struct access).

**Step 2: Add expand methods for Complex-specific ops**

In `complex.rs`, add expand traits:

```rust
pub trait ComplexExpand: CubePrimitive {
    fn __expand_conj_method(self, scope: &mut Scope) -> Self;
    fn __expand_real_val_method(self, scope: &mut Scope) -> NativeExpand<Self::Scalar>;
    fn __expand_imag_val_method(self, scope: &mut Scope) -> NativeExpand<Self::Scalar>;
}

impl<T: Complex> ComplexExpand for NativeExpand<T> {
    fn __expand_conj_method(self, scope: &mut Scope) -> Self {
        crate::frontend::operation::base::unary_expand(scope, self.into(), Arithmetic::Conj).into()
    }

    fn __expand_real_val_method(self, scope: &mut Scope) -> NativeExpand<Self::Scalar> {
        let expand_element: crate::ir::ManagedVariable = self.into();
        let item = <T::Scalar as CubePrimitive>::as_type(scope);
        crate::frontend::operation::base::unary_expand_fixed_output(
            scope, expand_element, item, Operator::Real,
        ).into()
    }

    fn __expand_imag_val_method(self, scope: &mut Scope) -> NativeExpand<Self::Scalar> {
        let expand_element: crate::ir::ManagedVariable = self.into();
        let item = <T::Scalar as CubePrimitive>::as_type(scope);
        crate::frontend::operation::base::unary_expand_fixed_output(
            scope, expand_element, item, Operator::Imag,
        ).into()
    }
}
```

Note: `Operator::Real` and `Operator::Imag` may not exist yet. If they don't, we need to add them to `crates/cubecl-ir/src/operator.rs`.

**Step 3: Add IR ops if needed**

In `crates/cubecl-ir/src/operator.rs`, add to the `Operator` enum:

```rust
Real,
Imag,
```

And add to `crates/cubecl-ir/src/arithmetic.rs`:

```rust
Conj,
```

**Step 4: Check compilation**

Run: `cargo check -p cubecl-core`
Expected: Compile success.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-ir/src/operator.rs crates/cubecl-ir/src/arithmetic.rs
git commit -m "feat: add Conj/Real/Imag IR ops and frontend expand"
```

---

### Task 8: Add CUDA codegen for Complex-specific ops (Conj/Real/Imag)

**Files:**
- Modify: `crates/cubecl-cpp/src/shared/instruction.rs` (add dispatch for new ops)
- Modify: `crates/cubecl-cpp/src/shared/unary.rs` (add Conj formatter)
- Possibly: `crates/cubecl-cpp/src/cuda/dialect.rs` (if custom formatting needed)

**Step 1: Add Conj unary formatter**

In `crates/cubecl-cpp/src/shared/unary.rs`, add:

```rust
pub struct Conj;
impl<D: Dialect> Unary<D> for Conj {
    fn format_scalar(f: &mut core::fmt::Formatter, input: Variable<D>, elem: Elem<D>) -> std::fmt::Result {
        write!(f, "thrust::conj({input})")
    }
}
```

**Step 2: Add Real/Imag extraction formatting**

These extract a scalar from a complex value:

```rust
pub struct RealExtract;
impl<D: Dialect> Unary<D> for RealExtract {
    fn format_scalar(f: &mut core::fmt::Formatter, input: Variable<D>, _elem: Elem<D>) -> std::fmt::Result {
        let out_elem = /* the output element type */;
        write!(f, "{out_elem}({input}.real())")
    }
}
```

This may need special handling since the output type is different from the input type (Complex → float).

**Step 3: Add instruction dispatch**

In `crates/cubecl-cpp/src/shared/instruction.rs`, find where `Arithmetic::Abs` etc. are dispatched and add:

```rust
gpu::Arithmetic::Conj => Conj::format(f, &it.input, &it.out),
```

And for `Operator::Real` / `Operator::Imag`:

```rust
gpu::Operator::Real => RealExtract::format(f, &it.input, &it.out),
gpu::Operator::Imag => ImagExtract::format(f, &it.input, &it.out),
```

**Step 4: Add IsNan/IsInf for complex types in CUDA**

In `crates/cubecl-cpp/src/shared/instruction.rs` or `comparison.rs`, where `IsNan` is handled, add complex-specific path:

```rust
// In the IsNan handler, check if input is complex:
if input_elem.is_complex() {
    write!(f, "({out} = (thrust::isnan({input}.real()) || thrust::isnan({input}.imag())))")
}
```

**Step 5: Check compilation**

Run: `cargo check -p cubecl-cpp`
Expected: Compile success.

**Step 6: Commit**

```bash
git add crates/cubecl-cpp/
git commit -m "feat: add CUDA codegen for Conj/Real/Imag complex ops"
```

---

### Task 9: Add CubeElement impl for Complex types

**Files:**
- Modify: `crates/cubecl-core/src/pod.rs` (add `CubeElement` impl for `Complex<f32>` / `Complex<f64>`)

**Step 1: Add `bytemuck::Pod` / `Zeroable` safety**

`num_complex::Complex<T>` is `#[repr(C)]` and `T: Copy`, so it should be safe to implement `Pod`. Check if `bytemuck` already has an impl for `num_complex::Complex`. If not:

```rust
unsafe impl bytemuck::Pod for num_complex::Complex<f32> {}
unsafe impl bytemuck::Zeroable for num_complex::Complex<f32> {}
unsafe impl bytemuck::Pod for num_complex::Complex<f64> {}
unsafe impl bytemuck::Zeroable for num_complex::Complex<f64> {}
```

**Step 2: Add `CubeElement` impl**

In `crates/cubecl-core/src/pod.rs`, add:

```rust
impl CubeElement for num_complex::Complex<f32> {
    fn type_name() -> &'static str {
        "cf32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Complex(ComplexKind::C32).into()
    }
    fn maximum_value() -> Self {
        num_complex::Complex::new(f32::MAX, 0.0)
    }
    fn minimum_value() -> Self {
        num_complex::Complex::new(f32::MIN, 0.0)
    }
}

impl CubeElement for num_complex::Complex<f64> {
    fn type_name() -> &'static str {
        "cf64"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Complex(ComplexKind::C64).into()
    }
    fn maximum_value() -> Self {
        num_complex::Complex::new(f64::MAX, 0.0)
    }
    fn minimum_value() -> Self {
        num_complex::Complex::new(f64::MIN, 0.0)
    }
}
```

**Step 3: Check compilation**

Run: `cargo check -p cubecl-core`
Expected: Compile success.

**Step 4: Commit**

```bash
git add crates/cubecl-core/src/pod.rs
git commit -m "feat: add CubeElement impl for Complex32/64"
```

---

### Task 10: Add runtime test for Complex addition

**Files:**
- Create: `crates/cubecl-core/src/runtime_tests/complex.rs`
- Modify: `crates/cubecl-core/src/runtime_tests/mod.rs` (add module + `testgen_complex!`)
- Modify: `crates/cubecl-cuda/src/lib.rs` (invoke `testgen_complex!`)

**Step 1: Create runtime test kernel**

Create `crates/cubecl-core/src/runtime_tests/complex.rs`:

```rust
use crate::frontend::element::complex::Complex;
use crate::prelude::*;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
pub fn kernel_complex_add<C: Complex>(a: &mut Array<C>, b: Array<C>) {
    a[UNIT_POS] = a[UNIT_POS] + b[UNIT_POS];
}

pub fn test_complex_add<R: Runtime, C: Complex + CubeElement>(client: ComputeClient<R>) {
    let a = vec![C::new(1.0, 2.0), C::new(3.0, 4.0)]; // adjust for num_complex API
    let b = vec![C::new(5.0, 6.0), C::new(7.0, 8.0)];

    let handle_a = client.create_from_slice(C::as_bytes(&a));
    let handle_b = client.create_from_slice(C::as_bytes(&b));

    kernel_complex_add::launch::<C, R>(
        &client,
        CubeCount::Static(2, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle_a.clone(), 2) },
        unsafe { ArrayArg::from_raw_parts(handle_b, 2) },
    );

    let actual = client.read_one_unchecked(handle_a);
    let actual = C::from_bytes(&actual);
    assert_eq!(actual[0], C::new(6.0, 8.0)); // (1+5, 2+6)
    assert_eq!(actual[1], C::new(10.0, 12.0)); // (3+7, 4+8)
}
```

Note: The `C::new(re, im)` constructor and `assert_eq!` require `num_complex::Complex<T>` to implement `PartialEq` (it does). Adjust the constructor call to `num_complex::Complex::new(re, im)`.

**Step 2: Register in `mod.rs`**

Add to `crates/cubecl-core/src/runtime_tests/mod.rs`:
```rust
pub mod complex;
```

And export macro:
```rust
#[macro_export]
macro_rules! testgen_complex {
    () => {
        use super::*;
        use num_complex;

        mod complex {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_add_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_add::<TestRuntime, num_complex::Complex<f32>>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_add_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_add::<TestRuntime, num_complex::Complex<f64>>(client);
            }
        }
    };
}
```

**Step 3: Invoke in CUDA tests**

In `crates/cubecl-cuda/src/lib.rs`, inside the `#[cfg(test)] mod tests` block (after line 76), add:

```rust
    cubecl_core::testgen_complex!();
```

**Step 4: Run the test**

Run: `cargo test -p cubecl-cuda test_complex_add`
Expected: PASS. This validates the full pipeline: frontend → IR → CUDA codegen → runtime.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/runtime_tests/complex.rs crates/cubecl-core/src/runtime_tests/mod.rs crates/cubecl-cuda/src/lib.rs
git commit -m "test: add complex addition runtime test"
```

---

### Task 11: Add runtime tests for complex multiply, conjugate, and abs

**Files:**
- Modify: `crates/cubecl-core/src/runtime_tests/complex.rs` (add more test kernels + functions)

**Step 1: Add multiply test**

Complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`

```rust
#[cube(launch)]
pub fn kernel_complex_mul<C: Complex>(a: &mut Array<C>, b: Array<C>) {
    a[UNIT_POS] = a[UNIT_POS] * b[UNIT_POS];
}

pub fn test_complex_mul<R: Runtime, C: Complex + CubeElement>(client: ComputeClient<R>) {
    // (1+2i) * (3+4i) = (3-8) + (4+6)i = -5 + 10i
    let a = vec![num_complex::Complex::new(1.0f32, 2.0)];
    let b = vec![num_complex::Complex::new(3.0f32, 4.0)];
    // ...
}
```

Adjust for f64 as needed.

**Step 2: Add conjugate test**

```rust
#[cube(launch)]
pub fn kernel_complex_conj<C: Complex>(a: &mut Array<C>) {
    a[UNIT_POS] = a[UNIT_POS].conj();
}
```

Expected: `conj(1+2i) = 1-2i`

**Step 3: Add abs test**

```rust
#[cube(launch)]
pub fn kernel_complex_abs<C: Complex>(a: Array<C>, out: &mut Array<C>) {
    out[UNIT_POS] = a[UNIT_POS].abs();
}
```

Wait — `abs` returns `Self` for Complex (it's defined via `impl_unary_func!` which keeps same type). But `abs(complex)` should return the magnitude as a float. This needs careful design.

Actually, in the `Arithmetic::Abs` IR op, the output type matches the input type. For complex, `abs(c)` could return the magnitude as a complex with zero imaginary part, OR we need a separate `magnitude` operation that returns float.

Let's defer the abs design question and test just mul and conj for now.

**Step 4: Run tests**

Run: `cargo test -p cubecl-cuda test_complex`
Expected: All complex tests pass.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/runtime_tests/complex.rs
git commit -m "test: add complex multiply and conjugate runtime tests"
```

---

### Task 12: Handle #[cube] macro integration for Complex

**Files:**
- Modify: `crates/cubecl-macros/src/parse/kernel.rs` (add `Complex` to recognized trait bounds)

**Step 1: Add Complex to recognized bounds**

In `crates/cubecl-macros/src/parse/kernel.rs`, around line 231-297, where `Float`, `Int`, `Numeric`, `CubePrimitive` are mapped to `DynamicScalar<Marker>`, add:

```rust
"Complex" => {
    map.insert(ident, GenericArg {
        expand_ty: parse_quote!(DynamicScalar<Marker>),
        kind: DefineKind::Type,
    });
}
```

**Step 2: Check compilation**

Run: `cargo check -p cubecl-macros`
Expected: Compile success.

**Step 3: Commit**

```bash
git add crates/cubecl-macros/src/parse/kernel.rs
git commit -m "feat: add Complex to #[cube] macro recognized bounds"
```

---

### Task 13: Full integration test run

**Step 1: Run all cubecl-ir tests**

Run: `cargo test -p cubecl-ir`
Expected: No tests exist, should complete immediately.

**Step 2: Run all cubecl-core tests**

Run: `cargo test -p cubecl-core`
Expected: Only trybuild compile-fail tests run. All pass.

**Step 3: Run all cubecl-cuda tests**

Run: `cargo test -p cubecl-cuda`
Expected: All existing tests pass + new complex tests pass.

**Step 4: Run full xtask test suite**

Run: `cargo xtask test`
Expected: All tests pass.

**Step 5: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: integration fixes for complex type support"
```
