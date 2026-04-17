# Complex CUDA Math Surface Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the CUDA complex math surface by making `abs(complex)` return a real value and adding explicit complex support for `tanh` and `powf`, with runtime tests for the supported operator set.

**Architecture:** Reshape the frontend `Abs` trait to support fixed output types, use that to lower complex `abs` as `Arithmetic::Abs` with a real-valued output, and route CUDA complex math through explicit `cuComplex` helper functions. Extend runtime tests to lock the contract against `num_complex` reference values.

**Tech Stack:** `cubecl-core`, `cubecl-cpp`, `cubecl-cuda`, `num-complex`, CUDA `cuComplex.h`

**Design doc:** `docs/plans/2026-04-17-complex-math-surface-design.md`

---

### Task 1: Add failing tests for the missing complex math contract

**Files:**
- Modify: `crates/cubecl-core/src/runtime_tests/complex.rs`
- Test: `cargo test -p cubecl-core --no-run`

**Step 1: Write failing test coverage for `abs(complex)`**

Add kernels and host-side tests that assert:

- `Complex<f32>.abs()` writes `f32`
- `Complex<f64>.abs()` writes `f64`
- results match `num_complex::Complex::norm()`

Use output arrays with element type `C::FloatElem`.

**Step 2: Write failing test coverage for `tanh` and `powf`**

Add kernels and host-side tests for:

- `tanh` on `Complex<f32>` and `Complex<f64>`
- `powf` on `Complex<f32>` and `Complex<f64>`

Use `num_complex` as the reference implementation and compare with tolerances.

**Step 3: Add coverage for the existing contract surface**

Add or consolidate tests for:

- `exp`
- `log`
- `sin`
- `cos`
- `sqrt`

This ensures the whole supported surface is explicitly covered, not only the two newly missing ops.

**Step 4: Run a compile-only check**

Run: `cargo test -p cubecl-core --no-run`
Expected: The new tests compile, even though runtime execution is deferred to CUDA-capable CI/tooling.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/runtime_tests/complex.rs
git commit -m "test: add complex math surface coverage"
```

---

### Task 2: Generalize frontend `Abs` to support real-valued complex outputs

**Files:**
- Modify: `crates/cubecl-core/src/frontend/operation/unary.rs`
- Modify: `crates/cubecl-core/src/frontend/element/complex.rs`
- Modify: `crates/cubecl-core/src/frontend/container/vector/ops.rs`
- Check: `crates/cubecl-core/src/frontend/element/numeric.rs`

**Step 1: Replace same-type `Abs` with a fixed-output trait shape**

Refactor `Abs` in `frontend/operation/unary.rs` so its return type is based on `Self::WithScalar<...>` instead of always `Self`.

Keep the real/int behavior unchanged by making their output type equal to themselves.

**Step 2: Teach complex `Abs` to lower with fixed output**

In `frontend/element/complex.rs`, implement the expand side of `Abs` so:

- the emitted operation is still `Arithmetic::Abs`
- the output item is `T::FloatElem`

This should mirror how `real_val()` and `imag_val()` already use `unary_expand_fixed_output(...)`.

**Step 3: Ensure vectors inherit the right output type**

Update any vector trait bounds or impls so `Vector<Complex<T>, N>::abs()` yields `Vector<T, N>` through `WithScalar`.

**Step 4: Compile-check the frontend**

Run: `cargo check -p cubecl-core`
Expected: `abs` compiles for existing numeric users and now type-checks correctly for complex users.

**Step 5: Commit**

```bash
git add crates/cubecl-core/src/frontend/operation/unary.rs crates/cubecl-core/src/frontend/element/complex.rs crates/cubecl-core/src/frontend/container/vector/ops.rs
git commit -m "feat: make complex abs return real values"
```

---

### Task 3: Expose `tanh` and `powf` for complex types in the frontend

**Files:**
- Modify: `crates/cubecl-core/src/frontend/operation/unary.rs`
- Modify: `crates/cubecl-core/src/frontend/operation/binary.rs`

**Step 1: Add complex support to `Tanh`**

Extend the `impl_unary_func!` invocation for `Tanh` to include:

- `num_complex::Complex<f32>`
- `num_complex::Complex<f64>`

**Step 2: Add complex support to `Powf`**

Extend the `impl_binary_func!` invocation for `Powf` to include:

- `num_complex::Complex<f32>`
- `num_complex::Complex<f64>`

**Step 3: Re-run a focused compile check**

Run: `cargo check -p cubecl-core`
Expected: complex kernels can now be written using `.tanh()` and `.powf(...)`.

**Step 4: Commit**

```bash
git add crates/cubecl-core/src/frontend/operation/unary.rs crates/cubecl-core/src/frontend/operation/binary.rs
git commit -m "feat: expose complex tanh and powf in frontend"
```

---

### Task 4: Add explicit CUDA complex math helpers

**Files:**
- Modify: `crates/cubecl-cpp/src/cuda/dialect.rs`

**Step 1: Add helper functions for complex unary math**

In `cuda/dialect.rs`, under the existing `cuComplex` overload block, add inline helpers for:

- `cubecl_abs(cuFloatComplex)` -> `float`
- `cubecl_abs(cuDoubleComplex)` -> `double`
- `cubecl_exp`
- `cubecl_log`
- `cubecl_sin`
- `cubecl_cos`
- `cubecl_sqrt`
- `cubecl_tanh`

**Step 2: Add helper functions for complex binary math**

Add inline overloads for:

- `cubecl_powf(cuFloatComplex, cuFloatComplex)`
- `cubecl_powf(cuDoubleComplex, cuDoubleComplex)`

Implement `powf(z, w)` as `exp(w * log(z))` using the same helper family.

**Step 3: Keep helpers usable from both host and device builds**

Mark them `__device__ __host__ inline` like the existing operator wrappers.

**Step 4: Compile the CUDA backend**

Run: `cargo check -p cubecl-cpp`
Expected: helper definitions compile cleanly with the current `cuComplex.h` path.

**Step 5: Commit**

```bash
git add crates/cubecl-cpp/src/cuda/dialect.rs
git commit -m "feat: add cuComplex math helpers for complex ops"
```

---

### Task 5: Route shared codegen through the explicit complex helpers

**Files:**
- Modify: `crates/cubecl-cpp/src/shared/unary.rs`
- Modify: `crates/cubecl-cpp/src/shared/binary.rs`

**Step 1: Add complex-aware unary formatting**

Special-case `CF32` and `CF64` in the unary formatter implementations for:

- `Abs`
- `Exp`
- `Log`
- `Sin`
- `Cos`
- `Sqrt`
- `Tanh`

Emit the helper calls from Task 4 instead of generic `exp(...)`, `sqrt(...)`, etc.

**Step 2: Add complex-aware binary formatting for `Powf`**

Special-case `CF32` and `CF64` in `shared/binary.rs` so complex `powf` emits `cubecl_powf(lhs, rhs)`.

**Step 3: Preserve existing scalar behavior**

Do not change codegen for real or integer element types.

**Step 4: Compile-check the codegen path**

Run: `cargo check -p cubecl-cpp -p cubecl-cuda`
Expected: shared formatting compiles and CUDA codegen still builds.

**Step 5: Commit**

```bash
git add crates/cubecl-cpp/src/shared/unary.rs crates/cubecl-cpp/src/shared/binary.rs
git commit -m "feat: route complex math codegen through explicit helpers"
```

---

### Task 6: Verify the whole surface and document remaining runtime limits

**Files:**
- Modify: `crates/cubecl-core/src/runtime_tests/complex.rs` if minor tolerance fixes are needed
- Optional docs note in: `docs/plans/2026-04-17-complex-math-surface-design.md`

**Step 1: Run compile-oriented verification**

Run:

```bash
cargo check -p cubecl-core -p cubecl-cpp -p cubecl-cuda
cargo test -p cubecl-core --no-run
cargo test -p cubecl-cuda --no-run
```

Expected: all compile successfully.

**Step 2: Run the CUDA runtime tests if the environment allows**

Run: `cargo test -p cubecl-cuda test_complex -- --nocapture`

Expected on a compatible CUDA environment:

- the complex runtime tests pass

Expected on this current machine:

- runtime launch may still fail with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`

**Step 3: Record the verification result**

If runtime execution is still blocked by the environment, keep that limitation explicit in the final summary instead of claiming runtime success.

**Step 4: Commit**

```bash
git add crates/cubecl-core/src/runtime_tests/complex.rs docs/plans/2026-04-17-complex-math-surface-design.md
git commit -m "test: verify complex CUDA math surface"
```
