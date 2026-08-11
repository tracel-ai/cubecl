# Triage Graph: tracel-ai/cubecl

## Scan (2026-05-09)

User: kimjune01. 2.1K stars, Rust compute language.
Org status: tracel-ai has burn PR open. Org-blocked for push. Triage only.

### Triaged issues

| # | Score | Signal | Title | Fix branch | Gate | Status |
|---|-------|--------|-------|------------|------|--------|
| 1283 | 5 | Bug label, reporter identified fix line | `Magnitude::magnitude()` returns `Self`, not `Self::Scalar` | `fix/magnitude-return-type` | codex: PASS | READY |
| 1316 | 4 | Bug label, Mesa confirmation, linked to #1200 | f64 ln/exp emit invalid GLSL.std.450 OpExtInst on SPIR-V | `fix/spirv-f64-transcendental-error` | codex: PASS, gemini: PASS | READY |
| 1318 | 5 | Bug label, linked to #1316/#1317 | cubecl-spirv panics on Atomic<u64> compare_exchange_weak | `fix/spirv-atomic-cas-scope` | codex: PASS, gemini: PASS | READY |
| 1276 | 1 | Bug label, maintainer acknowledged | cubecl-runtime panic (conv_direct OOB) | -- | -- | SKIP: maintainer WIP, no repro |

### Fix details

**#1283** (1 file, 1 line)
- File: `crates/cubecl-core/src/frontend/operation/unary.rs`
- Change: `impl_unary_func_scalar_out` macro line 77: `-> Self` to `-> Self::Scalar`
- Root cause: Runtime method signature diverged from expand-time signature

**#1316** (1 file, 33 lines added)
- File: `crates/cubecl-spirv/src/arithmetic.rs`
- Change: Added `assert_not_f64_transcendental` guard to 19 GLSL.std.450 ops
- Root cause: SPIR-V backend emitted f64 operands for instructions that per-spec only accept f16/f32
- Spec: GLSL.std.450 restricts trig/hyp/exp/log/angle ops to 16/32-bit floats
- Guarded ops: Exp, Log, Log1p, Sin, Cos, Tan, Tanh, Sinh, Cosh, Acos, Asin, Atan, Asinh, Acosh, Atanh, Atan2, Pow, Degrees, Radians
- Unguarded (f64-safe): Sqrt, InverseSqrt, Abs, Floor, Ceil, Round, Trunc, FMin, FMax, FClamp, Normalize, Magnitude

**#1318** (1 file, 1 line)
- File: `crates/cubecl-opt/src/instructions.rs`
- Change: `visit_read(self, &mut op.cmp)` (duplicate) -> `visit_read(self, &mut op.input)`
- Root cause: Copy-paste bug in optimizer visitor. `CompareAndSwap` visited `op.cmp` twice, never visited `op.input`. Optimizer lost the dependency on the atomic pointer, breaking the scope registration chain.

### Competing PRs

| # | Title | Overlap |
|---|-------|---------|
| 1321 | Add f64 support back for CUDA | None (different backend) |
| 1197 | feat: add exp2 and log2 operations | Adjacent to #1316 (new ops need same f64 guard) |
| 1301 | Add expm1 unary op | None |

### Dropped

| # | Reason |
|---|--------|
| 1276 | Maintainer says "We're aware of the issue and are working on it!" No repro available. |
