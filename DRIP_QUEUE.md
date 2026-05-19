# Drip Queue: tracel-ai/cubecl

## Status: BLOCKED (org-blocked for push, burn PR open)

When unblocked, push in this order. One PR at a time. Wait for merge/close before pushing next.

### Queue

| Order | Branch | Issue | Title | Risk | Size |
|-------|--------|-------|-------|------|------|
| 1 | `fix/magnitude-return-type` | #1283 | Magnitude::magnitude() returns Self::Scalar | Low | 1 line |
| 2 | `fix/spirv-atomic-cas-scope` | #1318 | Fix CAS optimizer visitor (missing op.input) | Low | 1 line |
| 3 | `fix/spirv-f64-transcendental-error` | #1316 | Reject f64 transcendentals at SPIR-V compile time | Medium | 33 lines |

### Ordering rationale

1. **#1283 first**: Simplest fix, reporter gave exact line. No controversy. Establishes credibility.
2. **#1318 second**: One-line copy-paste bug fix in the optimizer. Clear root cause. Both codex and gemini confirmed.
3. **#1316 last**: Largest change (19 call sites). Compile-time error is the conservative choice; maintainers may prefer polyfill. Higher review surface area.

### Gate results

| Branch | codex | gemini | cargo check |
|--------|-------|--------|-------------|
| `fix/magnitude-return-type` | PASS | -- | PASS (cubecl-core) |
| `fix/spirv-atomic-cas-scope` | PASS | PASS | PASS (cubecl-opt) |
| `fix/spirv-f64-transcendental-error` | PASS | PASS (spec-verified) | PASS (cubecl-spirv) |

### Pre-push checklist

- [ ] Confirm org-block is lifted (burn PR merged/closed)
- [ ] Rebase each branch onto latest upstream/main before push
- [ ] Run `cargo test` if CI access is available
- [ ] Open PR for queue position 1 only
- [ ] After merge, push queue position 2, repeat
