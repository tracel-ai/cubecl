//! Correctness test for the Metal `expm1` lowering on large/extreme inputs.
//!
//! MSL has no `expm1` builtin, so the dialect emits a numerically stable identity whose
//! naive form returns `NaN` for large `x` (where `exp(x)` overflows to `+inf`). This
//! exercises the dialect's `isinf` guard: large inputs must return `+inf` (not `NaN`),
//! near-zero must stay accurate, and large-negative must underflow toward `-1`.

use cubecl_core::{self as cubecl, prelude::*};

type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn expm1_kernel(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].exp_m1();
    }
}

fn ordered_f32_bits(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    if bits < 0 { i32::MIN - bits } else { bits }
}

fn assert_f32_ulp_le(actual: f32, expected: f32, max_ulp: u32) {
    if actual == expected {
        return;
    }
    let diff = ordered_f32_bits(actual).abs_diff(ordered_f32_bits(expected));
    assert!(
        diff <= max_ulp,
        "values differ by more than {max_ulp} ulp: actual={actual}, expected={expected}, ulp_diff={diff}"
    );
}

#[test]
fn expm1_handles_large_and_extreme_inputs() {
    let device = Default::default();
    let client = R::client(&device);

    // Regimes: 0.0 (exact zero), 1e-7 (catastrophic cancellation), 10.0 (well-conditioned),
    // 88.0 (f32 `exp` overflow boundary ~88.72; Metal saturates early to +inf), 100.0 (past
    // overflow: +inf), -100.0 (underflow to -1).
    let input = [0.0f32, 1e-7, 10.0, 88.0, 100.0, -100.0];
    let n = input.len();

    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(n * core::mem::size_of::<f32>());

    unsafe {
        expm1_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(output_handle);
    let actual = f32::from_bytes(&bytes);

    for (i, &x) in input.iter().enumerate() {
        let a = actual[i];
        let expected = x.exp_m1();

        // For x >= ~88, `exp(x)` overflows to `+inf`; the lowering must not yield `NaN`.
        assert!(!a.is_nan(), "expm1({x}) produced NaN; expected ~{expected}");

        if expected.is_infinite() {
            assert!(
                a.is_infinite() && a.is_sign_positive(),
                "expm1({x}) must be +inf; got {a}"
            );
        } else if expected >= f32::MAX / 16.0 {
            // Overflow boundary: Metal's `exp` saturates earlier than IEEE `expf`, so
            // +inf is acceptable; otherwise the finite result must still be close.
            assert!(
                (a.is_infinite() && a.is_sign_positive()) || {
                    let diff = ordered_f32_bits(a).abs_diff(ordered_f32_bits(expected));
                    diff <= 4
                },
                "expm1({x}) must be +inf or ~{expected}; got {a}"
            );
        } else {
            assert_f32_ulp_le(a, expected, 2);
        }
    }
}
