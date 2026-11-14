//! Float contraction: a kernel that turns it off rounds every product and sum on its own, and
//! one that allows it fuses the same products at every vector size.
//!
//! The operands make the two roundings visible. With `a = b = 1 + 2^-12`, the exact product is
//! `1 + 2^-11 + 2^-24`, which rounds to `1 + 2^-11` in `f32`. Adding `c = -(1 + 2^-11)` then
//! gives `0` when the product was rounded first, and `2^-24` when it was fused with the add.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;

/// The lanes of the widest vector the tests launch.
const LANES: usize = 4;

#[cube(launch_unchecked, no_contraction)]
pub fn kernel_mul_add_rounded<N: Size>(
    a: &[Vector<f32, N>],
    b: &[Vector<f32, N>],
    c: &[Vector<f32, N>],
    output: &mut [Vector<f32, N>],
) {
    // Each output reads its own copy of the operands: a product with two uses is never fused.
    output[0] = a[0] * b[0] + c[0];
    output[1] = -c[1] - a[1] * b[1];
}

/// `a * 1 + c * d`, and `(a * 1) + c * d` with `a` a constant, for which a one-lane kernel
/// drops the `* 1` or folds the constant product, then fuses `c * d`. A wider vector must fuse
/// the same product, not the `a * 1` it sees first.
#[cube(launch_unchecked)]
pub fn kernel_mul_add_through_folds<N: Size>(
    a: &[Vector<f32, N>],
    c: &[Vector<f32, N>],
    d: &[Vector<f32, N>],
    output: &mut [Vector<f32, N>],
) {
    output[0] = a[0] * Vector::new(1.0f32) + c[0] * d[0];
    let constant = Vector::<f32, N>::new(-(1.0 + 0.00048828125)) * Vector::new(1.0f32);
    output[1] = constant + c[1] * d[1];
}

/// Two vectors of `value`, one per output.
fn operand(value: f32, vector_size: usize) -> Vec<f32> {
    alloc::vec![value; 2 * vector_size]
}

/// `a * b + c` and `-c - a * b` for every lane of a vector of `vector_size`.
fn mul_adds(client: &Client, vector_size: usize) -> Vec<f32> {
    let factor = 1.0 + 2f32.powi(-12);
    let addend = -(1.0 + 2f32.powi(-11));
    let buffer = |values: Vec<f32>| {
        let len = values.len();
        let handle = client.create_from_slice(f32::as_bytes(&values));
        unsafe { BufferArg::from_raw_parts(handle, len) }
    };
    let outputs = 2 * vector_size;
    let output = client.empty(outputs * core::mem::size_of::<f32>());
    unsafe {
        kernel_mul_add_rounded::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            vector_size,
            buffer(operand(factor, vector_size)),
            buffer(operand(factor, vector_size)),
            buffer(operand(addend, vector_size)),
            BufferArg::from_raw_parts(output.clone(), outputs),
        )
    };
    f32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

/// Both sums of `kernel_mul_add_through_folds` for every lane of a vector of `vector_size`.
fn mul_adds_through_folds(client: &Client, vector_size: usize) -> Vec<f32> {
    let factor = 1.0 + 2f32.powi(-12);
    let addend = -(1.0 + 2f32.powi(-11));
    let buffer = |values: Vec<f32>| {
        let len = values.len();
        let handle = client.create_from_slice(f32::as_bytes(&values));
        unsafe { BufferArg::from_raw_parts(handle, len) }
    };
    let outputs = 2 * vector_size;
    let output = client.empty(outputs * core::mem::size_of::<f32>());
    unsafe {
        kernel_mul_add_through_folds::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            vector_size,
            buffer(operand(addend, vector_size)),
            buffer(operand(factor, vector_size)),
            buffer(operand(factor, vector_size)),
            BufferArg::from_raw_parts(output.clone(), outputs),
        )
    };
    f32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

pub fn test_every_vector_size_fuses_the_same_products(client: Client) {
    let one_lane = mul_adds_through_folds(&client, 1);
    assert_eq!(one_lane.len(), 2, "the one-lane kernel wrote {one_lane:?}");
    let expected: Vec<u32> = one_lane
        .iter()
        .flat_map(|value| core::iter::repeat_n(value.to_bits(), LANES))
        .collect();
    let lanes = mul_adds_through_folds(&client, LANES);
    assert_eq!(
        lanes
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected,
        "{LANES} lanes gave {lanes:?}, one lane {one_lane:?}",
    );
}

pub fn test_no_contraction_rounds_each_operation(client: Client) {
    // WGSL has no way to keep the driver from fusing, and the drivers do.
    if client.name().starts_with("wgpu<wgsl") {
        return;
    }
    for vector_size in [1, LANES] {
        let rounded = [0.0f32; 2];
        let expected: Vec<f32> = rounded
            .iter()
            .flat_map(|&value| core::iter::repeat_n(value, vector_size))
            .collect();
        let actual = mul_adds(&client, vector_size);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "vector size {vector_size}: {actual:?}, where a product fused with its sum gives \
             ±2^-24",
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_contraction {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_every_vector_size_fuses_the_same_products() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::contraction::test_every_vector_size_fuses_the_same_products(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_no_contraction_rounds_each_operation() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::contraction::test_no_contraction_rounds_each_operation(
                client,
            );
        }
    };
}
