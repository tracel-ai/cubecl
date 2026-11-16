//! What the `OpenCL` C emission defines where C leaves it open (float to integer conversions out of
//! range), and the cooperative-matrix element-wise operation on Intel's DPAS fragments.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::OpenClRuntime;

#[cube(launch_unchecked)]
fn float_to_int(input: &[f32], signed: &mut [i32], unsigned: &mut [u32]) {
    if ABSOLUTE_POS < input.len() {
        signed[ABSOLUTE_POS] = i32::cast_from(input[ABSOLUTE_POS]);
        unsigned[ABSOLUTE_POS] = u32::cast_from(input[ABSOLUTE_POS]);
    }
}

/// Out of range, a float saturates to the integer's bounds and NaN converts to 0 (as on CUDA and
/// WGSL); in range it rounds toward zero.
#[test]
fn float_to_int_saturates() {
    let client = R::client(&Default::default());
    let input = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e10,
        -1e10,
        -1.5,
        2.7,
        -2.7,
        4294967040.0,
    ];
    let n = input.len();
    let source = client.create_from_slice(f32::as_bytes(&input));
    let signed = client.empty(n * 4);
    let unsigned = client.empty(n * 4);
    unsafe {
        float_to_int::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(source, n),
            BufferArg::from_raw_parts(signed.clone(), n),
            BufferArg::from_raw_parts(unsigned.clone(), n),
        );
    }
    let signed = client
        .read_one(signed)
        .expect("the output should be read back");
    let unsigned = client
        .read_one(unsigned)
        .expect("the output should be read back");
    assert_eq!(
        i32::from_bytes(&signed),
        &[
            0,
            i32::MAX,
            i32::MIN,
            i32::MAX,
            i32::MIN,
            -1,
            2,
            -2,
            i32::MAX
        ]
    );
    assert_eq!(
        u32::from_bytes(&unsigned),
        &[0, u32::MAX, 0, u32::MAX, 0, 0, 2, 0, 4294967040]
    );
}

#[cube(launch_unchecked)]
fn accumulator_elementwise(input: &[f32], output: &mut [f32], beta: f32, #[comptime] n: usize) {
    let mut acc = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            8usize,
            n,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    cmma::load_with_layout(
        &mut acc,
        &input[0..comptime![8 * n]],
        n as u32,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::execute_elementwise_op(&acc, &acc, |row, col, elem| {
        elem * f32::cast_from(row) + f32::cast_from(col) + beta
    });
    cmma::store(
        &mut output[0..comptime![8 * n]],
        &acc,
        n as u32,
        cmma::MatrixLayout::RowMajor,
    );
}

/// An elementwise function of an accumulator sees each value's row and column (DPAS: 8 rows, a
/// plane wide).
#[test]
fn accumulator_elementwise_sees_rows_and_columns() {
    let client = R::client(&Default::default());
    if client.properties().features.matmul.cmma.is_empty() {
        return;
    }
    let n = client.properties().hardware.plane_size_max as usize;
    let input: Vec<f32> = (0..8 * n).map(|i| i as f32 * 0.5).collect();
    let source = client.create_from_slice(f32::as_bytes(&input));
    let output = client.empty(8 * n * 4);
    unsafe {
        accumulator_elementwise::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            BufferArg::from_raw_parts(source, 8 * n),
            BufferArg::from_raw_parts(output.clone(), 8 * n),
            2.0,
            n,
        );
    }
    let bytes = client
        .read_one(output)
        .expect("the output should be read back");
    let expected: Vec<f32> = (0..8 * n)
        .map(|i| input[i] * (i / n) as f32 + (i % n) as f32 + 2.0)
        .collect();
    assert_eq!(f32::from_bytes(&bytes), &expected[..]);
}
