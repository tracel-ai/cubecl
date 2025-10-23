use crate::{self as cubecl};

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
};

use cubecl_runtime::MmaConfig;
use half::{bf16, f16};

#[cube(launch)]
pub fn cast_matrix_f16(input: &Array<f32>, out: &mut Array<f16>) {
    let acc = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            16,
            16,
            16,
            cmma::MatrixLayout::Undefined,
        )
    };
    cmma::load_with_layout(&acc, &input.to_slice(), 16, cmma::MatrixLayout::RowMajor);

    let output = cmma::cast::<f32, f16>(&acc);

    cmma::store(
        &mut out.to_slice_mut(),
        &output,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube(launch)]
pub fn cast_matrix_bf16(input: &Array<f32>, out: &mut Array<bf16>) {
    let acc = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            16,
            16,
            16,
            cmma::MatrixLayout::Undefined,
        )
    };
    cmma::load_with_layout(&acc, &input.to_slice(), 16, cmma::MatrixLayout::RowMajor);

    let output = cmma::cast::<f32, bf16>(&acc);

    cmma::store(
        &mut out.to_slice_mut(),
        &output,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

pub fn test_cmma_cast_f16<R: Runtime>(client: ComputeClient<R::Server>, cube_dimensions: CubeDim) {
    if !client.properties().features.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16).into(),
        b_type: ElemType::Float(FloatKind::F16).into(),
        cd_type: ElemType::Float(FloatKind::F32).into(),
        m: 16,
        k: 16,
        n: 16,
    }) {
        // We can't execute the test, skip.
        return;
    }

    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let input = client.create(f32::as_bytes(&input));
    let out = client.empty(core::mem::size_of::<f16>() * 256);

    unsafe {
        cast_matrix_f16::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f32>(&input, 256, 1),
            ArrayArg::from_raw_parts::<f16>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out);
    let actual = f16::from_bytes(&actual);
    let expected: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}

pub fn test_cmma_cast_bf16<R: Runtime>(client: ComputeClient<R::Server>, cube_dimensions: CubeDim) {
    if !client.properties().features.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::BF16).into(),
        b_type: ElemType::Float(FloatKind::BF16).into(),
        cd_type: ElemType::Float(FloatKind::F32).into(),
        m: 16,
        k: 16,
        n: 16,
    }) {
        // We can't execute the test, skip.
        return;
    }

    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let input = client.create(f32::as_bytes(&input));
    let out = client.empty(core::mem::size_of::<f16>() * 256);

    unsafe {
        cast_matrix_bf16::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f32>(&input, 256, 1),
            ArrayArg::from_raw_parts::<f16>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out);
    let actual = bf16::from_bytes(&actual);
    let expected: Vec<bf16> = (0..256).map(|i| bf16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}
