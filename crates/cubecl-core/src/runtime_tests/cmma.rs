use crate::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
    runtime_tests::binary::assert_equals_approx,
};

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
};

use cubecl_common::{e2m1, e2m1x2, ue8m0};
use cubecl_ir::MatrixIdent;
use cubecl_runtime::{MmaConfig, ScaledMmaConfig};
use half::{bf16, f16};

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_f16_m16n16k16_gmem(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1_lined(
    lhs: &Array<Line<f16>>,
    rhs: &Array<Line<f16>>,
    out: &mut Array<Line<f32>>,
) {
    let a = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<Line<f32>>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        Line::cast_from(0.0),
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1_lined_offset(
    lhs: &Array<Line<f16>>,
    rhs: &Array<Line<f16>>,
    out: &mut Array<Line<f32>>,
    offset_lhs: u32,
    offset_rhs: u32,
    offset_out: u32,
) {
    let len_lhs = lhs.len();
    let len_rhs = rhs.len();
    let len_out = out.len();

    let a = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.slice(offset_lhs, len_lhs),
        16,
    );
    let b = cmma::Matrix::<Line<f16>>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.slice(offset_rhs, len_rhs),
        16,
    );
    let c = cmma::Matrix::<Line<f32>>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        Line::cast_from(0.0),
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(
        &mut out.slice_mut(offset_out, len_out),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_2(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f16>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        8,
        8,
        8,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        8,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        8,
        8,
        8,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        8,
    );
    let c = cmma::Matrix::<f16>::from_value(
        cmma::MatrixIdent::Accumulator,
        8,
        8,
        8,
        cmma::MatrixLayout::Undefined,
        half::f16::from_int(0),
    );

    cmma::execute::<f16, f16, f16, f16>(&a, &b, &c, &c);

    cmma::store(&mut out.to_slice_mut(), &c, 8, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_3(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        8,
        8,
        8,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        8,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        8,
        8,
        8,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        8,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8,
        8,
        8,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);

    cmma::store(&mut out.to_slice_mut(), &c, 8, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_tf32(lhs: &Array<tf32>, rhs: &Array<tf32>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<tf32>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        8,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        8,
    );
    let b = cmma::Matrix::<tf32>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        8,
        cmma::MatrixLayout::RowMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        8,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute::<tf32, tf32, f32, f32>(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

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

pub fn test_simple_1_lined<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
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

    let lhs: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..256).map(|i| f16::from_f32((i % 8) as f32)).collect();

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * 256);

    unsafe {
        kernel_simple_1_lined::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, 256 / 4, 4),
            ArrayArg::from_raw_parts::<f16>(&rhs, 256 / 4, 4),
            ArrayArg::from_raw_parts::<f32>(&out, 256 / 4, 4),
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_1_expected(), actual);
}

pub fn test_simple_1_lined_offset<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
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
    let offset_lhs = 1usize;
    let offset_rhs = 0usize;
    let offset_out = 0usize;
    let line_size = 2usize;

    let lhs: Vec<f16> = (0..256 + offset_lhs * line_size)
        .map(|i| f16::from_f32(i as f32 - (offset_lhs * line_size) as f32))
        .collect();
    let rhs: Vec<f16> = (0..256i32 + (offset_rhs * line_size) as i32)
        .map(|i| f16::from_f32(((i - (offset_rhs * line_size) as i32) % 8) as f32))
        .collect();

    let lhs_len = lhs.len() / line_size;
    let rhs_len = rhs.len() / line_size;
    let out_len = (256 / line_size) + offset_out;

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * line_size * out_len);

    unsafe {
        kernel_simple_1_lined_offset::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, lhs_len, line_size as u8),
            ArrayArg::from_raw_parts::<f16>(&rhs, rhs_len, line_size as u8),
            ArrayArg::from_raw_parts::<f32>(&out, out_len, line_size as u8),
            ScalarArg::new(offset_lhs as u32),
            ScalarArg::new(offset_rhs as u32),
            ScalarArg::new(offset_out as u32),
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(
        test_simple_1_expected(),
        actual[(offset_out * line_size)..actual.len()]
    );
}

pub fn test_simple_1<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
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

    let lhs: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..256).map(|i| f16::from_f32((i % 8) as f32)).collect();

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * 256);

    unsafe {
        kernel_simple_f16_m16n16k16_gmem::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, 256, 1),
            ArrayArg::from_raw_parts::<f16>(&rhs, 256, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_1_expected(), actual);
}

pub fn test_simple_1_expected() -> Vec<f32> {
    vec![
        504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504.,
        504., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400.,
        1400., 1400., 1400., 1400., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296.,
        2296., 2296., 2296., 2296., 2296., 2296., 2296., 3192., 3192., 3192., 3192., 3192., 3192.,
        3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 4088., 4088., 4088.,
        4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088.,
        4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984.,
        4984., 4984., 4984., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880.,
        5880., 5880., 5880., 5880., 5880., 5880., 6776., 6776., 6776., 6776., 6776., 6776., 6776.,
        6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 7672., 7672., 7672., 7672.,
        7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 8568.,
        8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568.,
        8568., 8568., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464.,
        9464., 9464., 9464., 9464., 9464., 10360., 10360., 10360., 10360., 10360., 10360., 10360.,
        10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 11256., 11256.,
        11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256.,
        11256., 11256., 11256., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152.,
        12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 13048., 13048., 13048.,
        13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048.,
        13048., 13048., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944.,
        13944., 13944., 13944., 13944., 13944., 13944., 13944.,
    ]
}

// pub fn test_simple_2<R: Runtime>(
//     client: ComputeClient<R>,
//     cube_dimensions: CubeDim,
// ) {
//     if !client.properties().features.cmma.contains(&MmaConfig {
//         a: Elem::Float(FloatKind::F16),
//         b: Elem::Float(FloatKind::F16),
//         c: Elem::Float(FloatKind::F16),
//         m: 8,
//         k: 8,
//         n: 8,
//     }) {
//         // We can't execute the test, skip.
//         return;
//     }

//     let lhs: Vec<f16> = (0..64).map(|i| f16::from_f32(i as f32)).collect();
//     let rhs: Vec<f16> = (0..64).map(|i| f16::from_f32((i % 8) as f32)).collect();

//     let lhs = client.create(f16::as_bytes(&lhs));
//     let rhs = client.create(f16::as_bytes(&rhs));
//     let out = client.empty(core::mem::size_of::<f16>() * 64);

//     unsafe {
//         kernel_simple_2::launch(
//             &client,
//             CubeCount::Static(1, 1, 1),
//             cube_dimensions,
//             ArrayArg::from_raw_parts::<f16>(&lhs, 64, 1),
//             ArrayArg::from_raw_parts::<f16>(&rhs, 64, 1),
//             ArrayArg::from_raw_parts::<f16>(&out, 64, 1),
//         )
//     };

//     let actual = client.read_one(out);
//     let actual = f16::from_bytes(&actual);

//     let expected: [f16; 64] = [0.0, 28.0, 56.0, 84.0, 112.0, 140.0, 168.0, 196.0, 0.0, 92.0, 184.0, 276.0, 368.0, 460.0, 552.0, 644.0, 0.0, 156.0, 312.0, 468.0, 624.0, 780.0, 936.0, 1092.0, 0.0, 220.0, 440.0, 660.0, 880.0, 1100.0, 1320.0, 1540.0, 0.0, 284.0, 568.0, 852.0, 1136.0, 1420.0, 1704.0, 1988.0, 0.0, 348.0, 696.0, 1044.0, 1392.0, 1740.0, 2088.0, 2436.0, 0.0, 412.0, 824.0, 1236.0, 1648.0, 2060.0, 2472.0, 2884.0, 0.0, 476.0, 952.0, 1428.0, 1904.0, 2380.0, 2856.0, 3332.0].map(|e| f16::from_f64(e));

//     assert_eq!(expected, actual);
// }

pub fn test_cmma_cast_f16<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
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
    let input = client.create_from_slice(f32::as_bytes(&input));
    let out = client.empty(core::mem::size_of::<f16>() * 256);

    unsafe {
        cast_matrix_f16::launch(
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

pub fn test_cmma_cast_bf16<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
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
    let input = client.create_from_slice(f32::as_bytes(&input));
    let out = client.empty(core::mem::size_of::<f16>() * 256);

    unsafe {
        cast_matrix_bf16::launch(
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

pub fn test_simple_tf32<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
    if !client.properties().features.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::TF32).into(),
        b_type: ElemType::Float(FloatKind::TF32).into(),
        cd_type: ElemType::Float(FloatKind::F32).into(),
        m: 16,
        k: 8,
        n: 16,
    }) {
        // We can't execute the test, skip.
        return;
    }

    let lhs: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let rhs: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();

    let lhs = client.create_from_slice(f32::as_bytes(&lhs));
    let rhs = client.create_from_slice(f32::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * 256);

    unsafe {
        kernel_simple_tf32::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f32>(&lhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&rhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    let expected = [
        0., 28., 56., 84., 112., 140., 168., 196., 0., 28., 56., 84., 112., 140., 168., 196., 0.,
        92., 184., 276., 368., 460., 552., 644., 0., 92., 184., 276., 368., 460., 552., 644., 0.,
        156., 312., 468., 624., 780., 936., 1092., 0., 156., 312., 468., 624., 780., 936., 1092.,
        0., 220., 440., 660., 880., 1100., 1320., 1540., 0., 220., 440., 660., 880., 1100., 1320.,
        1540., 0., 284., 568., 852., 1136., 1420., 1704., 1988., 0., 284., 568., 852., 1136.,
        1420., 1704., 1988., 0., 348., 696., 1044., 1392., 1740., 2088., 2436., 0., 348., 696.,
        1044., 1392., 1740., 2088., 2436., 0., 412., 824., 1236., 1648., 2060., 2472., 2884., 0.,
        412., 824., 1236., 1648., 2060., 2472., 2884., 0., 476., 952., 1428., 1904., 2380., 2856.,
        3332., 0., 476., 952., 1428., 1904., 2380., 2856., 3332., 0., 540., 1080., 1620., 2160.,
        2700., 3240., 3780., 0., 540., 1080., 1620., 2160., 2700., 3240., 3780., 0., 604., 1208.,
        1812., 2416., 3020., 3624., 4228., 0., 604., 1208., 1812., 2416., 3020., 3624., 4228., 0.,
        668., 1336., 2004., 2672., 3340., 4008., 4676., 0., 668., 1336., 2004., 2672., 3340.,
        4008., 4676., 0., 732., 1464., 2196., 2928., 3660., 4392., 5124., 0., 732., 1464., 2196.,
        2928., 3660., 4392., 5124., 0., 796., 1592., 2388., 3184., 3980., 4776., 5572., 0., 796.,
        1592., 2388., 3184., 3980., 4776., 5572., 0., 860., 1720., 2580., 3440., 4300., 5160.,
        6020., 0., 860., 1720., 2580., 3440., 4300., 5160., 6020., 0., 924., 1848., 2772., 3696.,
        4620., 5544., 6468., 0., 924., 1848., 2772., 3696., 4620., 5544., 6468., 0., 988., 1976.,
        2964., 3952., 4940., 5928., 6916., 0., 988., 1976., 2964., 3952., 4940., 5928., 6916.,
    ];

    assert_eq!(expected, actual);
}

#[cube(launch)]
pub fn kernel_strided(
    lhs: &Array<f16>,
    rhs: &Array<f16>,
    out: &mut Array<f32>,
    #[comptime] stride_lhs: u32,
    #[comptime] stride_rhs: u32,
) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        stride_lhs,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        stride_rhs,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);

    cmma::store(
        &mut out.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

pub fn test_cmma_strided<R: Runtime>(client: ComputeClient<R>, cube_dimensions: CubeDim) {
    // Lhs (row major) will have strided tiles
    let (m, n, k) = (16, 16, 32);
    let (t_m, t_n, t_k) = (16, 16, 16);
    if !client.properties().features.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16).into(),
        b_type: ElemType::Float(FloatKind::F16).into(),
        cd_type: ElemType::Float(FloatKind::F32).into(),
        m: t_m as u32,
        k: t_k as u32,
        n: t_n as u32,
    }) {
        // We can't execute the test, skip.
        return;
    }

    // Fills left tile while right tile is zero
    let lhs: Vec<f16> = (0..m * k)
        .map(|i| {
            if (i % k) < t_k {
                f16::from_f32((i - (i / k) * t_k) as f32)
            } else {
                f16::from_f32(0.)
            }
        })
        .collect();
    let rhs: Vec<f16> = (0..n * k).map(|i| f16::from_f32((i % 8) as f32)).collect();

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * m * n);

    unsafe {
        kernel_strided::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, m * k, 1),
            ArrayArg::from_raw_parts::<f16>(&rhs, k * n, 1),
            ArrayArg::from_raw_parts::<f32>(&out, m * n, 1),
            k as u32,
            n as u32,
        )
    };

    let actual = client.read_one(out);
    let actual = f32::from_bytes(&actual);

    let expected = [
        504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504., 504.,
        504., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400., 1400.,
        1400., 1400., 1400., 1400., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296., 2296.,
        2296., 2296., 2296., 2296., 2296., 2296., 2296., 3192., 3192., 3192., 3192., 3192., 3192.,
        3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 3192., 4088., 4088., 4088.,
        4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088., 4088.,
        4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984., 4984.,
        4984., 4984., 4984., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880., 5880.,
        5880., 5880., 5880., 5880., 5880., 5880., 6776., 6776., 6776., 6776., 6776., 6776., 6776.,
        6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 6776., 7672., 7672., 7672., 7672.,
        7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 7672., 8568.,
        8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568., 8568.,
        8568., 8568., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464., 9464.,
        9464., 9464., 9464., 9464., 9464., 10360., 10360., 10360., 10360., 10360., 10360., 10360.,
        10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 10360., 11256., 11256.,
        11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256., 11256.,
        11256., 11256., 11256., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152.,
        12152., 12152., 12152., 12152., 12152., 12152., 12152., 12152., 13048., 13048., 13048.,
        13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048., 13048.,
        13048., 13048., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944., 13944.,
        13944., 13944., 13944., 13944., 13944., 13944., 13944.,
    ];

    assert_eq!(expected, actual);
}

#[cube(launch)]
pub fn kernel_manual<A: Numeric, B: Numeric, CD: Numeric>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<CD>,
    out: &mut Tensor<CD>,
    #[comptime] size_m: u32,
    #[comptime] size_n: u32,
    #[comptime] size_k: u32,
) {
    let def = cmma::MmaDefinition::<A, B, CD>::new(size_m, size_n, size_k);
    let lane_id = UNIT_POS_PLANE;

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let line_size_a = def.line_size(MatrixIdent::A);
    let line_count_a = comptime!(elem_count_a / line_size_a);
    let mut registers_a = Array::<Line<A>>::vectorized(line_count_a, line_size_a);

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let line_size_b = def.line_size(MatrixIdent::B);
    let line_count_b = comptime!(elem_count_b / line_size_b);
    let mut registers_b = Array::<Line<B>>::vectorized(line_count_b, line_size_b);

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = comptime!(elem_count_c / line_size_c);
    let mut registers_c = Array::<Line<CD>>::vectorized(line_count_c, line_size_c);

    let elem_count_d = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_d = def.line_size(MatrixIdent::Accumulator);
    let line_count_d = comptime!(elem_count_d / line_size_d);

    // Load A
    #[unroll]
    for i in 0..line_count_a {
        let mut reg = Line::empty(line_size_a);
        #[unroll]
        for k in 0..line_size_a {
            let n_elem = i * line_size_a + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::A);
            let value = a[row * size_k + col];
            reg[k] = value;
        }
        registers_a[i] = reg;
    }

    // Load B
    #[unroll]
    for i in 0..line_count_b {
        let mut reg = Line::empty(line_size_b);
        #[unroll]
        for k in 0..line_size_b {
            let n_elem = i * line_size_b + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::B);
            let value = b[row * size_n + col];
            reg[k] = value;
        }
        registers_b[i] = reg;
    }

    // Load C
    #[unroll]
    for i in 0..line_count_c {
        let mut reg = Line::empty(line_size_c);
        #[unroll]
        for k in 0..line_size_c {
            let n_elem = i * line_size_c + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            let value = c[row * size_n + col];
            reg[k] = value;
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D
    #[unroll]
    for i in 0..line_count_d {
        let reg = registers_d[i];
        #[unroll]
        for k in 0..line_size_d {
            let n_elem = i * line_size_d + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            out[row * size_n + col] = reg[k];
        }
    }
}

pub fn test_cmma_manual<
    R: Runtime,
    A: CubeElement + Numeric,
    B: CubeElement + Numeric,
    CD: CubeElement + Numeric,
>(
    client: ComputeClient<R>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
) {
    if !client.properties().features.mma.contains(&MmaConfig {
        a_type: A::cube_type(),
        b_type: B::cube_type(),
        cd_type: CD::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    }) {
        // We can't execute the test, skip.
        println!(
            "Skipping test for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            CD::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..k).map(move |j| A::from_int((i * 2 + j) as i64)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index
    let rhs: Vec<B> = (0..k)
        .flat_map(|i| (0..n).map(move |j| B::from_int((i * 3 + j) as i64)))
        .collect();
    let acc = vec![CD::from_int(0); m * n];

    let lhs = client.create_from_slice(A::as_bytes(&lhs));
    let rhs = client.create_from_slice(B::as_bytes(&rhs));
    let out = client.create_from_slice(CD::as_bytes(&acc));

    unsafe {
        kernel_manual::launch::<A, B, CD, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<A>(&lhs, &[k, 1], &[m, k], 1),
            TensorArg::from_raw_parts::<B>(&rhs, &[n, 1], &[k, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
        )
    };

    let actual = client.read_one(out);
    let actual = CD::from_bytes(&actual);

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        // For each output row
        for j in 0..n {
            // For each output column
            // For each output column
            let mut sum = 0;
            for l in 0..k {
                // Dot product over k-dimension
                let lhs_val = (i * 2 + l) as i64; // LHS[i, l]
                let rhs_val = (l * 3 + j) as i64; // RHS[l, j]
                sum += lhs_val * rhs_val;
            }
            expected.push(CD::from_int(sum));
        }
    }

    // Need tolerance for slight differences because CPU integer version isn't exactly the same
    // as GPU MMA for fp8. 3% tolerance seems to work for both FP8 types.
    // Existing approximate comparison requires `Float`, so just do a simple one here.
    for (i, (expected_val, actual_val)) in expected.iter().zip(actual).enumerate() {
        let expected_val = expected_val.to_f64().unwrap();
        let actual_val = actual_val.to_f64().unwrap();
        let difference = (expected_val - actual_val).abs();
        let max_difference = expected_val * 0.03;
        if difference > max_difference {
            panic!(
                "Expected != actual at position {i}: (expected: {expected_val}, actual: {actual_val}, difference: {difference}, max_difference: {max_difference})"
            )
        }
    }
}

// Kinda hardcoded for f16 right now, but it's hard to make generic
#[cube(launch)]
pub fn kernel_manual_ldmatrix<AB: Numeric, CD: Numeric>(
    a: &Tensor<Line<AB>>,
    b: &Tensor<Line<AB>>,
    c: &Tensor<CD>,
    out: &mut Tensor<CD>,
    #[comptime] size_m: u32,
    #[comptime] size_n: u32,
    #[comptime] size_k: u32,
) {
    let bar = Barrier::new(BarrierLevel::cube_full(UNIT_POS == 0));
    let def = cmma::MmaDefinition::<AB, AB, CD>::new(size_m, size_n, size_k);
    let lane_id = UNIT_POS_PLANE;

    let elem_size = AB::type_size();
    let width = comptime![16 / elem_size];

    let mut stage_a = SharedMemory::new_aligned(size_m * size_k, 1u32, 16u32);
    let mut stage_b = SharedMemory::new_aligned(size_k * size_n, 1u32, 16u32);
    bar.memcpy_async_cooperative(&a.to_slice(), &mut stage_a.to_slice_mut());
    bar.memcpy_async_cooperative(&b.to_slice(), &mut stage_b.to_slice_mut());
    bar.arrive_and_wait();

    let row = lane_id % 16;

    let col_a = (lane_id / 16) * width;
    let start_a = row * size_k + col_a;
    let slice_a = stage_a.slice(start_a, start_a + width);
    let line_count_a = def.lines_per_lane(MatrixIdent::A);

    let registers_a = def.load_matrix(&slice_a, MatrixIdent::A, line_count_a, false);

    // B frags are only 2 registers, so top 16 threads do nothing
    let col_b = 0;
    let start_b = row * size_n + col_b;
    let slice_b = stage_b.slice(start_b, start_b + width);
    let line_count_b = def.lines_per_lane(MatrixIdent::B);

    let registers_b = def.load_matrix(&slice_b, MatrixIdent::B, line_count_b, true);

    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = def.lines_per_lane(MatrixIdent::Accumulator);
    let mut registers_c = Array::<Line<CD>>::vectorized(line_count_c, line_size_c);

    let line_size_d = def.line_size(MatrixIdent::Accumulator);
    let line_count_d = def.lines_per_lane(MatrixIdent::Accumulator);

    // Load C
    #[unroll]
    for i in 0..line_count_c {
        let mut reg = Line::empty(line_size_c);
        #[unroll]
        for k in 0..line_size_c {
            let n_elem = i * line_size_c + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            let value = c[row * size_n + col];
            reg[k] = value;
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D
    #[unroll]
    for i in 0..line_count_d {
        let reg = registers_d[i];
        #[unroll]
        for k in 0..line_size_d {
            let n_elem = i * line_size_d + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            out[row * size_n + col] = reg[k];
        }
    }
}

pub fn test_cmma_manual_ldmatrix<
    R: Runtime,
    AB: CubeElement + Numeric,
    CD: CubeElement + Numeric,
>(
    client: ComputeClient<R>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
) {
    if !client.properties().features.mma.contains(&MmaConfig {
        a_type: AB::cube_type(),
        b_type: AB::cube_type(),
        cd_type: CD::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    }) {
        // We can't execute the test, skip.
        println!(
            "Skipping test for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            AB::cube_type(),
            AB::cube_type(),
            CD::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs: Vec<AB> = (0..m)
        .flat_map(|i| (0..k).map(move |j| AB::from_int((i * 2 + j) as i64)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index
    let rhs: Vec<AB> = (0..k)
        .flat_map(|i| (0..n).map(move |j| AB::from_int((i * 3 + j) as i64)))
        .collect();
    let acc = vec![CD::from_int(0); m * n];

    let lhs = client.create_from_slice(AB::as_bytes(&lhs));
    let rhs = client.create_from_slice(AB::as_bytes(&rhs));
    let out = client.create_from_slice(CD::as_bytes(&acc));

    unsafe {
        kernel_manual_ldmatrix::launch::<AB, CD, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<AB>(&lhs, &[k, 1], &[m, k], 1),
            TensorArg::from_raw_parts::<AB>(&rhs, &[n, 1], &[k, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
        )
    };

    let actual = client.read_one(out);
    let actual = CD::from_bytes(&actual);

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        // For each output row
        for j in 0..n {
            // For each output column
            // For each output column
            let mut sum = 0;
            for l in 0..k {
                // Dot product over k-dimension
                let lhs_val = (i * 2 + l) as i64; // LHS[i, l]
                let rhs_val = (l * 3 + j) as i64; // RHS[l, j]
                sum += lhs_val * rhs_val;
            }
            expected.push(CD::from_int(sum));
        }
    }

    // Need tolerance for slight differences because CPU integer version isn't exactly the same
    // as GPU MMA for fp8. 3% tolerance seems to work for both FP8 types.
    // Existing approximate comparison requires `Float`, so just do a simple one here.
    for (i, (expected_val, actual_val)) in expected.iter().zip(actual).enumerate() {
        let expected_val = expected_val.to_f64().unwrap();
        let actual_val = actual_val.to_f64().unwrap();
        let difference = (expected_val - actual_val).abs();
        let max_difference = expected_val * 0.03;
        if difference > max_difference {
            panic!(
                "Expected != actual at position {i}: (expected: {expected_val}, actual: {actual_val}, difference: {difference}, max_difference: {max_difference})"
            )
        }
    }
}

#[cube(launch)]
pub fn kernel_scaled<A: CubePrimitive, B: CubePrimitive, CD: Numeric, S: Numeric>(
    a: &Tensor<Line<A>>,
    b: &Tensor<Line<B>>,
    c: &Tensor<Line<CD>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Line<CD>>,
    #[comptime] size_m: u32,
    #[comptime] size_n: u32,
    #[comptime] size_k: u32,
    #[comptime] scales_factor: u32,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();

    let def =
        cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(size_m, size_n, size_k, scales_factor);
    let lane_id = UNIT_POS_PLANE;

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let line_size_a = def.line_size(MatrixIdent::A);
    let line_count_a = comptime!(elem_count_a / line_size_a);
    let mut registers_a = Array::<Line<A>>::vectorized(line_count_a, line_size_a);

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let line_size_b = def.line_size(MatrixIdent::B);
    let line_count_b = comptime!(elem_count_b / line_size_b);
    let mut registers_b = Array::<Line<B>>::vectorized(line_count_b, line_size_b);

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = comptime!(elem_count_c / line_size_c);
    let mut registers_c = Array::<Line<CD>>::vectorized(line_count_c, line_size_c);

    let elem_count_d = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_d = def.line_size(MatrixIdent::Accumulator);
    let line_count_d = comptime!(elem_count_d / line_size_d);

    let scales_count = def.scales_count();
    let mut scales_register_a = Line::<S>::empty(def.scales_line_size());
    let mut scales_register_b = Line::<S>::empty(def.scales_line_size());

    // Load A
    #[unroll]
    for i in 0..line_count_a {
        let n_elem = i * line_size_a * a_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::A);
        let idx = row * size_k + col;
        let idx = idx / (a.line_size() * a_pack);

        registers_a[i] = a[idx];
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a * scales_factor + i];
    }

    // Load B
    #[unroll]
    for i in 0..line_count_b {
        let n_elem = i * line_size_b * b_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::B);
        let idx = col * size_k + row;
        let idx = idx / (b.line_size() * b_pack);

        registers_b[i] = b[idx];
    }

    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b * scales_factor + i];
    }

    // Load C
    #[unroll]
    for i in 0..line_count_c {
        let n_elem = i * line_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
        let idx = row * size_n + col;
        let value = c[idx / c.line_size()];
        registers_c[i] = value;
    }

    let registers_d = def.execute_scaled(
        &registers_a,
        &registers_b,
        &registers_c,
        scales_register_a,
        scales_register_b,
    );

    // Store D
    #[unroll]
    for i in 0..line_count_d {
        let n_elem = i * line_size_d;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
        let idx = row * size_n + col;
        out[idx / out.line_size()] = registers_d[i];
    }
}

pub fn test_cmma_scaled<R: Runtime, A: CubeElement + Numeric, B: CubeElement + Numeric>(
    client: ComputeClient<R>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type S = ue8m0;

    let a_elem = A::cube_type();
    let b_elem = B::cube_type();
    let a_line_size = 32 / a_elem.size_bits();
    let b_line_size = 32 / b_elem.size_bits();

    if !client
        .properties()
        .features
        .scaled_mma
        .contains(&ScaledMmaConfig {
            a_type: a_elem,
            b_type: b_elem,
            cd_type: f32::cube_type(),
            scales_type: S::cube_type(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: scales_factor as u32,
        })
    {
        // We can't execute the test, skip.
        println!(
            "Skipping test for a: {:?}, b: {:?}, scales: {:?} m: {m}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            S::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..k).map(move |j| A::from_int((i * 2 + j) as i64)))
        .collect();
    let lhs_scales: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| S::from_bits((i * 2 + j + 120) as u8)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs: Vec<B> = (0..n)
        .flat_map(|j| (0..k).map(move |i| B::from_int((i * 3 + j) as i64)))
        .collect();
    let rhs_scales: Vec<S> = (0..n)
        .flat_map(|j| (0..scales_factor).map(move |i| S::from_bits((i * 3 + j + 120) as u8)))
        .collect();

    let out: Vec<f32> = vec![0.0; m * n];

    let lhs = client.create_from_slice(A::as_bytes(&lhs));
    let lhs_scales = client.create_from_slice(S::as_bytes(&lhs_scales));
    let rhs = client.create_from_slice(B::as_bytes(&rhs));
    let rhs_scales = client.create_from_slice(S::as_bytes(&rhs_scales));
    let out = client.create_from_slice(f32::as_bytes(&out));

    unsafe {
        kernel_scaled::launch::<A, B, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<A>(&lhs, &[k, 1], &[m, k], a_line_size as u8),
            TensorArg::from_raw_parts::<B>(&rhs, &[k, 1], &[n, k], b_line_size as u8),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<S>(
                &lhs_scales,
                &[scales_factor, 1],
                &[m, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<S>(
                &rhs_scales,
                &[scales_factor, 1],
                &[n, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
            scales_factor as u32,
        )
    };

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        for j in 0..n {
            // For each output column
            let mut sum = 0.0;
            for l in 0..k {
                let l_scales = l / (k / scales_factor);

                // Dot product over k-dimension
                let lhs_val = (i * 2 + l) as f32; // LHS[i, l]
                let lhs_scale = ue8m0::from_bits((i * 2 + l_scales + 120) as u8).to_f32();
                let rhs_val = (l * 3 + j) as f32; // RHS[l, j]
                let rhs_scale = ue8m0::from_bits((l_scales * 3 + j + 120) as u8).to_f32();
                sum += lhs_val * lhs_scale * rhs_val * rhs_scale;
            }
            expected.push(sum);
        }
    }

    assert_equals_approx::<R, f32>(&client, out, &expected, 0.03);
}

pub fn test_cmma_scaled_fp4<R: Runtime>(
    client: ComputeClient<R>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type AB = e2m1x2;
    type S = ue8m0;

    let ab_elem = AB::cube_type();
    let ab_line_size = 32 / ab_elem.size_bits();

    if !client
        .properties()
        .features
        .scaled_mma
        .contains(&ScaledMmaConfig {
            a_type: ab_elem,
            b_type: ab_elem,
            cd_type: f32::cube_type(),
            scales_type: S::cube_type(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: scales_factor as u32,
        })
    {
        // We can't execute the test, skip.
        println!(
            "Skipping test for ab: {:?}, scales: {:?} m: {m}, n: {n}, k: {k}",
            AB::cube_type(),
            S::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs_data: Vec<f32> = (0..m)
        .flat_map(|i| (0..k).map(move |j| e2m1::from_bits(((i + j) % 15) as u8 + 1).to_f32()))
        .collect();
    //println!("lhs: {lhs_data:?}");
    let lhs = e2m1x2::from_f32_slice(&lhs_data);
    let lhs_scales_data: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| S::from_bits((i * 2 + j + 120) as u8)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs_data: Vec<f32> = (0..n)
        .flat_map(|j| (0..k).map(move |i| e2m1::from_bits(((i + j) % 15) as u8 + 1).to_f32()))
        .collect();
    let rhs = e2m1x2::from_f32_slice(&rhs_data);
    let rhs_scales_data: Vec<S> = (0..n)
        .flat_map(|j| (0..scales_factor).map(move |i| S::from_bits((i * 3 + j + 120) as u8)))
        .collect();

    let out = vec![0.0; m * n];

    let lhs = client.create_from_slice(AB::as_bytes(&lhs));
    let lhs_scales = client.create_from_slice(S::as_bytes(&lhs_scales_data));
    let rhs = client.create_from_slice(AB::as_bytes(&rhs));
    let rhs_scales = client.create_from_slice(S::as_bytes(&rhs_scales_data));
    let out = client.create_from_slice(f32::as_bytes(&out));

    unsafe {
        kernel_scaled::launch::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<AB>(&lhs, &[k / 2, 1], &[m, k / 2], ab_line_size as u8),
            TensorArg::from_raw_parts::<AB>(&rhs, &[k / 2, 1], &[n, k / 2], ab_line_size as u8),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<S>(
                &lhs_scales,
                &[scales_factor, 1],
                &[m, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<S>(
                &rhs_scales,
                &[scales_factor, 1],
                &[n, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
            scales_factor as u32,
        )
    };

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        for j in 0..n {
            // For each output column
            let mut sum = 0.0;
            for l in 0..k {
                let l_scales = l / (k / scales_factor);

                // Dot product over k-dimension
                let lhs_val = lhs_data[i * k + l]; // LHS[i, l]
                let lhs_scale = lhs_scales_data[i * scales_factor + l_scales].to_f32();
                let rhs_val = rhs_data[j * k + l];
                let rhs_scale = rhs_scales_data[j * scales_factor + l_scales].to_f32();
                sum += lhs_val * lhs_scale * rhs_val * rhs_scale;
            }
            expected.push(sum);
        }
    }

    assert_equals_approx::<R, f32>(&client, out, &expected, 0.03);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[test]
        fn test_cmma_simple_1() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1::<TestRuntime>(client, cube_dimensions);
        }

        #[test]
        fn test_cmma_simple_1_lined() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_lined::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_simple_1_lined_offset() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_lined_offset::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_simple_tf32() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_tf32::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_f16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_f16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_bf16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_bf16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_strided() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_strided::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_manual() {
            use cubecl_common::*;
            use half::{bf16, f16};

            fn test<
                A: CubeElement + Numeric,
                B: CubeElement + Numeric,
                CD: CubeElement + Numeric,
            >(
                m: usize,
                n: usize,
                k: usize,
            ) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_manual::<TestRuntime, A, B, CD>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                )
            }

            // CUDA
            test::<tf32, tf32, f32>(16, 8, 8);
            test::<f16, f16, f32>(16, 8, 16);
            test::<bf16, bf16, f32>(16, 8, 16);
            test::<e5m2, e5m2, f32>(16, 8, 32);
            test::<e4m3, e4m3, f32>(16, 8, 32);
            test::<e5m2, e4m3, f32>(16, 8, 32);
            test::<e4m3, e5m2, f32>(16, 8, 32);
            test::<i8, i8, i32>(16, 8, 32);
            test::<i8, u8, i32>(16, 8, 32);
            test::<u8, u8, i32>(16, 8, 32);
            test::<u8, i8, i32>(16, 8, 32);

            // HIP
            test::<f16, f16, f32>(16, 16, 16);
            // bf16 is broken in general right now, it generates a conflicting `__bf16_2` type
            //test::<bf16, bf16, f32>(16, 16, 16);
        }

        #[test]
        fn test_cmma_manual_ldmatrix() {
            use cubecl_common::*;
            use half::{bf16, f16};

            fn test<AB: CubeElement + Numeric, CD: CubeElement + Numeric>(
                m: usize,
                n: usize,
                k: usize,
            ) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_manual_ldmatrix::<TestRuntime, AB, CD>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                )
            }

            // CUDA
            test::<f16, f32>(16, 8, 16);
            test::<bf16, f32>(16, 8, 16);
        }

        #[test]
        fn test_cmma_scaled() {
            use cubecl_common::*;

            fn test<A: CubeElement + Numeric, B: CubeElement + Numeric>(
                m: usize,
                n: usize,
                k: usize,
                factor: usize,
            ) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_scaled::<TestRuntime, A, B>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                    factor,
                )
            }

            // FP4 needs more design for transferring properly as packed values
            test::<e5m2, e5m2>(16, 8, 32, 1);
            test::<e4m3, e4m3>(16, 8, 32, 1);
            test::<e5m2, e4m3>(16, 8, 32, 1);
            test::<e4m3, e5m2>(16, 8, 32, 1);
        }

        #[test]
        fn test_cmma_scaled_fp4() {
            use cubecl_common::*;

            fn test(m: usize, n: usize, k: usize, factor: usize) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_scaled_fp4::<TestRuntime>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                    factor,
                )
            }

            // FP4 needs more design for transferring properly as packed values
            test(16, 8, 64, 2);
        }

        fn cube_dim<R: Runtime>(client: &ComputeClient<R>) -> CubeDim {
            let plane_dim = client.properties().hardware.plane_size_max;
            CubeDim::new(plane_dim, 1, 1)
        }
    };
}
