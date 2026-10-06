use cubecl_runtime::runtime::Runtime;
use std::println;

use crate::{
    self as cubecl, cmma::Cube, prelude::barrier::Barrier,
    runtime_tests::binary::assert_equals_approx,
};

use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
};

use alloc::{vec, vec::Vec};
use cubecl_common::{e2m1, e2m1x2, e4m3, ue8m0};
use cubecl_ir::{
    OpaqueType,
    features::{MmaConfig, ScaledMmaConfig},
    types::{MatrixIdent, MatrixLayout},
};
use half::{bf16, f16};
use num_traits::NumCast;

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1_vectorized<N: Size>(
    lhs: &[Vector<f16, N>],
    rhs: &[Vector<f16, N>],
    out: &mut [Vector<f32, N>],
) {
    let a = cmma::Matrix::<Vector<f16, N>>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        16,
    );
    let b = cmma::Matrix::<Vector<f16, N>>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1_vectorized_offset<N: Size>(
    lhs: &[Vector<f16, N>],
    rhs: &[Vector<f16, N>],
    out: &mut [Vector<f32, N>],
    offset_lhs: usize,
    offset_rhs: usize,
    offset_out: usize,
) {
    let len_lhs = lhs.len();
    let len_rhs = rhs.len();
    let len_out = out.len();

    let a = cmma::Matrix::<Vector<f16, N>>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &lhs[offset_lhs..len_lhs],
        16,
    );
    let b = cmma::Matrix::<Vector<f16, N>>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        &rhs[offset_rhs..len_rhs],
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(
        &mut out[offset_out..len_out],
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_2(lhs: &[f16], rhs: &[f16], out: &mut [f16]) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        8,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        8,
    );
    let c = cmma::Matrix::<f16>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        half::f16::from_int(0),
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 8, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_3(lhs: &[f16], rhs: &[f16], out: &mut [f32]) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        8,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        8,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 8, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_tf32(lhs: &[tf32], rhs: &[tf32], out: &mut [f32]) {
    let a = cmma::Matrix::<tf32>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        8usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        8,
    );
    let b = cmma::Matrix::<tf32>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        8usize,
        cmma::MatrixLayout::RowMajor,
        rhs,
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_f16_workgroup_gmem(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f32],
    #[comptime] size: (usize, usize, usize),
) {
    let (m, n, k) = size;

    let a = cmma::Matrix::<f16, Cube>::from_slice(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        lhs,
        k as u32,
    );
    let b = cmma::Matrix::<f16, Cube>::from_slice(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        rhs,
        k as u32,
    );
    let c = cmma::Matrix::<f32, Cube>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, n as u32, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_f16_workgroup_tensor(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f32],
    size_k: u32,
    #[comptime] size: (usize, usize, usize),
) {
    let (m, n, k) = size;

    let lhs = TensorView::new(lhs, seq![m as u32, size_k]).finish();
    let rhs = TensorView::new(rhs, seq![n as u32, size_k]).finish();
    let mut out = TensorView::new(&*out, seq![m as u32, n as u32]).finish();

    let mut a = unsafe {
        cmma::Matrix::<f16, Cube>::uninitialized(
            cmma::MatrixIdent::A,
            m,
            n,
            k,
            MatrixLayout::Undefined,
        )
    };
    let mut b = unsafe {
        cmma::Matrix::<f16, Cube>::uninitialized(
            cmma::MatrixIdent::B,
            m,
            n,
            k,
            MatrixLayout::Undefined,
        )
    };
    let c = cmma::Matrix::<f32, Cube>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    for k_offs in range_stepped(0, size_k, k as u32) {
        let lhs = lhs.slice(seq![0, k_offs], seq![m as u32, k as u32]);
        let rhs = rhs.slice(seq![0, k_offs], seq![n as u32, k as u32]);
        let rhs = rhs.permuted(seq![1, 0]);

        cmma::load_tensor(&mut a, &lhs);
        cmma::load_tensor(&mut b, &rhs);

        cmma::execute(&a, &b, &c, &c);
    }

    cmma::store_tensor(&mut out, &c);
}

#[cube(launch)]
pub fn cast_matrix_f16(input: &[f32], out: &mut [f16]) {
    let mut acc = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
        )
    };
    cmma::load_with_layout(&mut acc, input, 16, cmma::MatrixLayout::RowMajor);

    let output: cmma::Matrix<f16> = cmma::cast(&acc);

    cmma::store(out, &output, 16, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
pub fn cast_matrix_bf16(input: &[f32], out: &mut [bf16]) {
    let mut acc = unsafe {
        cmma::Matrix::<f32>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
        )
    };
    cmma::load_with_layout(&mut acc, input, 16, cmma::MatrixLayout::RowMajor);

    let output: cmma::Matrix<bf16> = cmma::cast(&acc);

    cmma::store(out, &output, 16, cmma::MatrixLayout::RowMajor);
}

pub fn test_simple_1_vectorized<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
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
        kernel_simple_1_vectorized::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            4,
            BufferArg::from_raw_parts(lhs, 256 / 4),
            BufferArg::from_raw_parts(rhs, 256 / 4),
            BufferArg::from_raw_parts(out.clone(), 256 / 4),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_1_expected(), actual);
}

pub fn test_simple_1_vectorized_offset<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
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
    let vector_size = 2usize;

    let lhs: Vec<f16> = (0..256 + offset_lhs * vector_size)
        .map(|i| f16::from_f32(i as f32 - (offset_lhs * vector_size) as f32))
        .collect();
    let rhs: Vec<f16> = (0..256i32 + (offset_rhs * vector_size) as i32)
        .map(|i| f16::from_f32(((i - (offset_rhs * vector_size) as i32) % 8) as f32))
        .collect();

    let lhs_len = lhs.len() / vector_size;
    let rhs_len = rhs.len() / vector_size;
    let out_len = (256 / vector_size) + offset_out;

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * vector_size * out_len);

    unsafe {
        kernel_simple_1_vectorized_offset::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            vector_size,
            BufferArg::from_raw_parts(lhs, lhs_len),
            BufferArg::from_raw_parts(rhs, rhs_len),
            BufferArg::from_raw_parts(out.clone(), out_len),
            offset_lhs,
            offset_rhs,
            offset_out,
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(
        test_simple_1_expected(),
        actual[(offset_out * vector_size)..actual.len()]
    );
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T with the accumulator declaring a layout.
///
/// An accumulator's layout is its own and never reaches the product, so declaring
/// one must compile and change nothing.
pub fn kernel_accumulator_row_major(lhs: &[f16], rhs: &[f16], out: &mut [f32]) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        16,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}

pub fn test_accumulator_row_major<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
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
        kernel_accumulator_row_major::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            BufferArg::from_raw_parts(lhs, 256),
            BufferArg::from_raw_parts(rhs, 256),
            BufferArg::from_raw_parts(out.clone(), 256),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_1_expected(), actual);
}

pub fn test_simple_1<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    test_simple_1_typed::<R, f16, f32>(client, cube_dimensions);
}

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T, with `I` operands accumulating in `A`.
pub fn kernel_simple_m16n16k16_gmem<I: Float, A: Float>(lhs: &[I], rhs: &[I], out: &mut [A]) {
    let a = cmma::Matrix::<I>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        16,
    );
    let b = cmma::Matrix::<I>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        16,
    );
    let c = cmma::Matrix::<A>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        A::from_int(0),
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}

/// [`test_simple_1`] on `I` operands accumulating in `A`, where the device has that form.
/// The operands are small integers, exact in any half-precision type; the products are summed
/// in `A`, so a 16-bit accumulator is checked to its own precision.
pub fn test_simple_1_typed<R: Runtime, I: Float + CubeElement, A: Float + CubeElement>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: I::cube_type(),
        b_type: I::cube_type(),
        cd_type: A::cube_type(),
        m: 16,
        k: 16,
        n: 16,
    }) {
        println!("Unsupported, skipping");
        return;
    }

    let lhs: Vec<I> = (0..256).map(|i| I::from_int(i)).collect();
    let rhs: Vec<I> = (0..256).map(|i| I::from_int(i % 8)).collect();

    let lhs = client.create_from_slice(I::as_bytes(&lhs));
    let rhs = client.create_from_slice(I::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<A>() * 256);

    kernel_simple_m16n16k16_gmem::launch::<I, A>(
        &client,
        CubeCount::Static(1, 1, 1),
        cube_dimensions,
        unsafe { BufferArg::from_raw_parts(lhs, 256) },
        unsafe { BufferArg::from_raw_parts(rhs, 256) },
        unsafe { BufferArg::from_raw_parts(out.clone(), 256) },
    );

    let actual = client.read_one_unchecked(out);
    let actual = A::from_bytes(&actual);
    assert_eq!(actual.len(), 256, "a failed launch reads back nothing");

    let epsilon = A::EPSILON.to_f32().unwrap();
    for (i, (expected, actual)) in test_simple_1_expected().into_iter().zip(actual).enumerate() {
        let actual = actual.to_f32().unwrap();
        // Sixteen roundings of a running sum, each within half an ulp of it.
        let tolerance = expected * epsilon * 8.0;
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {i}: expected {expected}, got {actual}"
        );
    }
}

#[cube(launch)]
pub fn kernel_unsupported_fragment(out: &mut [f32]) {
    let acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        3usize,
        5usize,
        7usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    cmma::store(out, &acc, 5, cmma::MatrixLayout::RowMajor);
}

pub fn test_unsupported_fragment<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    let out = client.empty(core::mem::size_of::<f32>() * 16);

    kernel_unsupported_fragment::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        cube_dimensions,
        unsafe { BufferArg::from_raw_parts(out.clone(), 16) },
    );

    // The compile failure claims the buffer the launch was given; reading it
    // is what reports the error — a launch failure is not the flush's to
    // report any more.
    let error = client
        .read_one(out)
        .expect_err("a fragment shape no device advertises must fail compilation");
    let message = alloc::format!("{error:?}");
    assert!(
        message.contains("cooperative matrix fragment"),
        "expected a cooperative matrix validation error, got: {message}"
    );
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

pub fn test_simple_cube<R: Runtime>(client: Client, cube_dimensions: u32) {
    let ab_ty = ElemType::Float(FloatKind::F16);
    let cd_ty = ElemType::Float(FloatKind::F32);
    let config = client.features().matmul.cube_mma.iter().find(|cfg| {
        cfg.a_type == ab_ty
            && cfg.b_type == ab_ty
            && cfg.cd_type == cd_ty
            && cfg
                .units_per_block
                .map(|it| it == cube_dimensions)
                .unwrap_or(true)
    });
    if config.is_none() {
        // We can't execute the test, skip.
        println!("No valid config for cube matmul, skipping");
        return;
    }

    let config = config.unwrap();
    println!("Running config {config:?} with cube dim {cube_dimensions}");

    let m = config.m_granularity as usize;
    let n = config.n_granularity as usize;
    let k = config.k_granularity as usize;

    let lhs_size = m * k;
    let rhs_size = k * n;
    let acc_size = m * n;

    let lhs: Vec<f16> = (0..lhs_size).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..rhs_size)
        .map(|i| f16::from_f32((i % 8) as f32))
        .collect();

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * acc_size);

    unsafe {
        kernel_simple_f16_workgroup_gmem::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(cube_dimensions),
            BufferArg::from_raw_parts(lhs, lhs_size),
            BufferArg::from_raw_parts(rhs, rhs_size),
            BufferArg::from_raw_parts(out.clone(), acc_size),
            (m, n, k),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_cube_expected(m, n, k), actual);
}

pub fn test_simple_cube_tensor<R: Runtime>(client: Client, cube_dimensions: u32) {
    let ab_ty = ElemType::Float(FloatKind::F16);
    let cd_ty = ElemType::Float(FloatKind::F32);
    let config = client.features().matmul.cube_mma.iter().find(|cfg| {
        cfg.a_type == ab_ty
            && cfg.b_type == ab_ty
            && cfg.cd_type == cd_ty
            && cfg
                .units_per_block
                .map(|it| it == cube_dimensions)
                .unwrap_or(true)
    });
    if config.is_none() {
        // We can't execute the test, skip.
        log::info!("No valid config for cube matmul, skipping");
        return;
    }

    let config = config.unwrap();
    log::info!("Running config {config:?} with cube dim {cube_dimensions}");

    let m = config.m_granularity as usize;
    let n = config.n_granularity as usize;
    let k = config.k_granularity as usize;

    let k_chunks = 2;

    let size_k = k * k_chunks as usize;

    let lhs_size = m * size_k;
    let rhs_size = size_k * n;
    let acc_size = m * n;

    let lhs: Vec<f16> = (0..lhs_size).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..rhs_size)
        .map(|i| f16::from_f32((i % 8) as f32))
        .collect();

    let lhs = client.create_from_slice(f16::as_bytes(&lhs));
    let rhs = client.create_from_slice(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * acc_size);

    unsafe {
        kernel_simple_f16_workgroup_tensor::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(cube_dimensions),
            BufferArg::from_raw_parts(lhs, lhs_size),
            BufferArg::from_raw_parts(rhs, rhs_size),
            BufferArg::from_raw_parts(out.clone(), acc_size),
            size_k as u32,
            (m, n, k),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_simple_cube_expected(m, n, size_k), actual);
}

pub fn test_simple_cube_expected(m: usize, n: usize, k: usize) -> Vec<f32> {
    let lhs_size = m * k;
    let rhs_size = k * n;
    let acc_size = m * n;

    let lhs: Vec<f16> = (0..lhs_size).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..rhs_size)
        .map(|i| f16::from_f32((i % 8) as f32))
        .collect();
    let mut out = vec![0f32; acc_size];

    for m in 0..m {
        let lhs_offs = m * k;
        let out_offs = m * n;
        for n in 0..n {
            let rhs_offs = n * k;
            let mut sum = 0f32;
            for k in 0..k {
                let lhs = lhs[lhs_offs + k].to_f32();
                let rhs = rhs[rhs_offs + k].to_f32();
                sum += lhs * rhs;
            }
            out[out_offs + n] = sum;
        }
    }

    out
}

// pub fn test_simple_2<R: Runtime>(
//     client: Client,
//     cube_dimensions: CubeDim,
// ) {
//     if !client.features().matmul.cmma.contains(&MmaConfig {
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

//     let lhs = client.create_from_slice(f16::as_bytes(&lhs));
//     let rhs = client.create_from_slice(f16::as_bytes(&rhs));
//     let out = client.empty(core::mem::size_of::<f16>() * 64);

//     unsafe {
//         kernel_simple_2::launch(
//             &client,
//             CubeCount::Static(1, 1, 1),
//             cube_dimensions,
//             BufferArg::from_raw_parts::<f16>(&lhs, 64, 1),
//             BufferArg::from_raw_parts::<f16>(&rhs, 64, 1),
//             BufferArg::from_raw_parts::<f16>(&out, 64, 1),
//         )
//     };

//     let actual = client.read_one_unchecked(out);
//     let actual = f16::from_bytes(&actual);

//     let expected: [f16; 64] = [0.0, 28.0, 56.0, 84.0, 112.0, 140.0, 168.0, 196.0, 0.0, 92.0, 184.0, 276.0, 368.0, 460.0, 552.0, 644.0, 0.0, 156.0, 312.0, 468.0, 624.0, 780.0, 936.0, 1092.0, 0.0, 220.0, 440.0, 660.0, 880.0, 1100.0, 1320.0, 1540.0, 0.0, 284.0, 568.0, 852.0, 1136.0, 1420.0, 1704.0, 1988.0, 0.0, 348.0, 696.0, 1044.0, 1392.0, 1740.0, 2088.0, 2436.0, 0.0, 412.0, 824.0, 1236.0, 1648.0, 2060.0, 2472.0, 2884.0, 0.0, 476.0, 952.0, 1428.0, 1904.0, 2380.0, 2856.0, 3332.0].map(|e| f16::from_f64(e));

//     assert_eq!(expected, actual);
// }

pub fn test_cmma_cast_f16<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
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
            BufferArg::from_raw_parts(input, 256),
            BufferArg::from_raw_parts(out.clone(), 256),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f16::from_bytes(&actual);
    let expected: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}

/// Casting an `f32` accumulator to `bf16` needs a `bf16` accumulator fragment, which only some
/// devices have (RDNA's WMMA does, NVIDIA's does not).
pub fn test_cmma_cast_bf16<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::BF16),
        b_type: ElemType::Float(FloatKind::BF16),
        cd_type: ElemType::Float(FloatKind::BF16),
        m: 16,
        k: 16,
        n: 16,
    }) {
        println!("Unsupported, skipping");
        return;
    }

    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let input = client.create_from_slice(f32::as_bytes(&input));
    let out = client.empty(core::mem::size_of::<bf16>() * 256);

    unsafe {
        cast_matrix_bf16::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            BufferArg::from_raw_parts(input, 256),
            BufferArg::from_raw_parts(out.clone(), 256),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = bf16::from_bytes(&actual);
    let expected: Vec<bf16> = (0..256).map(|i| bf16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}

pub fn test_simple_tf32<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::TF32),
        b_type: ElemType::Float(FloatKind::TF32),
        cd_type: ElemType::Float(FloatKind::F32),
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
            BufferArg::from_raw_parts(lhs, 128),
            BufferArg::from_raw_parts(rhs, 128),
            BufferArg::from_raw_parts(out.clone(), 256),
        )
    };

    let actual = client.read_one_unchecked(out);
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
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f32],
    #[comptime] stride_lhs: u32,
    #[comptime] stride_rhs: u32,
) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        stride_lhs,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        stride_rhs,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );

    cmma::execute(&a, &b, &c, &c);

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}

pub fn test_cmma_strided<R: Runtime>(client: Client, cube_dimensions: CubeDim) {
    // Lhs (row major) will have strided tiles
    let (m, n, k) = (16, 16, 32);
    let (t_m, t_n, t_k) = (16, 16, 16);
    if !client.features().matmul.cmma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
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
            BufferArg::from_raw_parts(lhs, m * k),
            BufferArg::from_raw_parts(rhs, k * n),
            BufferArg::from_raw_parts(out.clone(), m * n),
            k as u32,
            n as u32,
        )
    };

    let actual = client.read_one_unchecked(out);
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
pub fn kernel_manual<A: Scalar, B: Scalar, CD: Numeric>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<CD>,
    out: &mut Tensor<CD>,
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
) {
    let def = cmma::MmaDefinition::<A, B, CD>::new(size_m, size_n, size_k);
    let lane_id = UNIT_POS_PLANE;

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let size!(NA) = vector_size_a;
    let vector_count_a = comptime!(elem_count_a / vector_size_a);
    let mut registers_a = Array::<Vector<A, NA>>::new(vector_count_a);

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let size!(NB) = vector_size_b;
    let vector_count_b = comptime!(elem_count_b / vector_size_b);
    let mut registers_b = Array::<Vector<B, NB>>::new(vector_count_b);

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let size!(NC) = vector_size_c;
    let vector_count_c = comptime!(elem_count_c / vector_size_c);
    let mut registers_c = Array::<Vector<CD, NC>>::new(vector_count_c);

    let elem_count_d = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_d = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_d = comptime!(elem_count_d / vector_size_d);

    // Load A
    #[unroll]
    for i in 0..vector_count_a {
        let mut reg = Vector::empty();
        #[unroll]
        for k in 0..vector_size_a {
            let n_elem = i * vector_size_a + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
            let value = a[(row * size_k as u32 + col) as usize];
            reg.insert(k, value);
        }
        registers_a[i] = reg;
    }

    // Load B
    #[unroll]
    for i in 0..vector_count_b {
        let mut reg = Vector::empty();
        #[unroll]
        for k in 0..vector_size_b {
            let n_elem = i * vector_size_b + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
            let value = b[(row * size_n as u32 + col) as usize];
            reg.insert(k, value);
        }
        registers_b[i] = reg;
    }

    // Load C
    #[unroll]
    for i in 0..vector_count_c {
        let mut reg = Vector::empty();
        #[unroll]
        for k in 0..vector_size_c {
            let n_elem = i * vector_size_c + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
            let value = c[(row * size_n as u32 + col) as usize];
            reg.insert(k, value);
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D
    #[unroll]
    for i in 0..vector_count_d {
        let reg = registers_d[i];
        #[unroll]
        for k in 0..vector_size_d {
            let n_elem = i * vector_size_d + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
            out[(row * size_n as u32 + col) as usize] = reg.extract(k);
        }
    }
}

pub fn test_cmma_manual<
    R: Runtime,
    A: CubeElement + Scalar + NumCast,
    B: CubeElement + Scalar + NumCast,
    CD: CubeElement + Numeric,
>(
    client: Client,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
) {
    if !client.features().matmul.mma.contains(&MmaConfig {
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
        .flat_map(|i| (0..k).map(move |j| A::from(i * 2 + j).unwrap()))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index
    let rhs: Vec<B> = (0..k)
        .flat_map(|i| (0..n).map(move |j| B::from(i * 3 + j).unwrap()))
        .collect();
    let acc = vec![CD::from_int(0); m * n];

    let lhs = client.create_from_slice(A::as_bytes(&lhs));
    let rhs = client.create_from_slice(B::as_bytes(&rhs));
    let out = client.create_from_slice(CD::as_bytes(&acc));

    unsafe {
        kernel_manual::launch::<A, B, CD>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts(lhs, [k, 1].into(), [m, k].into()),
            TensorArg::from_raw_parts(rhs, [n, 1].into(), [k, n].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            m,
            n,
            k,
        )
    };

    let actual = client.read_one_unchecked(out);
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
pub fn kernel_manual_ldmatrix<AB: Numeric, CD: Numeric, N: Size>(
    a: &Tensor<Vector<AB, N>>,
    b: &Tensor<Vector<AB, N>>,
    c: &Tensor<CD>,
    out: &mut Tensor<CD>,
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
) {
    let bar = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let def = cmma::MmaDefinition::<AB, AB, CD>::new(size_m, size_n, size_k);
    let lane_id = UNIT_POS_PLANE as usize;

    let elem_size = AB::size();
    let width = comptime![16 / elem_size];

    let mut stage_a = Shared::new_aligned_slice(size_m * size_k, 16usize);
    let mut stage_b = Shared::new_aligned_slice(size_k * size_n, 16usize);
    bar.memcpy_async_cooperative(a.as_slice(), stage_a.as_mut_slice());
    bar.memcpy_async_cooperative(b.as_slice(), stage_b.as_mut_slice());
    bar.arrive_and_wait();

    let row = lane_id % 16;

    let col_a = (lane_id / 16) * width;
    let start_a = row * size_k + col_a;
    let slice_a = &stage_a[start_a..start_a + width];
    let vector_count_a = def.vectors_per_lane(MatrixIdent::A);

    let size!(NA) = def.vector_size(MatrixIdent::A);
    let registers_a = def.load_matrix::<_, NA>(slice_a, MatrixIdent::A, vector_count_a, false);

    // B frags are only 2 registers, so top 16 threads do nothing
    let col_b = 0;
    let start_b = row * size_n + col_b;
    let slice_b = &stage_b[start_b..start_b + width];
    let vector_count_b = def.vectors_per_lane(MatrixIdent::B);

    let size!(NB) = def.vector_size(MatrixIdent::B);
    let registers_b = def.load_matrix::<_, NB>(slice_b, MatrixIdent::B, vector_count_b, true);

    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let size!(NC) = vector_size_c;
    let vector_count_c = def.vectors_per_lane(MatrixIdent::Accumulator);
    let mut registers_c = Array::<Vector<CD, NC>>::new(vector_count_c);

    let vector_size_d = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_d = def.vectors_per_lane(MatrixIdent::Accumulator);

    // Load C
    #[unroll]
    for i in 0..vector_count_c {
        let mut reg = Vector::empty();
        #[unroll]
        for k in 0..vector_size_c {
            let n_elem = i * vector_size_c + k;
            let (row, col) =
                def.position_of_nth(lane_id as u32, n_elem as u32, MatrixIdent::Accumulator);
            let value = c[row as usize * size_n + col as usize];
            reg.insert(k, value);
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D
    #[unroll]
    for i in 0..vector_count_d {
        let reg = registers_d[i];
        #[unroll]
        for k in 0..vector_size_d {
            let n_elem = i * vector_size_d + k;
            let (row, col) =
                def.position_of_nth(lane_id as u32, n_elem as u32, MatrixIdent::Accumulator);
            out[row as usize * size_n + col as usize] = reg.extract(k);
        }
    }
}

pub fn test_cmma_manual_ldmatrix<
    R: Runtime,
    AB: CubeElement + Numeric,
    CD: CubeElement + Numeric,
>(
    client: Client,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
) {
    if !client.features().matmul.mma.contains(&MmaConfig {
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

    // The kernel stages both operands through shared memory with a barrier, so a runtime with
    // the matrix instructions but not the barrier cannot run it either.
    if !client.properties().supports_type(OpaqueType::Barrier) {
        println!("Skipping test: the runtime has no barrier to stage the operands through");
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
        kernel_manual_ldmatrix::launch::<AB, CD>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            1,
            TensorArg::from_raw_parts(lhs, [k, 1].into(), [m, k].into()),
            TensorArg::from_raw_parts(rhs, [n, 1].into(), [k, n].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            m,
            n,
            k,
        )
    };

    let actual = client.read_one_unchecked(out);
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
pub fn kernel_scaled<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size, NB: Size, NC: Size>(
    a: &Tensor<Vector<A, NA>>,
    b: &Tensor<Vector<B, NB>>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
    #[comptime] scales_factor: usize,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();

    let def =
        cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(size_m, size_n, size_k, scales_factor);
    let lane_id = UNIT_POS_PLANE;

    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_count_a = def.vectors_per_lane(MatrixIdent::A);
    let mut registers_a = Array::<Vector<A, NA>>::new(vector_count_a);

    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_count_b = def.vectors_per_lane(MatrixIdent::B);
    let mut registers_b = Array::<Vector<B, NB>>::new(vector_count_b);

    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_c = def.vectors_per_lane(MatrixIdent::Accumulator);
    let mut registers_c = Array::<Vector<CD, NC>>::new(vector_count_c);

    let vector_size_d = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_d = def.vectors_per_lane(MatrixIdent::Accumulator);

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();

    let mut scales_register_a = Vector::<S, NS>::empty();
    let mut scales_register_b = Vector::<S, NS>::empty();

    // Load A
    #[unroll]
    for i in 0..vector_count_a {
        let n_elem = i * vector_size_a * a_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        let idx = row as usize * size_k + col as usize;
        let idx = idx / (a.vector_size() * a_pack);

        registers_a[i] = a[idx];
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a.insert(i, scales_a[scales_idx_a as usize * scales_factor + i]);
    }

    // Load B
    #[unroll]
    for i in 0..vector_count_b {
        let n_elem = i * vector_size_b * b_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
        let idx = col as usize * size_k + row as usize;
        let idx = idx / (b.vector_size() * b_pack);

        registers_b[i] = b[idx];
    }

    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b.insert(i, scales_b[scales_idx_b as usize * scales_factor + i]);
    }

    // Load C
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = row as usize * size_n + col as usize;
        let value = c[idx / c.vector_size()];
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
    for i in 0..vector_count_d {
        let n_elem = i * vector_size_d;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = (row as usize * size_n + col as usize) / out.vector_size();
        out[idx] = registers_d[i];
    }
}

pub fn test_cmma_scaled<
    R: Runtime,
    A: CubeElement + Scalar + NumCast,
    B: CubeElement + Scalar + NumCast,
>(
    client: Client,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type S = ue8m0;

    let a_elem = A::cube_type();
    let b_elem = B::cube_type();
    let a_vector_size = 32 / a_elem.size_bits();
    let b_vector_size = 32 / b_elem.size_bits();

    if !client
        .features()
        .matmul
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
        .flat_map(|i| (0..k).map(move |j| A::from(i * 2 + j).unwrap()))
        .collect();
    let lhs_scales: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| S::from_bits((i * 2 + j + 120) as u8)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs: Vec<B> = (0..n)
        .flat_map(|j| (0..k).map(move |i| B::from(i * 3 + j).unwrap()))
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
        kernel_scaled::launch::<A, B, f32, S>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            a_vector_size,
            b_vector_size,
            2,
            TensorArg::from_raw_parts(lhs, [k, 1].into(), [m, k].into()),
            TensorArg::from_raw_parts(rhs, [k, 1].into(), [n, k].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(
                lhs_scales,
                [scales_factor, 1].into(),
                [m, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(
                rhs_scales,
                [scales_factor, 1].into(),
                [n, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            m,
            n,
            k,
            scales_factor,
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

    assert_equals_approx::<f32>(&client, out, &expected, 0.03);
}

pub fn test_cmma_scaled_fp4<R: Runtime, S: CubeElement + Scalar + NumCast>(
    client: Client,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type AB = e2m1x2;

    let ab_elem = AB::cube_type();
    let ab_vector_size = 32 / ab_elem.size_bits();

    if !client
        .features()
        .matmul
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
    // Powers of two from 1/8 to 8, exact in both `ue8m0` and `ue4m3`.
    let scale = |index: usize| S::from(2f32.powi((index % 7) as i32 - 3)).unwrap();
    let lhs_scales_data: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| scale(i * 2 + j)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs_data: Vec<f32> = (0..n)
        .flat_map(|j| (0..k).map(move |i| e2m1::from_bits(((i + j) % 15) as u8 + 1).to_f32()))
        .collect();
    let rhs = e2m1x2::from_f32_slice(&rhs_data);
    let rhs_scales_data: Vec<S> = (0..n)
        .flat_map(|j| (0..scales_factor).map(move |i| scale(i * 3 + j)))
        .collect();

    let out = vec![0.0; m * n];

    let lhs = client.create_from_slice(AB::as_bytes(&lhs));
    let lhs_scales = client.create_from_slice(S::as_bytes(&lhs_scales_data));
    let rhs = client.create_from_slice(AB::as_bytes(&rhs));
    let rhs_scales = client.create_from_slice(S::as_bytes(&rhs_scales_data));
    let out = client.create_from_slice(f32::as_bytes(&out));

    unsafe {
        kernel_scaled::launch::<AB, AB, f32, S>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ab_vector_size,
            ab_vector_size,
            2,
            TensorArg::from_raw_parts(lhs, [k / 2, 1].into(), [m, k / 2].into()),
            TensorArg::from_raw_parts(rhs, [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(
                lhs_scales,
                [scales_factor, 1].into(),
                [m, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(
                rhs_scales,
                [scales_factor, 1].into(),
                [n, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m, n].into()),
            m,
            n,
            k,
            scales_factor,
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
                let lhs_scale = lhs_scales_data[i * scales_factor + l_scales]
                    .to_f32()
                    .unwrap();
                let rhs_val = rhs_data[j * k + l];
                let rhs_scale = rhs_scales_data[j * scales_factor + l_scales]
                    .to_f32()
                    .unwrap();
                sum += lhs_val * lhs_scale * rhs_val * rhs_scale;
            }
            expected.push(sum);
        }
    }

    assert_equals_approx::<f32>(&client, out, &expected, 0.03);
}

#[cube(launch)]
pub fn kernel_ldmatrix_form_registers(
    input: &[u32],
    output: &mut [u32],
    #[comptime] form: cmma::LdMatrixForm,
    #[comptime] num_matrices: usize,
    #[comptime] transpose: bool,
    #[comptime] rows: usize,
) {
    let words = comptime![rows * 4];
    let mut stage = Shared::<[u32]>::new_aligned_slice(words, 16usize);
    let lane = UNIT_POS_PLANE as usize;
    let mut i = lane;
    while i < words {
        stage[i] = input[i];
        i += PLANE_DIM as usize;
    }
    sync_cube();

    let start = (lane % rows) * 4;
    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 8usize, 16usize);
    let size!(NW) = 1usize;
    let registers = def.load_matrix_with_form::<u32, NW>(
        &stage[start..start + 4],
        form,
        num_matrices,
        transpose,
    );
    let count = comptime![num_matrices * form.registers_per_matrix()];
    #[unroll]
    for r in 0..count {
        output[lane * count + r] = registers[r].extract(0usize);
    }
}

#[cube(launch)]
pub fn kernel_stmatrix_m16n8_registers(output: &mut [u32], #[comptime] num_matrices: usize) {
    let rows = comptime![num_matrices * 8];
    let words = comptime![rows * 4];
    let mut stage = Shared::<[u32]>::new_aligned_slice(words, 16usize);
    let lane = UNIT_POS_PLANE as usize;

    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 8usize, 16usize);
    let size!(NW) = 1usize;
    let mut registers = Array::<Vector<u32, NW>>::new(num_matrices);
    #[unroll]
    for r in 0..num_matrices {
        let mut word = 0u32;
        #[unroll]
        for k in 0..4u32 {
            word |= stmatrix_byte(r as u32, lane as u32, k) << (k * 8);
        }
        let mut register = Vector::empty();
        register.insert(0usize, word);
        registers[r] = register;
    }
    let start = (lane % rows) * 4;
    def.store_matrix_with_form::<u32, NW>(
        &mut stage[start..start + 4],
        &registers,
        cmma::StMatrixForm::M16N8B8,
        num_matrices,
        true,
    );
    sync_cube();

    let mut i = lane;
    while i < words {
        output[i] = stage[i];
        i += PLANE_DIM as usize;
    }
}

#[cube]
fn stmatrix_byte(matrix: u32, lane: u32, byte: u32) -> u32 {
    (matrix * 67 + lane * 4 + byte) % 256
}

/// How the elements of a row sit in shared memory for an `ldmatrix` form.
#[derive(Clone, Copy)]
enum LdMatrixRowElements {
    Bytes,
    Packed { bits: usize },
}

impl LdMatrixRowElements {
    fn of(form: cmma::LdMatrixForm) -> Self {
        match form {
            cmma::LdMatrixForm::M8N8B16 | cmma::LdMatrixForm::M16N16B8 => Self::Bytes,
            cmma::LdMatrixForm::M8N16B4x16P64 | cmma::LdMatrixForm::M16N16B4x16P64 => {
                Self::Packed { bits: 4 }
            }
            cmma::LdMatrixForm::M8N16B6x16P32 | cmma::LdMatrixForm::M16N16B6x16P32 => {
                Self::Packed { bits: 6 }
            }
        }
    }

    /// Element `element` of memory row `row`, distinct across rows sixteen apart.
    fn value(self, row: usize, element: usize) -> u8 {
        match self {
            Self::Bytes => ((row * 16 + element + (row / 16) * 7) % 256) as u8,
            Self::Packed { bits: 4 } => ((row * 5 + element + (row / 16) * 3) % 16) as u8,
            Self::Packed { .. } => ((row * 7 + element * 3) % 64) as u8,
        }
    }

    /// Sixteen elements laid out as a 16-byte row, padding bytes set to a pattern no element has.
    fn row_bytes(self, row: usize) -> [u8; 16] {
        let values: Vec<u8> = (0..16).map(|element| self.value(row, element)).collect();
        match self {
            Self::Bytes => values.try_into().unwrap(),
            Self::Packed { bits } => pack_sub_byte_row(&values, bits),
        }
    }
}

fn pack_sub_byte_row(values: &[u8], bits: usize) -> [u8; 16] {
    let mut row = [0xA5u8; 16];
    row.iter_mut()
        .take(16 * bits / 8)
        .for_each(|byte| *byte = 0);
    for (element, &value) in values.iter().enumerate() {
        for bit in 0..bits {
            if (value >> bit) & 1 == 1 {
                let position = element * bits + bit;
                row[position / 8] |= 1 << (position % 8);
            }
        }
    }
    row
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn check_ldmatrix_form(
    client: &Client,
    cube_dimensions: CubeDim,
    form: cmma::LdMatrixForm,
    num_matrices: usize,
    transpose: bool,
) {
    if !client.features().matmul.ldmatrix_forms.contains(&form) {
        println!("Skipping ldmatrix {form:?}: not advertised");
        return;
    }
    let elements = LdMatrixRowElements::of(form);
    let registers_per_matrix = form.registers_per_matrix();
    let rows_per_matrix = 8 * registers_per_matrix;
    let rows = rows_per_matrix * num_matrices;
    let memory: Vec<u8> = (0..rows).flat_map(|row| elements.row_bytes(row)).collect();
    let words = words_of(&memory);

    let count = num_matrices * registers_per_matrix;
    let input = client.create_from_slice(u32::as_bytes(&words));
    let output = client.empty(32 * count * core::mem::size_of::<u32>());
    unsafe {
        kernel_ldmatrix_form_registers::launch(
            client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            BufferArg::from_raw_parts(input, words.len()),
            BufferArg::from_raw_parts(output.clone(), 32 * count),
            form,
            num_matrices,
            transpose,
            rows,
        )
    };
    let actual = client.read_one_unchecked(output);
    let actual = u32::from_bytes(&actual);

    // Register `half` of matrix `matrix` holds logical rows `8 * half..8 * half + 8`; lane `t`
    // holds columns `(t % 4) * 4..+4` of row `t / 4` of them, one per byte. Without
    // transposition element (row, col) is element `col` of memory row `row`, with it element
    // `row` of memory row `col`. Sub-byte values sit in the low bits of their byte.
    for lane in 0..32 {
        for register in 0..count {
            let matrix = register / registers_per_matrix;
            let half = register % registers_per_matrix;
            let bytes = actual[lane * count + register].to_le_bytes();
            for (byte, &got) in bytes.iter().enumerate() {
                let row = lane / 4 + 8 * half;
                let col = (lane % 4) * 4 + byte;
                let (memory_row, element) = if transpose { (col, row) } else { (row, col) };
                let expected = elements.value(matrix * rows_per_matrix + memory_row, element);
                assert_eq!(
                    got, expected,
                    "ldmatrix {form:?} x{num_matrices}: lane {lane}, register {register}, byte {byte}"
                );
            }
        }
    }
}

pub fn test_ldmatrix_forms_place_each_byte_where_ptx_says<R: Runtime>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    for num_matrices in [1, 2, 4] {
        check_ldmatrix_form(
            &client,
            cube_dimensions,
            cmma::LdMatrixForm::M8N16B4x16P64,
            num_matrices,
            false,
        );
        check_ldmatrix_form(
            &client,
            cube_dimensions,
            cmma::LdMatrixForm::M8N16B6x16P32,
            num_matrices,
            false,
        );
    }
    for num_matrices in [1, 2] {
        for form in [
            cmma::LdMatrixForm::M16N16B8,
            cmma::LdMatrixForm::M16N16B4x16P64,
            cmma::LdMatrixForm::M16N16B6x16P32,
        ] {
            check_ldmatrix_form(&client, cube_dimensions, form, num_matrices, true);
        }
    }
}

pub fn test_stmatrix_m16n8_places_each_byte_where_ptx_says<R: Runtime>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    if !client
        .features()
        .matmul
        .stmatrix_forms
        .contains(&cmma::StMatrixForm::M16N8B8)
    {
        println!("Skipping stmatrix M16N8B8: not advertised");
        return;
    }
    for num_matrices in [1, 2, 4] {
        let words = num_matrices * 8 * 4;
        let output = client.empty(words * core::mem::size_of::<u32>());
        unsafe {
            kernel_stmatrix_m16n8_registers::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                cube_dimensions,
                BufferArg::from_raw_parts(output.clone(), words),
                num_matrices,
            )
        };
        let memory = client.read_one_unchecked(output);

        // Byte `e` of lane `t` is logical (t / 4 + 8 * (e / 2), 2 * (t % 4) + e % 2) of a 16x8
        // matrix, stored transposed: byte `row` of memory row `col`.
        for matrix in 0..num_matrices {
            for lane in 0..32 {
                for byte in 0..4 {
                    let row = lane / 4 + 8 * (byte / 2);
                    let col = 2 * (lane % 4) + byte % 2;
                    let expected = ((matrix * 67 + lane * 4 + byte) % 256) as u8;
                    assert_eq!(
                        memory[(matrix * 8 + col) * 16 + row],
                        expected,
                        "stmatrix x{num_matrices}: matrix {matrix}, lane {lane}, byte {byte}"
                    );
                }
            }
        }
    }
}

#[cube(launch)]
pub fn kernel_scale_selectors_e4m3<NA: Size, NB: Size, NC: Size>(
    a: &[Vector<e4m3, NA>],
    b: &[Vector<e4m3, NB>],
    scales_a: &[ue8m0],
    scales_b: &[ue8m0],
    out: &mut [Vector<f32, NC>],
    #[comptime] scale_a_selector: cmma::BlockScaleSelector,
    #[comptime] scale_b_selector: cmma::BlockScaleSelector,
) {
    let size_k = 32usize;
    let size_n = 8usize;
    let def = cmma::MmaDefinition::<e4m3, e4m3, f32>::new_scaled::<ue8m0>(
        16usize, 8usize, 32usize, 1usize,
    );
    let lane_id = UNIT_POS_PLANE;

    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_count_a = def.vectors_per_lane(MatrixIdent::A);
    let mut registers_a = Array::<Vector<e4m3, NA>>::new(vector_count_a);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_count_b = def.vectors_per_lane(MatrixIdent::B);
    let mut registers_b = Array::<Vector<e4m3, NB>>::new(vector_count_b);
    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_c = def.vectors_per_lane(MatrixIdent::Accumulator);
    let mut registers_c = Array::<Vector<f32, NC>>::new(vector_count_c);

    #[unroll]
    for i in 0..vector_count_a {
        let (row, col) = def.position_of_nth(lane_id, (i * vector_size_a) as u32, MatrixIdent::A);
        registers_a[i] = a[(row as usize * size_k + col as usize) / a.vector_size()];
    }
    #[unroll]
    for i in 0..vector_count_b {
        let (row, col) = def.position_of_nth(lane_id, (i * vector_size_b) as u32, MatrixIdent::B);
        registers_b[i] = b[(col as usize * size_k + row as usize) / b.vector_size()];
    }
    #[unroll]
    for i in 0..vector_count_c {
        registers_c[i] = Vector::new(0.0);
    }

    let size!(NS) = def.scales_vector_size();
    let mut scale_register_a = Vector::<ue8m0, NS>::empty();
    let mut scale_register_b = Vector::<ue8m0, NS>::empty();
    #[unroll]
    for byte in 0..4usize {
        scale_register_a.insert(byte, scales_a[lane_id as usize * 4 + byte]);
        scale_register_b.insert(byte, scales_b[lane_id as usize * 4 + byte]);
    }

    let registers_d = def.execute_scaled_with_selectors(
        &registers_a,
        &registers_b,
        &registers_c,
        scale_register_a,
        scale_register_b,
        scale_a_selector,
        scale_b_selector,
    );

    #[unroll]
    for i in 0..vector_count_c {
        let (row, col) = def.position_of_nth(
            lane_id,
            (i * vector_size_c) as u32,
            MatrixIdent::Accumulator,
        );
        out[(row as usize * size_n + col as usize) / out.vector_size()] = registers_d[i];
    }
}

/// The exponent in byte `byte` of lane `lane`'s scale register, distinct for every byte of every
/// unit within a quad, so a selector that reads the wrong one changes the result.
fn scale_exponent(lane: usize, byte: usize, operand: usize) -> u8 {
    (118 + (lane % 4) * 4 + byte + (lane / 4 + operand) % 3) as u8
}

pub fn test_scale_selectors_read_the_byte_and_lanes_they_name<R: Runtime>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    let config = ScaledMmaConfig {
        a_type: e4m3::cube_type(),
        b_type: e4m3::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: ue8m0::cube_type(),
        m: 16,
        n: 8,
        k: 32,
        scales_factor: 1,
    };
    if !client.features().matmul.scaled_mma.contains(&config) {
        println!("Skipping scale selectors: {config:?} not advertised");
        return;
    }
    let selector = |byte_id, thread_id| cmma::BlockScaleSelector { byte_id, thread_id };
    for (scale_a_selector, scale_b_selector) in [
        (selector(0, 0), selector(0, 0)),
        (selector(3, 1), selector(2, 3)),
        (selector(1, 0), selector(3, 2)),
        (selector(2, 1), selector(1, 1)),
    ] {
        let (m, n, k) = (16usize, 8usize, 32usize);
        let a_values: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 9) as f32) - 4.0).collect();
        let b_values: Vec<f32> = (0..n * k).map(|i| ((i * 5 % 7) as f32) - 3.0).collect();
        let a: Vec<e4m3> = a_values
            .iter()
            .map(|value| e4m3::from_f32(*value))
            .collect();
        let b: Vec<e4m3> = b_values
            .iter()
            .map(|value| e4m3::from_f32(*value))
            .collect();
        let scales = |operand| -> Vec<ue8m0> {
            (0..128)
                .map(|i| ue8m0::from_bits(scale_exponent(i / 4, i % 4, operand)))
                .collect()
        };

        let a_handle = client.create_from_slice(e4m3::as_bytes(&a));
        let b_handle = client.create_from_slice(e4m3::as_bytes(&b));
        let scales_a = client.create_from_slice(ue8m0::as_bytes(&scales(0)));
        let scales_b = client.create_from_slice(ue8m0::as_bytes(&scales(1)));
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
        unsafe {
            kernel_scale_selectors_e4m3::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                cube_dimensions,
                4,
                4,
                2,
                BufferArg::from_raw_parts(a_handle, m * k / 4),
                BufferArg::from_raw_parts(b_handle, n * k / 4),
                BufferArg::from_raw_parts(scales_a, 128),
                BufferArg::from_raw_parts(scales_b, 128),
                BufferArg::from_raw_parts(out.clone(), m * n / 2),
                scale_a_selector,
                scale_b_selector,
            )
        };

        // Row `i` takes its scale from lane `4 * (i % 8) + 2 * thread_id + i / 8`, column `j`
        // from lane `4 * j + thread_id`, both at byte `byte_id`.
        let mut expected = Vec::with_capacity(m * n);
        for i in 0..m {
            let lane_a = 4 * (i % 8) + 2 * scale_a_selector.thread_id + i / 8;
            let scale_a =
                ue8m0::from_bits(scale_exponent(lane_a, scale_a_selector.byte_id, 0)).to_f32();
            for j in 0..n {
                let lane_b = 4 * j + scale_b_selector.thread_id;
                let scale_b =
                    ue8m0::from_bits(scale_exponent(lane_b, scale_b_selector.byte_id, 1)).to_f32();
                let dot: f32 = (0..k)
                    .map(|l| a_values[i * k + l] * b_values[j * k + l])
                    .sum();
                expected.push(dot * scale_a * scale_b);
            }
        }
        assert_equals_approx::<f32>(&client, out, &expected, 0.001);
    }
}

#[cube(launch)]
pub fn kernel_fp4_tile_through_ldmatrix<S: Scalar, NC: Size>(
    a: &[u32],
    b: &[u32],
    scales_a: &[S],
    scales_b: &[S],
    out: &mut [Vector<f32, NC>],
    #[comptime] scales_factor: usize,
    #[comptime] scale_a_selector: cmma::BlockScaleSelector,
    #[comptime] scale_b_selector: cmma::BlockScaleSelector,
) {
    // A is 16 rows of 32 bytes and B 8 columns of 32 bytes, both contiguous along `k`.
    let mut stage_a = Shared::<[u32]>::new_aligned_slice(128usize, 16usize);
    let mut stage_b = Shared::<[u32]>::new_aligned_slice(64usize, 16usize);
    let lane = UNIT_POS_PLANE as usize;
    #[unroll]
    for i in 0..4usize {
        stage_a[lane + i * 32] = a[lane + i * 32];
    }
    #[unroll]
    for i in 0..2usize {
        stage_b[lane + i * 32] = b[lane + i * 32];
    }
    sync_cube();

    let def = cmma::MmaDefinition::<e2m1x2, e2m1x2, f32>::new_scaled::<S>(
        16usize,
        8usize,
        64usize,
        scales_factor,
    );
    let size!(NW) = 1usize;
    // The four 8x16-byte matrices of A are rows 0-7 and 8-15 of bytes 0-15, then of bytes 16-31.
    let row_a = lane % 8 + 8 * ((lane / 8) % 2);
    let start_a = row_a * 8 + (lane / 16) * 4;
    let words_a = def.load_matrix::<u32, NW>(
        &stage_a[start_a..start_a + 4],
        MatrixIdent::A,
        4usize,
        false,
    );
    let start_b = (lane % 8) * 8 + ((lane / 8) % 2) * 4;
    let words_b = def.load_matrix::<u32, NW>(
        &stage_b[start_b..start_b + 4],
        MatrixIdent::B,
        2usize,
        false,
    );

    let size!(N4) = 4usize;
    let mut registers_a = Array::<Vector<e2m1x2, N4>>::new(4usize);
    #[unroll]
    for i in 0..4usize {
        registers_a[i] = Vector::<e2m1x2, N4>::reinterpret(words_a[i].extract(0usize));
    }
    let mut registers_b = Array::<Vector<e2m1x2, N4>>::new(2usize);
    #[unroll]
    for i in 0..2usize {
        registers_b[i] = Vector::<e2m1x2, N4>::reinterpret(words_b[i].extract(0usize));
    }
    let mut registers_c = Array::<Vector<f32, NC>>::new(2usize);
    #[unroll]
    for i in 0..2usize {
        registers_c[i] = Vector::new(0.0);
    }

    let size!(NS) = def.scales_vector_size();
    let mut scale_register_a = Vector::<S, NS>::empty();
    let mut scale_register_b = Vector::<S, NS>::empty();
    #[unroll]
    for byte in 0..4usize {
        scale_register_a.insert(byte, scales_a[lane * 4 + byte]);
        scale_register_b.insert(byte, scales_b[lane * 4 + byte]);
    }

    let registers_d = def.execute_scaled_with_selectors(
        &registers_a,
        &registers_b,
        &registers_c,
        scale_register_a,
        scale_register_b,
        scale_a_selector,
        scale_b_selector,
    );

    #[unroll]
    for i in 0..2usize {
        let (row, col) = def.position_of_nth(lane as u32, (i * 2) as u32, MatrixIdent::Accumulator);
        out[(row as usize * 8 + col as usize) / 2] = registers_d[i];
    }
}

/// Runs one 16x8x64 FP4 tile, operands loaded through `ldmatrix`, with each lane's scale
/// register given by `scale_of(lane, byte, operand)`, against the CPU reference.
fn check_fp4_tile_through_ldmatrix<S: CubeElement + Scalar + NumCast>(
    client: &Client,
    cube_dimensions: CubeDim,
    scales_factor: usize,
    scale_of: impl Fn(usize, usize, usize) -> S,
    scale_a_selector: cmma::BlockScaleSelector,
    scale_b_selector: cmma::BlockScaleSelector,
) {
    let config = ScaledMmaConfig {
        a_type: e2m1x2::cube_type(),
        b_type: e2m1x2::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: S::cube_type(),
        m: 16,
        n: 8,
        k: 64,
        scales_factor: scales_factor as u32,
    };
    if !client.features().matmul.scaled_mma.contains(&config) {
        println!("Skipping FP4 tile: {config:?} not advertised");
        return;
    }
    let (m, n, k) = (16usize, 8usize, 64usize);
    let a_values: Vec<f32> = (0..m * k)
        .map(|i| e2m1::from_bits(((i * 7 + i / 64) % 16) as u8).to_f32())
        .collect();
    let b_values: Vec<f32> = (0..n * k)
        .map(|i| e2m1::from_bits(((i * 5 + 3 * (i / 64)) % 16) as u8).to_f32())
        .collect();
    let scales_a: Vec<S> = (0..128).map(|i| scale_of(i / 4, i % 4, 0)).collect();
    let scales_b: Vec<S> = (0..128).map(|i| scale_of(i / 4, i % 4, 1)).collect();

    let a = client.create_from_slice(e2m1x2::as_bytes(&e2m1x2::from_f32_slice(&a_values)));
    let b = client.create_from_slice(e2m1x2::as_bytes(&e2m1x2::from_f32_slice(&b_values)));
    let scales_a_handle = client.create_from_slice(S::as_bytes(&scales_a));
    let scales_b_handle = client.create_from_slice(S::as_bytes(&scales_b));
    let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    unsafe {
        kernel_fp4_tile_through_ldmatrix::launch::<S>(
            client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            2,
            BufferArg::from_raw_parts(a, 128),
            BufferArg::from_raw_parts(b, 64),
            BufferArg::from_raw_parts(scales_a_handle, 128),
            BufferArg::from_raw_parts(scales_b_handle, 128),
            BufferArg::from_raw_parts(out.clone(), m * n / 2),
            scales_factor,
            scale_a_selector,
            scale_b_selector,
        )
    };

    // Row `i` takes its scales from lane `4 * (i % 8) + 2 * thread_id + i / 8`, column `j` from
    // lane `4 * j + thread_id`, block `l / (k / scales_factor)` at byte `byte_id` plus the block.
    let block = k / scales_factor;
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        let lane_a = 4 * (i % 8) + 2 * scale_a_selector.thread_id + i / 8;
        for j in 0..n {
            let lane_b = 4 * j + scale_b_selector.thread_id;
            let sum: f32 = (0..k)
                .map(|l| {
                    let scale_a = scales_a[lane_a * 4 + scale_a_selector.byte_id + l / block];
                    let scale_b = scales_b[lane_b * 4 + scale_b_selector.byte_id + l / block];
                    a_values[i * k + l]
                        * scale_a.to_f32().unwrap()
                        * b_values[j * k + l]
                        * scale_b.to_f32().unwrap()
                })
                .sum();
            expected.push(sum);
        }
    }
    assert_equals_approx::<f32>(client, out, &expected, 0.001);
}

pub fn test_fp4_scale_selectors_read_the_byte_and_lanes_they_name<R: Runtime>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    let selector = |byte_id, thread_id| cmma::BlockScaleSelector { byte_id, thread_id };
    let ue8m0_scale = |lane, byte, operand| ue8m0::from_bits(scale_exponent(lane, byte, operand));
    let ue4m3_values = [
        0.5f32, 1.0, 1.5, 2.0, 0.75, 3.0, 0.25, 1.25, 1.75, 2.5, 0.375, 0.625,
    ];
    let ue4m3_scale = |lane: usize, byte: usize, operand: usize| {
        e4m3::from_f32(ue4m3_values[((lane % 4) * 4 + byte + (lane / 4) * 5 + operand * 7) % 12])
    };
    for (scale_a_selector, scale_b_selector) in [
        (selector(2, 1), selector(2, 3)),
        (selector(2, 0), selector(0, 1)),
    ] {
        check_fp4_tile_through_ldmatrix(
            &client,
            cube_dimensions,
            2,
            ue8m0_scale,
            scale_a_selector,
            scale_b_selector,
        );
    }
    check_fp4_tile_through_ldmatrix(
        &client,
        cube_dimensions,
        4,
        ue4m3_scale,
        selector(0, 1),
        selector(0, 2),
    );
    check_fp4_tile_through_ldmatrix(
        &client,
        cube_dimensions,
        4,
        ue8m0_scale,
        selector(0, 1),
        selector(0, 3),
    );
}

pub fn test_nvfp4_tile_loaded_through_ldmatrix_matches_reference<R: Runtime>(
    client: Client,
    cube_dimensions: CubeDim,
) {
    // The units the default selectors read: lane `t` holds the scales of row `t / 4 + 8 * (t % 2)`
    // of A and of column `t / 4` of B, one per block of 16 along `k`.
    let values = [0.5f32, 1.0, 1.5, 2.0, 0.75, 3.0, 0.25, 1.25];
    let scale_of = |lane: usize, block: usize, operand: usize| {
        let owner = if operand == 0 {
            lane / 4 + 8 * (lane % 2)
        } else {
            lane / 4
        };
        e4m3::from_f32(values[(owner * 3 + block * 5 + operand) % 8])
    };
    check_fp4_tile_through_ldmatrix(
        &client,
        cube_dimensions,
        4,
        scale_of,
        cmma::BlockScaleSelector::default(),
        cmma::BlockScaleSelector::default(),
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_1() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1::<TestRuntime>(client, cube_dimensions);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_1_bf16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_typed::<TestRuntime, half::bf16, f32>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_1_bf16_accumulator() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_typed::<
                TestRuntime,
                half::bf16,
                half::bf16,
            >(client, cube_dimensions);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_1_vectorized() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_vectorized::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_1_vectorized_offset() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_vectorized_offset::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_accumulator_row_major() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_accumulator_row_major::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_unsupported_fragment() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_unsupported_fragment::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_simple_tf32() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_tf32::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cube_mma_simple() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cmma::test_simple_cube::<TestRuntime>(client.clone(), 32);
            cubecl_core::runtime_tests::cmma::test_simple_cube::<TestRuntime>(client.clone(), 64);
            cubecl_core::runtime_tests::cmma::test_simple_cube::<TestRuntime>(client.clone(), 128);
            cubecl_core::runtime_tests::cmma::test_simple_cube::<TestRuntime>(client.clone(), 256);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cube_mma_simple_tensor() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cmma::test_simple_cube_tensor::<TestRuntime>(
                client.clone(),
                32,
            );
            cubecl_core::runtime_tests::cmma::test_simple_cube_tensor::<TestRuntime>(
                client.clone(),
                64,
            );
            cubecl_core::runtime_tests::cmma::test_simple_cube_tensor::<TestRuntime>(
                client.clone(),
                128,
            );
            cubecl_core::runtime_tests::cmma::test_simple_cube_tensor::<TestRuntime>(
                client.clone(),
                256,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_cast_f16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_f16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_cast_bf16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_bf16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_strided() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_strided::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_manual() {
            use cubecl_common::*;
            use cubecl_core::num_traits::cast::NumCast;
            use half::{bf16, f16};

            fn test<
                A: CubeElement + Scalar + NumCast,
                B: CubeElement + Scalar + NumCast,
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

        #[$crate::runtime_tests::test_log::test]
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

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_scaled() {
            use cubecl_common::*;
            use cubecl_core::num_traits::cast::NumCast;

            fn test<A: CubeElement + Scalar + NumCast, B: CubeElement + Scalar + NumCast>(
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

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_scaled_fp4() {
            use cubecl_common::*;
            use cubecl_core::num_traits::cast::NumCast;

            fn test<S: CubeElement + Scalar + NumCast>(m: usize, n: usize, k: usize, factor: usize) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_scaled_fp4::<TestRuntime, S>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                    factor,
                )
            }

            // FP4 needs more design for transferring properly as packed values
            test::<ue8m0>(16, 8, 64, 2);
            test::<ue8m0>(16, 8, 64, 4);
            test::<e4m3>(16, 8, 64, 4);
        }

        /// Each sm_120 `ldmatrix` form, at each count it allows, delivers every byte to the lane
        /// and register the PTX fragment figures name.
        #[$crate::runtime_tests::test_log::test]
        fn test_ldmatrix_forms_place_each_byte_where_ptx_says() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_ldmatrix_forms_place_each_byte_where_ptx_says::<
                TestRuntime,
            >(client, cube_dimensions);
        }

        /// `stmatrix` `m16n8 .trans .b8` writes every byte of every lane where PTX puts it.
        #[$crate::runtime_tests::test_log::test]
        fn test_stmatrix_m16n8_places_each_byte_where_ptx_says() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_stmatrix_m16n8_places_each_byte_where_ptx_says::<
                TestRuntime,
            >(client, cube_dimensions);
        }

        /// Non-zero scale selectors of a `scale_vec::1X` MMA read the byte and units they name.
        #[$crate::runtime_tests::test_log::test]
        fn test_scale_selectors_read_the_byte_and_lanes_they_name() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_scale_selectors_read_the_byte_and_lanes_they_name::<
                TestRuntime,
            >(client, cube_dimensions);
        }

        /// Non-zero scale selectors of the FP4 `scale_vec::2X` and `4X` MMAs read the bytes and
        /// units they name.
        #[$crate::runtime_tests::test_log::test]
        fn test_fp4_scale_selectors_read_the_byte_and_lanes_they_name() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_fp4_scale_selectors_read_the_byte_and_lanes_they_name::<
                TestRuntime,
            >(client, cube_dimensions);
        }

        /// An NVFP4 tile whose operands are loaded through `ldmatrix` matches the CPU reference.
        #[$crate::runtime_tests::test_log::test]
        fn test_nvfp4_tile_loaded_through_ldmatrix_matches_reference() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_nvfp4_tile_loaded_through_ldmatrix_matches_reference::<
                TestRuntime,
            >(client, cube_dimensions);
        }

        fn cube_dim<R: Runtime>(client: &Client) -> CubeDim {
            let plane_dim = client.properties().hardware.plane_size_max;
            CubeDim::new_1d(plane_dim)
        }
    };
}
