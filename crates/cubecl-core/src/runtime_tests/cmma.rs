use crate as cubecl;

use crate::Feature;
use cubecl::{
    ir::{Elem, FloatKind},
    prelude::*,
};
use half::{bf16, f16};

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
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

pub fn test_simple_1<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_dimensions: CubeDim,
) {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    }) {
        // We can't execute the test, skip.
        return;
    }

    let lhs: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
    let rhs: Vec<f16> = (0..256).map(|i| f16::from_f32((i % 8) as f32)).collect();

    let lhs = client.create(f16::as_bytes(&lhs));
    let rhs = client.create(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * 256);

    unsafe {
        kernel_simple_1::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f16>(&lhs, 256, 1),
            ArrayArg::from_raw_parts::<f16>(&rhs, 256, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out.binding());
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

pub fn test_cmma_cast_f16<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_dimensions: CubeDim,
) {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
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

    let actual = client.read_one(out.binding());
    let actual = f16::from_bytes(&actual);
    let expected: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}

pub fn test_cmma_cast_bf16<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_dimensions: CubeDim,
) {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::BF16),
        b: Elem::Float(FloatKind::BF16),
        c: Elem::Float(FloatKind::F32),
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

    let actual = client.read_one(out.binding());
    let actual = bf16::from_bytes(&actual);
    let expected: Vec<bf16> = (0..256).map(|i| bf16::from_f32(i as f32)).collect();

    assert_eq!(actual, expected);
}

pub fn test_simple_tf32<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_dimensions: CubeDim,
) {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::TF32),
        b: Elem::Float(FloatKind::TF32),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 8,
        n: 16,
    }) {
        // We can't execute the test, skip.
        return;
    }

    let lhs: Vec<f32> = (0..128).map(|i| (i as f32)).collect();
    let rhs: Vec<f32> = (0..128).map(|i| ((i % 8) as f32)).collect();

    let lhs = client.create(f32::as_bytes(&lhs));
    let rhs = client.create(f32::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * 256);

    unsafe {
        kernel_simple_tf32::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            ArrayArg::from_raw_parts::<f32>(&lhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&rhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read_one(out.binding());
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

pub fn test_cmma_strided<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_dimensions: CubeDim,
) {
    // Lhs (row major) will have strided tiles
    let (m, n, k) = (16, 16, 32);
    let (t_m, t_n, t_k) = (16, 16, 16);
    if !client.properties().feature_enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: t_m as u8,
        k: t_k as u8,
        n: t_n as u8,
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

    let lhs = client.create(f16::as_bytes(&lhs));
    let rhs = client.create(f16::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<f32>() * m * n);

    unsafe {
        kernel_strided::launch::<R>(
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

    let actual = client.read_one(out.binding());
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_core::CubeDim;

        #[test]
        fn test_cmma_simple_1() {
            let client = TestRuntime::client(&Default::default());
            // In HIP the thread block size must be 32
            #[cfg(feature = "is_hip")]
            let cube_dimensions = CubeDim::new(32, 1, 1);
            #[cfg(not(feature = "is_hip"))]
            let cube_dimensions = CubeDim::new(16, 16, 1);
            cubecl_core::runtime_tests::cmma::test_simple_1::<TestRuntime>(client, cube_dimensions);
        }

        #[test]
        fn test_cmma_simple_tf32() {
            let client = TestRuntime::client(&Default::default());
            // In HIP the thread block size must be 32
            #[cfg(feature = "is_hip")]
            let cube_dimensions = CubeDim::new(32, 1, 1);
            #[cfg(not(feature = "is_hip"))]
            let cube_dimensions = CubeDim::new(16, 16, 1);
            cubecl_core::runtime_tests::cmma::test_simple_tf32::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_f16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = CubeDim::new(32, 1, 1);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_f16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_bf16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = CubeDim::new(32, 1, 1);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_bf16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_strided() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = CubeDim::new(16, 16, 1);
            cubecl_core::runtime_tests::cmma::test_cmma_strided::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }
    };
}
