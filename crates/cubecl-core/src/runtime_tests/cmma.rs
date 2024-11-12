use crate as cubecl;

use crate::Feature;
use cubecl::{
    ir::{Elem, FloatKind},
    prelude::*,
};
use half::f16;

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_1(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
        lhs.as_slice(),
        16,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
        rhs.as_slice(),
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

    cmma::store(out.as_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
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
        lhs.as_slice(),
        8,
    );
    let b = cmma::Matrix::<tf32>::from_slice(
        cmma::MatrixIdent::B,
        16,
        16,
        8,
        cmma::MatrixLayout::RowMajor,
        rhs.as_slice(),
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

    cmma::store(out.as_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
}

pub fn test_simple_1<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
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
            // For HIP, change dim to:
            // CubeDim::new(32, 1, 1),
            CubeDim::new(16, 16, 1),
            ArrayArg::from_raw_parts::<f16>(&lhs, 256, 1),
            ArrayArg::from_raw_parts::<f16>(&rhs, 256, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read(out.binding());
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

pub fn test_simple_tf32<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
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
            CubeDim::new(16, 16, 1),
            ArrayArg::from_raw_parts::<f32>(&lhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&rhs, 128, 1),
            ArrayArg::from_raw_parts::<f32>(&out, 256, 1),
        )
    };

    let actual = client.read(out.binding());
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        #[test]
        fn test_cmma_simple_1() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cmma::test_simple_1::<TestRuntime>(client);
        }

        #[test]
        fn test_cmma_simple_tf32() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cmma::test_simple_tf32::<TestRuntime>(client);
        }
    };
}
