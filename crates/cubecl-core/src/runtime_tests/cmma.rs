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
            CubeDim::new(16, 16, 1),
            ArrayArg::from_raw_parts(&lhs, 256, 1),
            ArrayArg::from_raw_parts(&rhs, 256, 1),
            ArrayArg::from_raw_parts(&out, 256, 1),
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
    };
}
