use approx::assert_abs_diff_eq;
use core::f32::consts::{PI, TAU};
use cubecl_runtime::runtime::Runtime;

use crate::trigonometry::*;

pub fn test_to_degrees<R: Runtime>() {
    let cases = [
        (0.0, 0.0),
        (PI / 6.0, 30.0),
        (PI / 4.0, 45.0),
        (PI / 2.0, 90.0),
        (PI, 180.0),
        (TAU, 360.0),
    ];

    for (input, expected) in cases {
        assert_abs_diff_eq!(
            to_degrees::harness::<R, f32>(input),
            expected,
            epsilon = 1e-5
        );
    }
}

pub fn test_to_radians<R: Runtime>() {
    let cases = [
        (0.0, 0.0),
        (30.0, PI / 6.0),
        (45.0, PI / 4.0),
        (90.0, PI / 2.0),
        (180.0, PI),
        (360.0, TAU),
    ];

    for (input, expected) in cases {
        assert_abs_diff_eq!(
            to_radians::harness::<R, f32>(input),
            expected,
            epsilon = 1e-5
        );
    }
}

#[macro_export]
macro_rules! testgen_trigonometry {
    () => {
        mod trigonometry {
            use super::*;
            use $crate::tests::trigonometry::*;

            #[$crate::tests::test_log::test]
            fn test_to_degrees_conversion() {
                test_to_degrees::<TestRuntime>();
            }

            #[$crate::tests::test_log::test]
            fn test_to_radians_conversion() {
                test_to_radians::<TestRuntime>();
            }
        }
    };
}
