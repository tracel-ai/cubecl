use cubecl_runtime::runtime::Runtime;

use crate::{self as cubecl, prelude::*};

#[cube]
fn double(x: u32) -> u32 {
    x * 2
}

#[cube]
fn multiply_add(a: i32, b: i32, c: i32) -> i32 {
    a * b + c
}

#[cube]
fn scale<F: Float>(x: F, #[comptime] factor: u32) -> F {
    x * F::cast_from(factor)
}

#[cube]
fn truncate_add<F: Float, I: Int>(x: F, y: I) -> I {
    I::cast_from(x) + y
}

pub fn test_harness_scalar<R: Runtime>() {
    assert_eq!(double::harness::<R>(1), 2);
    assert_eq!(multiply_add::harness::<R>(-3, 4, 5), -7);
}

/// A function in a block isn't reachable by path from its own module, so the
/// harness must not name it through `super`.
pub fn test_harness_block_scoped<R: Runtime>() {
    #[cube]
    fn negate(x: i32) -> i32 {
        -x
    }

    assert_eq!(negate::harness::<R>(3), -3);
}

pub fn test_harness_generic<R: Runtime>() {
    assert_eq!(scale::harness::<R, f32>(1.5, 4), 6.0);
    assert_eq!(truncate_add::harness::<R, f32, i32>(2.75, 3), 5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_harness {
    () => {
        mod harness {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn harness_scalar() {
                cubecl_core::runtime_tests::harness::test_harness_scalar::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_block_scoped() {
                cubecl_core::runtime_tests::harness::test_harness_block_scoped::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn harness_generic() {
                cubecl_core::runtime_tests::harness::test_harness_generic::<TestRuntime>();
            }
        }
    };
}
