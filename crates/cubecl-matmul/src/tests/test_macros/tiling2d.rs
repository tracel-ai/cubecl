#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_matmul_tiling2d {
    () => {
        mod matmul_tiling2d {
            $crate::testgen_matmul_tiling2d!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_matmul::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_one_cube() {
                cubecl_matmul::tests::tiling2d::test_one_cube::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_several_cubes() {
                cubecl_matmul::tests::tiling2d::test_several_cubes::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_with_check_bounds() {
                cubecl_matmul::tests::tiling2d::test_with_check_bounds::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_with_batches() {
                cubecl_matmul::tests::tiling2d::test_with_batches::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }
    };
    ([$($float:ident),*]) => {
        mod matmul_tiling2d {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_matmul_tiling2d!($float);
                })*
            }
        }
    };
}
