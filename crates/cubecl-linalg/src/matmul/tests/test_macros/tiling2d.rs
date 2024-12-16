#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tiling2d_matmul {
    () => {
        mod tiling2d {
            $crate::testgen_tiling2d_matmul!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_tiling2d_matmul_one_cube() {
                cubecl_linalg::matmul::tests::tiling2d::test_tiling2d_matmul_one_cube::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_tiling2d_matmul_several_cubes() {
                cubecl_linalg::matmul::tests::tiling2d::test_tiling2d_matmul_several_cubes::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_tiling2d_matmul_with_check_bounds() {
                cubecl_linalg::matmul::tests::tiling2d::test_tiling2d_matmul_with_check_bounds::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_tiling2d_matmul_with_batches() {
                cubecl_linalg::matmul::tests::tiling2d::test_tiling2d_matmul_with_batches::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }
    };
    ([$($float:ident),*]) => {
        mod tiling2d {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_tiling2d_matmul!($float);
                })*
            }
        }
    };
}
