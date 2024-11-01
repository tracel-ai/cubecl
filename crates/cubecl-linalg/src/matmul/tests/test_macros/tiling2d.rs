#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tiling2d {
    () => {
        mod tiling2d {
            $crate::testgen_tiling2d!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;

            pub type FloatT = $float;

            $crate::testgen_tiling2d_matmul!();
    };
    ([$($float:ident),*]) => {
        mod tiling2d {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_tiling2d!($float);
                })*
            }
        }
    };
}
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tiling2d_matmul {
    () => {
        use super::*;

        #[test]
        pub fn test_matmul_tiling2d_one_cube() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_one_cube::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_tiling2d_several_cubes() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_several_cubes::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_tiling2d_with_check_bounds() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_with_check_bounds::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_tiling2d_with_batches() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_with_batches::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }
    };
}
