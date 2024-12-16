#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_simple {
    () => {
        mod simple {
            $crate::testgen_simple!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            $crate::testgen_simple_matmul!();
    };
    ([$($float:ident),*]) => {
        mod simple {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_simple!($float);
                })*
            }
        }
    };
}
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_simple_matmul {
    () => {
        use super::*;

        #[test]
        pub fn test_matmul_simple_small() {
            cubecl_linalg::matmul::tests::simple::test_matmul_simple_small::<TestRuntime, FloatT>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_simple_large() {
            cubecl_linalg::matmul::tests::simple::test_matmul_simple_large::<TestRuntime, FloatT>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_simple_with_check_bounds() {
            cubecl_linalg::matmul::tests::simple::test_matmul_simple_with_check_bounds::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_simple_with_batches() {
            cubecl_linalg::matmul::tests::simple::test_matmul_simple_with_batches::<
                TestRuntime,
                FloatT,
            >(&Default::default())
        }
    };
}
