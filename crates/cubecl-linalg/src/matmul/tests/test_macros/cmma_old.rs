#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_cmma_old {
    () => {
        mod cmma_old {
            $crate::testgen_cmma_old!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;

            pub type FloatT = $float;

            #[test]
            pub fn test_matmul_cmma_old_all() {
                cubecl_linalg::matmul::tests::cmma_old::table_test::test_cmma_all::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }
    };
    ([$($float:ident),*]) => {
        mod cmma_old {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_cmma_old!($float);
                })*
            }
        }
    };
}
