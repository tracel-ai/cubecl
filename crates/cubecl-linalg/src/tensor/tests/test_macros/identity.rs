#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tensor_identity {
    () => {
        mod identity {
            $crate::testgen_tensor_identity!(f32);
        }
    };
    ($numeric:ident) => {
            use super::*;
            use cubecl_linalg::tensor::tests;
            use cubecl_core::flex32;

            pub type NumericT = $numeric;

            #[test]
            pub fn test_tiny() {
                cubecl_linalg::tensor::tests::identity::test_identity::<TestRuntime, NumericT>(&Default::default(), 3);
            }

            #[test]
            pub fn test_small() {
                cubecl_linalg::tensor::tests::identity::test_identity::<TestRuntime, NumericT>(&Default::default(), 16);
            }

            #[test]
            pub fn test_normal() {
                cubecl_linalg::tensor::tests::identity::test_identity::<TestRuntime, NumericT>(&Default::default(), 256)
            }

            #[test]
            pub fn test_large() {
                cubecl_linalg::tensor::tests::identity::test_identity::<TestRuntime, NumericT>(&Default::default(), 4096)
            }
    };
    ([$($numeric:ident),*]) => {
        mod identity {
            use super::*;
            ::paste::paste! {
                $(mod [<$numeric _ty>] {
                    use super::*;

                    $crate::testgen_tensor_identity!($numeric);
                })*
            }
        }
    };
}
