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
            use $crate::tests::tensor::identity::test_identity;
            use cubecl_core::flex32;

            pub type NumericT = $numeric;

            #[test]
            pub fn test_tiny() {
                test_identity::<TestRuntime, NumericT>(&Default::default(), 4);
            }

            #[test]
            pub fn test_small() {
                test_identity::<TestRuntime, NumericT>(&Default::default(), 16);
            }

            #[test]
            pub fn test_normal() {
                test_identity::<TestRuntime, NumericT>(&Default::default(), 256)
            }

            #[test]
            pub fn test_large() {
                test_identity::<TestRuntime, NumericT>(&Default::default(), 1024)
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
