#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_qr_cgr {
    () => {
        mod cgr {
            $crate::testgen_qr_cgr!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use cubecl_std::tests;
            use cubecl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_tiny() {
                cubecl_qr::tests::cgr::test_qr_cgr::<TestRuntime, FloatT>(&Default::default(), 3);
            }

            #[test]
            pub fn test_small() {
                cubecl_qr::tests::cgr::test_qr_cgr::<TestRuntime, FloatT>(&Default::default(), 47);
            }

            #[test]
            pub fn test_medium() {
                cubecl_qr::tests::cgr::test_qr_cgr::<TestRuntime, FloatT>(&Default::default(), 157);
            }

            #[test]
            pub fn test_big() {
                cubecl_qr::tests::cgr::test_qr_cgr::<TestRuntime, FloatT>(&Default::default(), 517);
            }

    };
    ([$($float:ident),*]) => {
        mod cgr {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_qr_cgr!($float);
                })*
            }
        }
    };
}
