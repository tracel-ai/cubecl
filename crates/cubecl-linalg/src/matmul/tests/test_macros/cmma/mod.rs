#![allow(missing_docs)]

pub mod suite;

#[macro_export]
macro_rules! testgen_matmul_accelerated {
    ($eg:ty, $es:ty) => {
        type Precision = ($eg, $es);

        $crate::matmul_standard_tests!();

        #[test]
        pub fn virtual_smem_test_side_by_side_lhs() {
            $crate::matmul::components::stage::virtual_smem_test_side_by_side::<TestRuntime>(TestRuntime::client(&Default::default()), $crate::matmul::components::Ident::Lhs);
        }

        #[test]
        pub fn virtual_smem_test_interleaved_lhs() {
            $crate::matmul::components::stage::virtual_smem_test_interleaved::<TestRuntime>(TestRuntime::client(&Default::default()), $crate::matmul::components::Ident::Lhs);
        }

        #[test]
        pub fn virtual_smem_test_side_by_side_rhs() {
            $crate::matmul::components::stage::virtual_smem_test_side_by_side::<TestRuntime>(TestRuntime::client(&Default::default()), $crate::matmul::components::Ident::Rhs);
        }

        #[test]
        pub fn virtual_smem_test_interleaved_rhs() {
            $crate::matmul::components::stage::virtual_smem_test_interleaved::<TestRuntime>(TestRuntime::client(&Default::default()), $crate::matmul::components::Ident::Rhs);
        }
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated::Accelerated;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float, $float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_quantized {
    () => {
        #[allow(non_snake_case)]
        mod matmul_quantized {
            use super::*;

            type Precision = $crate::matmul::tests::SymQ8;
            type TMM = $crate::matmul::components::tile::accelerated::Accelerated;

            $crate::matmul_standard_tests!();
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_plane {
    ($float:ident) => {
        type Precision = ($eg, $es);

        $crate::matmul_standard_tests!();
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
        mod matmul_plane {
            use super::*;
            type TMM = $crate::matmul::components::tile::plane::PlaneMma;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_matmul_accelerated!($float, $float);
                })*
            }
        }
    };
}
