#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tensor_into_contiguous {
    () => {
        mod into_contiguous {
            use super::*;
            use $crate::tests::tensor::into_contiguous::{
                test_into_contiguous_packed_halving, test_into_contiguous_packed_multi_vector,
                test_into_contiguous_packed_repack, test_into_contiguous_packed_vector_size_one,
            };

            #[$crate::tests::test_log::test]
            pub fn test_packed_repack() {
                test_into_contiguous_packed_repack::<TestRuntime>(&Default::default());
            }

            #[$crate::tests::test_log::test]
            pub fn test_packed_vector_size_one() {
                test_into_contiguous_packed_vector_size_one::<TestRuntime>(&Default::default());
            }

            #[$crate::tests::test_log::test]
            pub fn test_packed_multi_vector() {
                test_into_contiguous_packed_multi_vector::<TestRuntime>(&Default::default());
            }

            #[$crate::tests::test_log::test]
            pub fn test_packed_halving() {
                test_into_contiguous_packed_halving::<TestRuntime>(&Default::default());
            }
        }
    };
}
