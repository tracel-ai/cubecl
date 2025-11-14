#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tensor_permute {
    () => {
        mod permute {
            $crate::testgen_tensor_permute!(f32);
        }
    };
    ($numeric:ident) => {
            use super::*;
            use $crate::tests::tensor::permute::*;

            pub type NumericT = $numeric;

            #[test]
            pub fn test_2d_transpose_small() {
                test_permute_2d_transpose::<TestRuntime, NumericT>(&Default::default(), 8, 12);
            }

            #[test]
            pub fn test_2d_transpose_square() {
                test_permute_2d_transpose::<TestRuntime, NumericT>(&Default::default(), 64, 64);
            }

            #[test]
            pub fn test_2d_transpose_large() {
                test_permute_2d_transpose::<TestRuntime, NumericT>(&Default::default(), 128, 256);
            }

            #[test]
            pub fn test_3d_batch_transpose_small() {
                test_permute_3d_batch_transpose::<TestRuntime, NumericT>(&Default::default(), 2, 8, 8);
            }

            #[test]
            pub fn test_3d_batch_transpose_medium() {
                test_permute_3d_batch_transpose::<TestRuntime, NumericT>(&Default::default(), 4, 32, 64);
            }

            #[test]
            pub fn test_3d_complex_permutation() {
                test_permute_3d_complex::<TestRuntime, NumericT>(&Default::default(), 4, 8, 16);
            }

            #[test]
            pub fn test_empty_tensor() {
                test_permute_empty::<TestRuntime, NumericT>(&Default::default());
            }

            #[test]
            pub fn test_single_element() {
                test_permute_single_element::<TestRuntime, NumericT>(&Default::default());
            }

            #[test]
            pub fn test_4d_last_two_transpose_small() {
                test_permute_4d_last_two_transpose::<TestRuntime, NumericT>(&Default::default(), 2, 3, 16, 24);
            }

            #[test]
            pub fn test_4d_last_two_transpose_medium() {
                test_permute_4d_last_two_transpose::<TestRuntime, NumericT>(&Default::default(), 4, 8, 32, 64);
            }

            #[test]
            pub fn test_4d_complex_permutation() {
                test_permute_4d_complex::<TestRuntime, NumericT>(&Default::default(), 2, 4, 8, 16);
            }

            #[test]
            pub fn test_channel_shuffle_small() {
                test_permute_channel_shuffle::<TestRuntime, NumericT>(&Default::default(), 2, 4, 8, 8);
            }

            #[test]
            pub fn test_channel_shuffle_medium() {
                test_permute_channel_shuffle::<TestRuntime, NumericT>(&Default::default(), 4, 16, 32, 32);
            }

            #[test]
            pub fn test_attention_transpose_small() {
                test_permute_attention_transpose::<TestRuntime, NumericT>(&Default::default(), 2, 8, 16, 64);
            }

            #[test]
            pub fn test_attention_transpose_medium() {
                test_permute_attention_transpose::<TestRuntime, NumericT>(&Default::default(), 4, 12, 128, 64);
            }

            #[test]
            pub fn test_tiny_transpose_4x4() {
                // 16 elements - uses plane shuffle
                test_permute_small_transpose::<TestRuntime, NumericT>(&Default::default(), 4);
            }

            #[test]
            pub fn test_small_transpose_8x8() {
                // 64 elements - falls back to tiled transpose (too big for plane shuffle)
                test_permute_small_transpose::<TestRuntime, NumericT>(&Default::default(), 8);
            }

            #[test]
            pub fn test_small_transpose_16x16() {
                // 256 elements - uses tiled transpose
                test_permute_small_transpose::<TestRuntime, NumericT>(&Default::default(), 16);
            }

            #[test]
            pub fn test_small_transpose_32x32() {
                // 1024 elements - uses tiled transpose
                test_permute_small_transpose::<TestRuntime, NumericT>(&Default::default(), 32);
            }
    };
    ([$($numeric:ident),*]) => {
        mod permute {
            use super::*;
            ::paste::paste! {
                $(mod [<$numeric _ty>] {
                    use super::*;

                    $crate::testgen_tensor_permute!($numeric);
                })*
            }
        }
    };
}
