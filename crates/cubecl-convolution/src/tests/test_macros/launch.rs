#[macro_export]
macro_rules! testgen_convolution_launch {
    ($algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use super::*;
        use $crate::tests::test_algo;

        #[test]
        pub fn test() {
            test_algo::<$algorithm, $precision, TestRuntime>($selection, $problem);
        }
    };
}
