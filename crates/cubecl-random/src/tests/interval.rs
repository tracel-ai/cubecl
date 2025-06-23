#[macro_export]
macro_rules! testgen_random_interval {
    () => {
        mod test_random_interval {
            use super::*;

            #[test]
            fn values_half_bounded_interval() {
                let client = TestRuntime::client(&Default::default());
                test_kernel_to_unit_interval::<TestRuntime>(client)
            }
        }
    };
}
