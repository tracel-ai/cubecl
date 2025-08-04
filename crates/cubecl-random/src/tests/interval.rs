#[macro_export]
macro_rules! testgen_random_interval {
    () => {
        mod test_random_interval {
            use super::*;

            #[cube(launch)]
            pub(crate) fn kernel_to_unit_interval_co(input: &Array<u32>, output: &mut Array<f32>) {
                output[ABSOLUTE_POS] = super::to_unit_interval_closed_open(input[ABSOLUTE_POS]);
            }

            #[cube(launch)]
            pub(crate) fn kernel_to_unit_interval_oo(input: &Array<u32>, output: &mut Array<f32>) {
                output[ABSOLUTE_POS] = super::to_unit_interval_open(input[ABSOLUTE_POS]);
            }

            #[test]
            fn values_closed_open_interval() {
                let client = TestRuntime::client(&Default::default());

                let input = client.create(u32::as_bytes(&[0, u32::MAX]));
                let output = client.empty(input.size() as usize);

                kernel_to_unit_interval_co::launch::<TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::default(),
                    unsafe { ArrayArg::from_raw_parts::<u32>(&input, 2, 1) },
                    unsafe { ArrayArg::from_raw_parts::<f32>(&output, 2, 1) },
                );

                let actual = client.read_one(output);
                let actual = f32::from_bytes(&actual);

                // Previous implementation would map to `[0, 1]` but the interval should be half open `[0, 1)`
                assert_eq!(
                    actual[0], 0.0,
                    "Expected 0 to map to 0.0, but got {}",
                    actual[0]
                );

                assert!(
                    actual[1] < 1.0,
                    "Expected u32::MAX to map to a value strictly less than 1.0, but got {}",
                    actual[1]
                );

                assert!(
                    actual[1] >= 0.9999999,
                    "Expected u32::MAX to map close to 1.0, but got {}",
                    actual[1]
                );
            }

            #[test]
            fn values_open_interval() {
                let client = TestRuntime::client(&Default::default());

                let input = client.create(u32::as_bytes(&[0, u32::MAX]));
                let output = client.empty(input.size() as usize);

                kernel_to_unit_interval_oo::launch::<TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::default(),
                    unsafe { ArrayArg::from_raw_parts::<u32>(&input, 2, 1) },
                    unsafe { ArrayArg::from_raw_parts::<f32>(&output, 2, 1) },
                );

                let actual = client.read_one(output);
                let actual = f32::from_bytes(&actual);

                assert!(
                    actual[0] > 0.0,
                    "Expected 0 to map to a value strictly greater than 0.0, but got {}",
                    actual[0]
                );

                assert!(
                    actual[1] < 1.0,
                    "Expected u32::MAX to map to a value strictly less than 1.0, but got {}",
                    actual[1]
                );

                assert!(
                    actual[1] >= 0.9999999,
                    "Expected u32::MAX to map close to 1.0, but got {}",
                    actual[1]
                );
            }
        }
    };
}
