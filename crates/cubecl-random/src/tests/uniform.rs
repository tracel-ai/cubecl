#[macro_export]
macro_rules! testgen_random_uniform {
    () => {
        mod test_random_uniform {
            use super::*;

            pub fn get_random_uniform_data<R: Runtime, E: CubeElement + Numeric>(
                shape: &[usize],
                lower_bound: E,
                upper_bound: E,
            ) -> Vec<E> {
                seed(0);
                let client = R::client(&Default::default());

                let elem_size = size_of::<E>();
                let (handle, strides) = client.empty_tensor(shape, elem_size);
                let mut output: TensorHandleRef<'_, R> = unsafe {
                    TensorHandleRef::from_raw_parts(&handle, strides.as_slice(), shape, elem_size)
                };

                random_uniform::<R, E>(&client, lower_bound, upper_bound, &mut output);

                let output_data = client.read_one(handle.binding());
                let output_data = E::from_bytes(&output_data);

                output_data.to_owned()
            }

            #[test]
            fn values_all_within_interval_uniform() {
                let shape = &[24, 24];

                let output_data = get_random_uniform_data::<TestRuntime, f32>(shape, 5., 17.);

                for e in output_data {
                    assert!(e >= 5. && e < 17.);
                }
            }

            #[test]
            fn at_least_one_value_per_bin_uniform() {
                let shape = &[64, 64];

                let output_data = get_random_uniform_data::<TestRuntime, f32>(shape, -5., 10.);

                let stats = calculate_bin_stats(&output_data, 3, -5., 10.);
                assert!(stats[0].count >= 1);
                assert!(stats[1].count >= 1);
                assert!(stats[2].count >= 1);
            }

            #[test]
            fn runs_test() {
                let shape = &[512, 512];

                let output_data = get_random_uniform_data::<TestRuntime, f32>(shape, 0., 1.);

                assert_wald_wolfowitz_runs_test(&output_data, 0., 1.);
            }

            #[test]
            fn at_least_one_value_per_bin_int_uniform() {
                let shape = &[64, 64];
                let output_data = get_random_uniform_data::<TestRuntime, i32>(shape, -10, 10);

                assert_at_least_one_value_per_bin(&output_data, 10, -10., 10.);
            }
        }
    };
}
