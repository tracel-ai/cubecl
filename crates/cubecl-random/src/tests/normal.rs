#[macro_export]
macro_rules! testgen_random_normal {
    () => {
        mod test_random_normal {
            use super::*;

            pub fn get_random_normal_data<R: Runtime, E: CubeElement + Numeric>(
                shape: &[usize],
                mean: E,
                std: E,
            ) -> Vec<E> {
                seed(0);

                let client = R::client(&Default::default());
                let output = TensorHandle::<R, E>::empty(&client, shape.to_vec());

                random_normal::<R, E>(&client, mean, std, output.as_ref());

                let output_data = client.read_one_tensor(output.as_copy_descriptor());
                let output_data = E::from_bytes(&output_data);

                output_data.to_owned()
            }

            #[test]
            fn empirical_mean_close_to_expectation() {
                let client = TestRuntime::client(&Default::default());
                let shape = &[100, 100];
                let mean = 10.;
                let std = 2.;

                let output_data = get_random_normal_data::<TestRuntime, f32>(shape, mean, std);

                assert_mean_approx_equal(&output_data, mean);

                let shape = &[1000, 1000];
                let mean = 0.;
                let std = 1.;

                let output_data = get_random_normal_data::<TestRuntime, f32>(shape, mean, std);

                assert_mean_approx_equal(&output_data, mean);
            }

            #[test]
            fn normal_respects_68_95_99_rule() {
                // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
                let client = TestRuntime::client(&Default::default());
                let shape = &[1000, 1000];
                let mu = 0.;
                let s = 1.;

                let output_data = get_random_normal_data::<TestRuntime, f32>(shape, mu, s);

                assert_normal_respects_68_95_99_rule(&output_data, mu, s);
            }
        }
    };
}
