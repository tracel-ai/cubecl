#[macro_export]
macro_rules! testgen_random_bernoulli {
    () => {
        mod test_random_bernoulli {
            use super::*;

            pub fn get_random_bernoulli_data<R: Runtime, E: CubeElement + Numeric>(
                shape: &[usize],
                prob: f32,
            ) -> Vec<E> {
                seed(0);

                let client = R::client(&Default::default());
                let output = TensorHandle::<R, E>::empty(&client, shape.to_vec());

                random_bernoulli::<R, E>(&client, prob, output.as_ref());

                let output_data = client.read_one_tensor(output.as_copy_descriptor());
                let output_data = E::from_bytes(&output_data);

                output_data.to_owned()
            }

            #[test]
            fn number_of_1_proportional_to_prob_f32() {
                let client = TestRuntime::client(&Default::default());
                let shape = &[40, 40];
                let prob = 0.7;

                let output_data = get_random_bernoulli_data::<TestRuntime, f32>(shape, prob);

                assert_number_of_1_proportional_to_prob(&output_data, prob);
            }

            #[test]
            fn number_of_1_proportional_to_prob_i32() {
                let client = TestRuntime::client(&Default::default());
                let shape = &[40, 40];
                let prob = 0.7;

                let output_data = get_random_bernoulli_data::<TestRuntime, i32>(shape, prob);

                assert_number_of_1_proportional_to_prob(&output_data, prob);
            }

            #[test]
            fn wald_wolfowitz_runs_test() {
                let shape = &[512, 512];

                let output_data = get_random_bernoulli_data::<TestRuntime, f32>(shape, 0.5);

                // High bound slightly over 1 so 1.0 is included in second bin
                assert_wald_wolfowitz_runs_test(&output_data, 0., 1.1);
            }
        }
    };
}
