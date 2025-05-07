use cubecl::prelude::*;
use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};
use cubecl_core as cubecl;

use crate::{random_bernoulli, random_normal, random_uniform};

fn test_results<E: Numeric>(
    data: &[E],
    expected_mean: f32,
    expected_std: f32,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
) {
    let mut sum = 0.;
    for elem in data {
        let elem = elem.to_f32().unwrap();
        if let Some(lower_bound) = lower_bound {
            assert!(
                elem >= lower_bound,
                "element is below lower bound: {} < {}",
                elem,
                lower_bound
            );
        }
        if let Some(upper_bound) = upper_bound {
            assert!(
                elem <= upper_bound,
                "element is above upper bound: {} > {}",
                elem,
                upper_bound
            );
        }
        sum += elem;
    }
    let mean = sum / (data.len() as f32);

    let mut sum = 0.0;
    for elem in data {
        let elem = elem.to_f32().unwrap();
        let d = elem - mean;
        sum += d * d;
    }
    let var = sum / (data.len() as f32);
    let std = var.sqrt();

    assert!(
        ((mean - expected_mean).abs() / var) < 3.0,
        "Uniform RNG validation failed: mean={}, expected mean={}, std={}, expected std={}",
        mean,
        expected_mean,
        std,
        expected_std,
    );
}

pub fn test_random_uniform<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &[usize],
    lower_bound: E,
    upper_bound: E,
) {
    let elem_size = size_of::<E>();
    let (handle, strides) = client.empty_tensor(shape, elem_size);
    let mut output: TensorHandleRef<'_, R> =
        unsafe { TensorHandleRef::from_raw_parts(&handle, strides.as_slice(), shape, elem_size) };

    // generate random tensor
    random_uniform(client, lower_bound, upper_bound, &mut output);

    assert_eq!(output.shape, shape);

    let random_data = client.read_one(handle.binding());
    let random_data = E::from_bytes(&random_data);
    println!("{:?}", random_data);

    let range = (upper_bound + lower_bound).to_f32().unwrap();
    let expected_mean = range / 2.;
    let expected_std = (range * range / 12.).sqrt();
    let lower_bound = lower_bound.to_f32().unwrap();
    let upper_bound = upper_bound.to_f32().unwrap();

    test_results(
        random_data,
        expected_mean,
        expected_std,
        Some(lower_bound),
        Some(upper_bound),
    );
}

pub fn test_random_bernoulli<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &[usize],
    probability: f32,
) {
    let elem_size = size_of::<E>();
    let (handle, strides) = client.empty_tensor(shape, elem_size);
    let mut output: TensorHandleRef<'_, R> =
        unsafe { TensorHandleRef::from_raw_parts(&handle, strides.as_slice(), shape, elem_size) };

    // generate random tensor
    random_bernoulli::<R, E>(client, probability, &mut output);

    assert_eq!(output.shape, shape);

    let random_data = client.read_one(handle.binding());
    let random_data = E::from_bytes(&random_data);
    println!("{:?}", random_data);

    let expected_mean = probability;
    let expected_std = (expected_mean * (1. - expected_mean)).sqrt();
    let lower_bound = 0.;
    let upper_bound = 1.;

    test_results(
        random_data,
        expected_mean,
        expected_std,
        Some(lower_bound),
        Some(upper_bound),
    );
}

pub fn test_random_normal<R: Runtime, E: CubeElement + Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &[usize],
    mean: E,
    std: E,
) {
    let elem_size = size_of::<E>();
    let (handle, strides) = client.empty_tensor(shape, elem_size);
    let mut output: TensorHandleRef<'_, R> =
        unsafe { TensorHandleRef::from_raw_parts(&handle, strides.as_slice(), shape, elem_size) };

    // generate random tensor
    random_normal(client, mean, std, &mut output);

    assert_eq!(output.shape, shape);

    let random_data = client.read_one(handle.binding());
    let random_data = E::from_bytes(&random_data);
    println!("{:?}", random_data);

    let expected_mean = mean.to_f32().unwrap();
    let expected_std = std.to_f32().unwrap();
    test_results(random_data, expected_mean, expected_std, None, None);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_random {
    () => {
        use super::*;

        #[test]
        fn test_random_uniform() {
            let client = TestRuntime::client(&Default::default());
            // doesn't work with f64
            cubecl_random::test::test_random_uniform::<TestRuntime, f32>(
                &client,
                &[10, 10],
                0.0,
                100.0,
            );
            cubecl_random::test::test_random_uniform::<TestRuntime, i64>(
                &client,
                &[10, 10],
                0,
                100,
            );
            cubecl_random::test::test_random_uniform::<TestRuntime, i32>(
                &client,
                &[10, 10],
                0,
                100,
            );
        }

        #[test]
        fn test_random_bernoulli() {
            let client = TestRuntime::client(&Default::default());
            // doesn't work with f64
            cubecl_random::test::test_random_bernoulli::<TestRuntime, f32>(&client, &[10, 10], 0.2);
            cubecl_random::test::test_random_bernoulli::<TestRuntime, i64>(&client, &[10, 10], 0.2);
            cubecl_random::test::test_random_bernoulli::<TestRuntime, i32>(&client, &[10, 10], 0.2);
        }

        #[test]
        fn test_random_normal() {
            let client = TestRuntime::client(&Default::default());
            // doesn't work with f64
            cubecl_random::test::test_random_normal::<TestRuntime, f32>(
                &client,
                &[10, 10],
                13.0,
                100.0,
            );
            cubecl_random::test::test_random_normal::<TestRuntime, i64>(
                &client,
                &[10, 10],
                13,
                100,
            );
            cubecl_random::test::test_random_normal::<TestRuntime, i32>(
                &client,
                &[10, 10],
                13,
                100,
            );
        }
    };
}
