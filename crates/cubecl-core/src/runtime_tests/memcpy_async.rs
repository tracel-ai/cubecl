use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;
use pipeline::{Pipeline, PipelineGroup};

#[cube(launch)]
fn one_load<F: Float>(lhs: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let pipeline = Pipeline::new(1u32, PipelineGroup::Unit);

    let start = UNIT_POS_X * 2u32;
    let end = start + 2u32;

    pipeline.producer_acquire();
    pipeline.memcpy_async(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    pipeline.consumer_wait();
    for i in start..end {
        output[i] = lhs_smem[i];
    }
    pipeline.consumer_release();
}

#[cube(launch)]
fn two_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32, // should be even
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let pipeline = Pipeline::new(1u32, PipelineGroup::Unit);

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    pipeline.producer_acquire();
    pipeline.memcpy_async(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    pipeline.memcpy_async(rhs.slice(start, end), rhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    pipeline.consumer_wait();
    let mut dot = Line::cast_from(0u32);
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }
    pipeline.consumer_release();

    output[UNIT_POS_X] = dot;
}

#[cube(launch)]
fn two_independant_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32,
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let pipeline = Pipeline::new(2u32, PipelineGroup::Unit);

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    for i in start..end {
        lhs_smem[i] = Line::cast_from(0u32);
        rhs_smem[i] = Line::cast_from(0u32);
        output[i] = Line::cast_from(0u32);
    }

    pipeline.producer_acquire();
    pipeline.memcpy_async(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    pipeline.producer_acquire();
    pipeline.memcpy_async(rhs.slice(start, end), rhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    let mut dot = Line::cast_from(0u32);

    pipeline.consumer_wait();
    pipeline.consumer_wait();
    pipeline.consumer_wait();
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }
    pipeline.consumer_release();
    pipeline.consumer_release();
    pipeline.consumer_release();

    output[UNIT_POS_X] = dot;
}

pub fn test_memcpy_one_load<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let lhs = client.create(as_bytes![F: 10., 11., 12., 13.]);
    let output = client.empty(4 * core::mem::size_of::<F>());

    unsafe {
        one_load::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [F::new(10.0), F::new(11.0), F::new(12.0), F::new(13.0)];

    assert_eq!(actual, expected);
}

pub fn test_memcpy_two_loads<R: Runtime, F: Float + CubeElement>(
    independant: bool,
    client: ComputeClient<R::Server, R::Channel>,
) {
    let num_data = 4;
    let lhs_data = generate_random_data(num_data, 42);
    let rhs_data = generate_random_data(num_data, 43);

    let lhs = client.create(F::as_bytes(&lhs_data));
    let rhs = client.create(F::as_bytes(&rhs_data));
    let output = client.empty(2 * core::mem::size_of::<F>());

    if independant {
        unsafe {
            two_independant_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    } else {
        unsafe {
            two_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    }

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    let middle = num_data / 2;
    let expected = [
        dot(&lhs_data[..middle], &rhs_data[..middle]),
        dot(&lhs_data[middle..], &rhs_data[middle..]),
    ];

    assert_eq!(actual, expected);
}

fn dot<F: Float>(vec1: &[F], vec2: &[F]) -> F {
    let mut sum = F::from_int(0);
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

// TODO tmp
pub(crate) fn generate_random_data<F: Float + CubeElement>(
    num_elements: usize,
    mut seed: u64,
) -> Vec<F> {
    fn lcg(seed: &mut u64) -> f32 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: f64 = 2u64.pow(32) as f64;

        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
        (*seed as f64 / M * 2.0 - 1.0) as f32
    }

    (0..num_elements).map(|_| F::new(lcg(&mut seed))).collect()
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_memcpy_async {
    () => {
        use super::*;

        #[test]
        fn test_memcpy_async_one_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::memcpy_async::test_memcpy_one_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_memcpy_async_two_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::memcpy_async::test_memcpy_two_loads::<TestRuntime, FloatType>(
                false,
                client,
            );
        }

        #[test]
        fn test_memcpy_async_two_independant_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::memcpy_async::test_memcpy_two_loads::<TestRuntime, FloatType>(
                true,
                client,
            );
        }
    };
}
