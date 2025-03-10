use crate::{self as cubecl, as_bytes, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube(launch)]
/// Calculate the sum of an array, using pipelining
fn pipelined_sum<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] batch_len: u32,
) {
    let smem_size = 2 * batch_len;
    let num_batches = input.len() / batch_len;
    let mut shared_memory = SharedMemory::<F>::new_lined(smem_size, input.line_size());
    let pipeline = Pipeline::new(2);

    let mut sum = Line::<F>::empty(input.line_size()).fill(F::new(0.));

    // Copy the first batch to shared memory
    pipeline.producer_acquire();
    pipeline.memcpy_async(
        &input.slice(0, batch_len),
        &mut shared_memory.slice_mut(0, batch_len),
    );
    pipeline.producer_commit();

    for input_batch in 1..num_batches {
        // Copy and compute index always alternate
        let copy_index = input_batch % 2;
        let compute_index = (input_batch + 1) % 2;

        // Copy the next batch to shared memory
        pipeline.producer_acquire();
        pipeline.memcpy_async(
            &input.slice(batch_len * input_batch, batch_len * (input_batch + 1)),
            &mut shared_memory.slice_mut(batch_len * copy_index, batch_len * (copy_index + 1)),
        );
        pipeline.producer_commit();

        // Compute the batch that is ready
        pipeline.consumer_wait();
        let compute_slice =
            shared_memory.slice(batch_len * compute_index, batch_len * (compute_index + 1));
        for i in 0..batch_len {
            sum += compute_slice[i];
        }
        pipeline.consumer_release();
    }

    // Compute the last batch
    pipeline.consumer_wait();
    let compute_slice = shared_memory.slice(
        batch_len * ((num_batches + 1) % 2),
        batch_len * ((num_batches + 1) % 2 + 1),
    );
    for i in 0..batch_len {
        sum += compute_slice[i];
    }
    pipeline.consumer_release();

    output[0] = sum;
}

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let pipeline = pipeline::Pipeline::<F>::new(2);
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    if UNIT_POS == 0 {
        let source = input.slice(2, 3);
        let mut destination = smem.slice_mut(0, 1);

        pipeline.producer_acquire();
        pipeline.memcpy_async(&source, &mut destination);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        output[0] = smem[0];
        pipeline.consumer_release();
    }
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        async_copy_test::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

pub fn test_pipelined_sum<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        pipelined_sum::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 8, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
            2,
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(36.0));
}

#[cube(launch)]
fn one_load<F: Float>(lhs: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let pipeline = Pipeline::new(1);

    let start = UNIT_POS_X * 2u32;
    let end = start + 2u32;

    pipeline.producer_acquire();
    pipeline.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
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

    let pipeline = Pipeline::new(1);

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    pipeline.producer_acquire();
    pipeline.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    pipeline.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));
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
fn two_independent_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32,
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let pipeline = Pipeline::new(2);

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    for i in start..end {
        lhs_smem[i] = Line::cast_from(0u32);
        rhs_smem[i] = Line::cast_from(0u32);
        output[i] = Line::cast_from(0u32);
    }

    pipeline.producer_acquire();
    pipeline.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    pipeline.producer_acquire();
    pipeline.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));
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
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

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
    independent: bool,
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let num_data = 4;
    let lhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();
    let rhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();

    let lhs = client.create(F::as_bytes(&lhs_data));
    let rhs = client.create(F::as_bytes(&rhs_data));
    let output = client.empty(2 * core::mem::size_of::<F>());

    if independent {
        unsafe {
            two_independent_loads::launch::<F, R>(
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_pipeline {
    () => {
        use super::*;

        #[test]
        fn test_pipeline_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_async_copy::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_pipeline_sum() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_pipelined_sum::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_pipeline_memcpy_async_one_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_memcpy_one_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_pipeline_memcpy_async_two_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_memcpy_two_loads::<TestRuntime, FloatType>(
                false, client,
            );
        }

        #[test]
        fn test_pipeline_memcpy_async_two_independent_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_memcpy_two_loads::<TestRuntime, FloatType>(
                true, client,
            );
        }
    };
}
