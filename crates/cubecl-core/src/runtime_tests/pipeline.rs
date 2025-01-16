use crate::{self as cubecl, as_bytes, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube(launch)]
fn pipelined_computation<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] batch_len: u32,
) {
    let smem_size = 2 * batch_len;
    let num_batches = input.len() / batch_len;
    let mut shared_memory = SharedMemory::<F>::new_lined(smem_size, input.line_size());
    let pipeline = Pipeline::new();

    let mut sum = Line::<F>::empty(input.line_size()).fill(F::new(0.));

    pipeline.producer_acquire();
    pipeline.memcpy_async(
        input.slice(0, batch_len),
        shared_memory.slice_mut(0, batch_len),
    );
    pipeline.producer_commit();

    for batch in 1..num_batches {
        let copy_index = batch % 2;
        let compute_index = (batch + 1) % 2;

        pipeline.producer_acquire();
        pipeline.memcpy_async(
            input.slice(batch_len * batch, batch_len * (batch + 1)),
            shared_memory.slice_mut(batch_len * copy_index, batch_len * (copy_index + 1)),
        );
        pipeline.producer_commit();

        pipeline.consumer_await();
        let compute_slice = input.slice(batch_len * compute_index, batch_len * (compute_index + 1));
        for i in 0..compute_slice.len() {
            sum += compute_slice[i];
        }
        pipeline.consumer_release();
    }

    pipeline.consumer_await();
    let compute_slice = input.slice(batch_len * 1, batch_len * 2);
    for i in 0..compute_slice.len() {
        sum += compute_slice[i];
    }
    pipeline.consumer_release();

    output[0] = sum;
}

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let pipeline = pipeline::Pipeline::<F>::new();
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    if UNIT_POS == 0 {
        let source = input.slice(2, 3);
        let destination = smem.slice_mut(0, 1);

        pipeline.producer_acquire();
        pipeline.memcpy_async(source, destination);
        pipeline.producer_commit();

        pipeline.consumer_await();
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

pub fn test_pipelined_computation<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        pipelined_computation::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
            5,
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(10.0));
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
        fn test_pipeline_computation() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_pipelined_computation::<
                TestRuntime,
                FloatType,
            >(client);
        }
    };
}
