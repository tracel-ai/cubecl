use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let pipeline = pipeline::Pipeline::<F>::new();
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    if UNIT_POS == 0 {
        let source = input.slice(2, 3);
        let destination = smem.slice_mut(0, 1);

        // pipeline.producer_acquire();
        pipeline.memcpy_async(source, destination);
        // pipeline.producer_commit();

        // pipeline.consumer_await();
        output[0] = smem[0];
        // pipeline.consumer_release();
    }
}

pub fn test_pipeline<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_pipeline {
    () => {
        use super::*;

        #[test]
        fn test_pipeline() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_pipeline::<TestRuntime, FloatType>(client);
        }
    };
}
