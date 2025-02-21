use crate::{self as cubecl, as_bytes, Feature};
use cubecl::prelude::*;

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let barrier = barrier::Barrier::<F>::new_unit_level();
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    let source = input.slice(2, 3);
    let destination = smem.slice_mut(0, 1);

    barrier.memcpy_async(source, destination);

    barrier.wait();
    output[0] = smem[0];
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_barrier {
    () => {
        use super::*;

        #[test]
        fn test_barrier_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_async_copy::<TestRuntime, FloatType>(client);
        }
    };
}
