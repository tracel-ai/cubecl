use crate::{self as cubecl, as_bytes, runtime_tests::memcpy_async, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube]
fn tile_computation<F: Float>(
    lhs: Slice<Line<F>>,
    rhs: Slice<Line<F>>,
    mut out: SliceMut<Line<F>>,
) {
    if UNIT_POS_X == 0 {
        for i in 0..2 {
            out[i] = lhs[i];
        }
        for i in 0..2 {
            out[i + 2] = rhs[i];
        }
    }
}

#[cube]
fn memcpy_sync<F: Float>(source: Slice<Line<F>>, mut destination: SliceMut<Line<F>>) {
    for i in 0..source.len() {
        destination[i] = source[i];
    }
}

#[cube(launch)]
fn computation<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(2u32, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(2u32, 1u32);

    let pipeline = Pipeline::new();

    pipeline.producer_acquire();

    let start = 0u32;
    let end = 2u32;
    if UNIT_POS_X == 0 {
        pipeline.memcpy_async(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    } else {
        // pipeline.memcpy_async(rhs.slice(start, end), rhs_smem.slice_mut(start, end));
    }

    pipeline.producer_commit();

    sync_units();

    pipeline.consumer_wait();
    // Perform matmul on SMEM
    tile_computation(
        lhs_smem.slice(start, end),
        rhs_smem.slice(start, end),
        output.slice_mut(start, end),
    );
    pipeline.consumer_release();
}

pub fn test_memcpy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    // if !client.properties().feature_enabled(Feature::Pipeline) {
    //     // We can't execute the test, skip.
    //     return;
    // }

    let lhs = client.create(as_bytes![F: 10., 11., 12., 13., 14., 15.]);
    let rhs = client.create(as_bytes![F: 10., 11., 12., 13.]);
    let output = client.empty(4 * core::mem::size_of::<F>());

    unsafe {
        computation::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&rhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [
        F::new(10.0),
        F::new(11.0),
        F::new(12.0),
        F::new(13.0),
        F::new(14.0),
        F::new(15.0),
    ];

    assert_eq!(actual, expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_memcpy_async {
    () => {
        use super::*;

        #[test]
        fn test_memcpy_async() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::memcpy_async::test_memcpy::<TestRuntime, FloatType>(client);
        }
    };
}
