use crate::{self as cubecl, as_bytes, runtime_tests::memcpy_async, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube]
fn tile_computation_1<F: Float>(
    lhs: Slice<Line<F>>,
    rhs: Slice<Line<F>>,
    mut out: SliceMut<Line<F>>,
) {
    for i in 0..2 {
        out[i] = Line::cast_from(10u32) * lhs[i];
    }
    for i in 0..2 {
        out[i] += rhs[i];
    }
}

#[cube]
fn tile_computation_2<F: Float>(
    lhs: &SharedMemory<Line<F>>,
    rhs: &SharedMemory<Line<F>>,
    out: &mut Tensor<Line<F>>,
    start: u32,
    end: u32,
) {
    for i in start..end {
        out[i] = Line::cast_from(10u32) * lhs[i];
    }
    for i in start..end {
        out[i] += rhs[i];
    }
}

#[cube]
fn tile_computation_3<F: Float>(
    lhs: &SharedMemory<Line<F>>,
    rhs: &SharedMemory<Line<F>>,
    out: &mut Tensor<Line<F>>,
    start: u32,
    end: u32,
    pipeline: Pipeline<F>,
) {
    pipeline.consumer_wait();
    for i in start..end {
        out[i] = Line::cast_from(10u32) * lhs[i];
    }
    for i in start..end {
        out[i] += rhs[i];
    }
    pipeline.consumer_release();
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
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let pipeline = Pipeline::new(2u32);

    let start = UNIT_POS_X * 2u32;
    let end = start + 2u32;

    pipeline.producer_acquire();
    pipeline.memcpy_async(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    // memcpy_sync(lhs.slice(start, end), lhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    pipeline.producer_acquire();
    pipeline.memcpy_async(rhs.slice(start, end), rhs_smem.slice_mut(start, end));
    // memcpy_sync(rhs.slice(start, end), rhs_smem.slice_mut(start, end));
    pipeline.producer_commit();

    // tile_computation_0: inline
    // pipeline.consumer_wait();
    // for i in start..end {
    //     output[i] = Line::cast_from(10u32) * lhs[i];
    // }
    // for i in start..end {
    //     output[i] += rhs[i];
    // }
    // pipeline.consumer_release();

    // pipeline.consumer_wait();
    // tile_computation_1(
    //     lhs_smem.slice(start, end),
    //     rhs_smem.slice(start, end),
    //     output.slice_mut(start, end),
    // );
    // pipeline.consumer_release();

    // pipeline.consumer_wait();
    // tile_computation_2(&lhs_smem, &rhs_smem, output, start, end);
    // pipeline.consumer_release();

    tile_computation_3(&lhs_smem, &rhs_smem, output, start, end, pipeline);
}

pub fn test_memcpy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    // if !client.properties().feature_enabled(Feature::Pipeline) {
    //     // We can't execute the test, skip.
    //     return;
    // }

    let lhs = client.create(as_bytes![F: 10., 11., 12., 13.]);
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
    let expected = [F::new(110.0), F::new(121.0), F::new(132.0), F::new(143.0)];

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
