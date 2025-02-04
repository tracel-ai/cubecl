use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;
use pipeline::{Pipeline, PipelineGroup};

#[cube]
fn memcpy_sync<F: Float>(source: Slice<Line<F>>, mut destination: SliceMut<Line<F>>) {
    for i in 0..source.len() {
        destination[i] = source[i];
    }
}

#[cube(launch)]
fn computation<F: Float>(lhs: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let pipeline = Pipeline::new(2u32, PipelineGroup::Unit);

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

pub fn test_memcpy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let lhs = client.create(as_bytes![F: 10., 11., 12., 13.]);
    let output = client.empty(4 * core::mem::size_of::<F>());

    unsafe {
        computation::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [F::new(10.0), F::new(11.0), F::new(12.0), F::new(13.0)];

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
