use crate::{self as cubecl, as_bytes, runtime_tests::memcpy_async, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube]
fn tile_computation<F: Float>(
    lhs: Slice<Line<F>>,
    rhs: Slice<Line<F>>,
    mut out: SliceMut<Line<F>>,
) {
    for i in 0..lhs.len() {
        out[i] = Line::cast_from(F::new(10.)) * lhs[i] + rhs[i];
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
    let unit_id = UNIT_POS_X;
    let mut lhs_smem = SharedMemory::<F>::new_lined(16u32, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(16u32, 1u32);

    let pipeline = Pipeline::new();

    // Load Lhs to SMEM
    pipeline.memcpy_async(
        lhs.slice(unit_id * 2u32, unit_id * 2u32 + 2u32),
        lhs_smem.slice_mut(unit_id * 2u32, unit_id * 2u32 + 2u32),
    );

    // Load Rhs to SMEM
    pipeline.memcpy_async(
        rhs.slice(unit_id * 2u32, unit_id * 2u32 + 2u32),
        rhs_smem.slice_mut(unit_id * 2u32, unit_id * 2u32 + 2u32),
    );

    // sync
    sync_units();

    // Perform matmul on SMEM
    tile_computation(
        lhs_smem.slice(unit_id * 2u32, unit_id * 2u32 + 2u32),
        rhs_smem.slice(unit_id * 2u32, unit_id * 2u32 + 2u32),
        output.slice_mut(unit_id * 2u32, unit_id * 2u32 + 2u32),
    );
}

pub fn test_memcpy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    // if !client.properties().feature_enabled(Feature::Pipeline) {
    //     // We can't execute the test, skip.
    //     return;
    // }

    let lhs = client
        .create(as_bytes![F: 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]);
    let rhs = client
        .create(as_bytes![F: 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]);
    let output = client.empty(16 * core::mem::size_of::<F>());

    unsafe {
        computation::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(8, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&rhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [
        F::new(0.0),
        F::new(11.0),
        F::new(22.0),
        F::new(33.0),
        F::new(44.0),
        F::new(55.0),
        F::new(66.0),
        F::new(77.0),
        F::new(88.0),
        F::new(99.0),
        F::new(110.0),
        F::new(121.0),
        F::new(132.0),
        F::new(143.0),
        F::new(154.0),
        F::new(165.0),
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
