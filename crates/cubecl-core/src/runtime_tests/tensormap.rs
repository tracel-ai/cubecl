use std::fmt::Debug;

use crate::{
    self as cubecl,
    prelude::barrier::{ArrivalToken, Barrier, BarrierLevel},
    Feature, TmaFeature,
};

use cubecl::prelude::*;
use cubecl_common::TensorMapFormat;
use cubecl_runtime::{server::ComputeServer, storage::ComputeStorage};

#[cube(launch)]
pub fn tensormap_load<F: Float>(input: &TensorMap<F>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::<F>::new_proxied(BarrierLevel::cube_coop(0u32));
    let mut token = ArrivalToken::new();
    let mut stage = SharedMemory::<F>::new_lined(32u32 * 32, 1u32);

    if UNIT_POS == 0 {
        barrier.memcpy_async_bulk_to_shared_2d(input, &mut stage.to_slice_mut(), 8, 8);
        barrier.arrive_tx(1, 32 * 32 * F::elem_size(), &mut token);
    } else {
        barrier.arrive(&mut token);
    }
    barrier.wait(token);

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos] = stage[out_pos];
}

pub fn test_tensormap_load<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..64 * 64).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let handle = client.create(F::as_bytes(&values));
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, &[64, 1], &[64, 64], 1) };
    let out = client.empty(32 * 32 * size_of::<F>());

    tensormap_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 32),
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![32, 32],
            },
            input,
            2,
            F::as_elem_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts::<F>(&out, 32 * 32, 1) },
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);
    let expected: Vec<F> = (8..40)
        .flat_map(|i| i * 64..i * 64 + 32)
        .map(|it| F::from_int(it + 8))
        .collect();

    assert_eq!(actual, &expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tensormap {
    () => {
        use super::*;

        #[test]
        fn test_tensormap_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
