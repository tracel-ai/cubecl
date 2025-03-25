use std::fmt::Debug;

use crate::{
    self as cubecl, Feature, TmaFeature,
    prelude::barrier::{Barrier, BarrierLevel},
};

use cubecl::prelude::*;
use cubecl_runtime::{server::ComputeServer, storage::ComputeStorage};

#[cube(launch)]
pub fn tensormap_load<F: Float>(input: &TensorMap<F>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::<F>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
    let mut stage = SharedMemory::<F>::new_aligned(32u32 * 16, 1u32, 128u32);

    if UNIT_POS == 0 {
        barrier.memcpy_async_tensor_to_shared_2d(input, &mut stage.to_slice_mut(), 0, 8);
        barrier.arrive_tx(1, 32 * 16 * F::elem_size());
    } else {
        barrier.arrive();
    }
    barrier.wait();

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos] = stage[out_pos];
}

#[cube(launch)]
pub fn tensormap_store<F: Float>(input: &Array<Line<F>>, output: &mut TensorMap<F>) {
    let mut shared = SharedMemory::new_aligned(32u32 * 16, 1u32, 128u32);

    let in_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    shared[in_pos] = input[in_pos];

    sync_proxy_shared();
    sync_units();

    if UNIT_POS == 0 {
        memcpy_async_tensor_to_global_2d(&shared.to_slice(), output, 16, 8);
        memcpy_async_tensor_commit();
        memcpy_async_tensor_wait_read(0u32);
    }
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
    let shape = vec![64, 64];
    let (handle, strides) = client.create_tensor(F::as_bytes(&values), &shape, size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, &strides, &shape, 1) };
    let out = client.empty(16 * 32 * size_of::<F>());

    tensormap_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 32],
            },
            input,
            F::as_elem_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts::<F>(&out, 32 * 16, 1) },
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);
    let expected: Vec<F> = (0..16)
        .flat_map(|i| i * 64..i * 64 + 32)
        .map(|it| F::from_int(it + 8))
        .collect();

    assert_eq!(actual, &expected);
}

pub fn test_tensormap_store<R: Runtime, F: Float + CubeElement>(
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

    let values = (0..32 * 16).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let handle = client.create(F::as_bytes(&values));
    let (out, out_strides) = client.create_tensor(
        &vec![0u8; 64 * 64 * size_of::<F>()],
        &[64, 64],
        size_of::<F>(),
    );

    tensormap_store::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 32 * 16, 1) },
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 32],
            },
            unsafe { TensorArg::from_raw_parts::<F>(&out, &out_strides, &[64, 64], 1) },
            F::as_elem_native_unchecked(),
        ),
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);
    let mut expected: Vec<F> = vec![F::from_int(0); 64 * 64];
    for y in 0..16 {
        for x in 0..32 {
            let val = y * 32 + x;
            let y = y + 16;
            let x = x + 8;
            let index = y * 64 + x;
            expected[index] = F::from_int(val as i64);
        }
    }

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

        #[test]
        fn test_tensormap_store() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_store::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
