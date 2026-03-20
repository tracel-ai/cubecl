use crate::{Runtime, prelude::*};
use alloc::vec::Vec;
use cubecl_common::device::Device;

pub fn test_all_reduce_sync_collective<R: Runtime>() {
    let type_id = 0;
    let client = R::client(&Default::default());
    let device_ids = client.enumerate_devices(type_id);
    let device_count = device_ids.len();

    if device_count < 2 {
        return;
    }
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id))
        .collect();

    const SIZE: usize = 100;
    const NUM_HANDLES: usize = 8;

    let jobs = devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let client = R::client(device);
            let handles = (0..NUM_HANDLES)
                .map(|j| {
                    let src = [i as f32 + j as f32; SIZE];
                    client.create_from_slice(f32::as_bytes(&src))
                })
                .collect::<Vec<_>>();
            (client, handles)
        })
        .collect::<Vec<_>>();

    for (client, handles) in jobs.iter() {
        for handle in handles.iter() {
            client.all_reduce(
                handle.clone(),
                handle.clone(),
                cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32),
                device_ids.clone(),
                cubecl_runtime::server::ReduceOperation::Sum,
            );
        }

        // We perform the collective sync AFTER all all_reduce calls.
        client.sync_collective();
    }

    let value_base: f32 = device_ids.iter().map(|id| id.index_id as f32).sum();

    for (client, handles) in jobs.into_iter() {
        for (j, handle) in handles.into_iter().enumerate() {
            let actual = client.read_one(handle).unwrap();
            let actual = f32::from_bytes(&actual);
            let expected = [value_base + j as f32 * device_count as f32; SIZE];
            assert_eq!(actual, expected);
        }
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all_reduce {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_sync_collective() {
            cubecl_core::runtime_tests::all_reduce::test_all_reduce_sync_collective::<TestRuntime>(
            );
        }
    };
}
