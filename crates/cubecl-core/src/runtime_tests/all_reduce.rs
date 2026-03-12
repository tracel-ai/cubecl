use alloc::vec::Vec;
use std::println;

use cubecl_common::device::{Device, DeviceId};

use crate::Runtime;
use crate::prelude::*;

pub fn test_all_reduce_sync_collective<R: Runtime>() {
    let type_id = 0;
    let device_count = R::Device::device_count(type_id);

    if device_count < 2 {
        return;
    }
    let device_ids: Vec<DeviceId> = (0..device_count)
        .map(|i| DeviceId::new(type_id, i as u32))
        .collect();
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id).into())
        .collect();

    println!("All reduce between {devices:?} ...");

    const SIZE: usize = 100;
    const NUM_LOOP: usize = 10;

    let handles = devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let client = R::client(&device);
            let src = [i as f32; SIZE];
            let handle = client.create_from_slice(f32::as_bytes(&src));
            (client, handle)
        })
        .collect::<Vec<_>>();

    for (client, handle) in handles.iter() {
        // We call all_reduce multiple times (for no good reason).
        for _ in 0..NUM_LOOP {
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

    for (client, handle) in handles.into_iter() {
        let actual = client.read_one(handle).unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = [value_base * NUM_LOOP as f32 * device_count as f32; SIZE];
        assert_eq!(actual, expected);
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
