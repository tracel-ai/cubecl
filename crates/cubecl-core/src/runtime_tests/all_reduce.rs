use alloc::vec::Vec;
use std::println;
use std::thread::spawn;
use std::vec;

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

    let value: f32 = device_ids.iter().map(|id| id.index_id as f32).sum();
    const SIZE: usize = 100;
    let expected = [value; SIZE];

    let mut handles = vec![];
    for (i, device) in devices.iter().enumerate() {
        let device_ids_loop = device_ids.clone();
        let client_loop = R::client(&device);
        let expected_loop = expected.clone();
        let device_index = device_ids[i].index_id;
        let handle = spawn(move || {
            for _ in 0..10 {
                let src = [device_index as f32; SIZE];
                let input = client_loop.create_from_slice(f32::as_bytes(&src));
                client_loop.all_reduce(
                    input.clone(),
                    input.clone(),
                    cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32),
                    device_ids_loop.clone(),
                    cubecl_runtime::server::ReduceOperation::Sum,
                );

                client_loop.sync_collective();
                let actual = client_loop.read_one(input).unwrap();
                let actual = f32::from_bytes(&actual);
                assert_eq!(actual, expected_loop);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
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
