use alloc::vec::Vec;
use core::time::Duration;
use std::println;
use std::thread;
use std::thread::spawn;
use std::vec;

use cubecl_common::device::{Device, DeviceId};

use crate::Runtime;
use crate::prelude::*;

pub fn test_all_reduce<R: Runtime>() {
    let type_id = 0;
    let device_count = R::Device::device_count(type_id);

    if device_count < 2 {
        return;
    }

    for (device_0_id, device_1_id) in num_combination(type_id, device_count as u32) {
        let device_0 = R::Device::from_id(device_0_id);
        let device_1 = R::Device::from_id(device_1_id);

        let device_ids = vec![device_0_id, device_1_id];

        println!("All reduce between {device_0:?} and {device_1:?} ...");

        let expected = [device_1_id.index_id as f32 + device_0_id.index_id as f32; 5];
        let device_list = vec![device_0, device_1];
        let mut handles = vec![];
        for (device, device_id) in device_list
            .iter()
            .zip(vec![device_0_id.index_id, device_1_id.index_id])
        {
            let device_ids_loop = device_ids.clone();
            let expected_loop = expected.clone();
            let client_loop = R::client(&device);
            let handle = spawn(move || {
                let src = [device_id as f32; 5];
                let input = client_loop.create_from_slice(f32::as_bytes(&src));

                client_loop.all_reduce(input.clone(), input.clone(), device_ids_loop);
                thread::sleep(Duration::from_millis(1000));

                let actual = client_loop.read_one(input).unwrap();
                let actual = f32::from_bytes(&actual);

                println!("actual, {:?}", actual);
                println!("expected, {:?}", expected);
                assert_eq!(actual, expected_loop);
            });
            handles.push(handle);
        }
        for h in handles {
            let _ = h.join();
        }
        break;
    }
}

pub fn test_all_reduce_sync<R: Runtime>() {
    let type_id = 0;
    let device_count = R::Device::device_count(type_id);

    if device_count < 2 {
        return;
    }

    // for i in 0..1000 {
    for (device_0_id, device_1_id) in num_combination(type_id, device_count as u32) {
        let device_0 = R::Device::from_id(device_0_id);
        let device_1 = R::Device::from_id(device_1_id);

        let device_ids = vec![device_0_id, device_1_id];

        println!("All reduce between {device_0:?} and {device_1:?} ...");

        let expected = [device_1_id.index_id as f32 + device_0_id.index_id as f32; 5];
        let device_list = vec![device_0, device_1];
        let mut buffers = vec![];
        for (device, device_id) in device_list
            .iter()
            .zip(vec![device_0_id.index_id, device_1_id.index_id])
        {
            let client = R::client(device);
            let src = [device_id as f32; 5];
            let input = client.create_from_slice(f32::as_bytes(&src));
            buffers.push(input);
        }

        let mut handles = vec![];
        for (i, device) in device_list.iter().enumerate() {
            let device_ids_loop = device_ids.clone();
            let client_loop = R::client(&device);
            let input_loop = buffers[i].clone();
            let handle = spawn(move || {
                client_loop.all_reduce(input_loop.clone(), input_loop.clone(), device_ids_loop);
                println!("All reduce launched");
                // thread::sleep(Duration::from_millis(1000));

                // handles.push(input);

                // let actual = client_loop.read_one(input).unwrap();
                // let actual = f32::from_bytes(&actual);

                // println!("actual, {:?}", actual);
                // println!("expected, {:?}", expected);
                // assert_eq!(actual, expected_loop);
            });
            handles.push(handle);
        }
        // for h in handles {
        //     let _ = h.join();
        // }

        // thread::sleep(Duration::from_millis(1000));

        for (i, device) in device_list.iter().enumerate() {
            let client_loop = R::client(&device);
            let src = buffers[i].clone();

            // client_loop.flush().unwrap();

            client_loop.sync_collective();
            let actual = client_loop.read_one(src).unwrap();
            let actual = f32::from_bytes(&actual);

            println!("actual, {:?}", actual);
            println!("expected, {:?}", expected);
            assert_eq!(actual, expected);
        }
        println!("\n =========================================================== \n")
    }
    // }
}

pub fn test_to_client<R: Runtime>() {
    let type_id = 0;
    let device_count = R::Device::device_count(type_id);

    if device_count < 2 {
        return;
    }

    for (device_0, device_1) in num_combination(type_id, device_count as u32) {
        let device_0 = R::Device::from_id(device_0);
        let device_1 = R::Device::from_id(device_1);

        println!("Moving data from {device_0:?} to {device_1:?} ...");

        let client_0 = R::client(&device_0);
        let client_1 = R::client(&device_1);

        let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = client_0.create_from_slice(f32::as_bytes(&expected));

        let output = client_0.to_client(input, &client_1);

        let actual = client_1.read_one_unchecked(output);
        let actual = f32::from_bytes(&actual);

        assert_eq!(actual, expected);
    }
}

fn num_combination(type_id: u16, n: u32) -> Vec<(DeviceId, DeviceId)> {
    let mut results = Vec::new();

    for i in 0..n {
        for j in i + 1..n {
            results.push((DeviceId::new(type_id, i), DeviceId::new(type_id, j)));
        }
    }

    results
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_to_client {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_to_client() {
            cubecl_core::runtime_tests::to_client::test_to_client::<TestRuntime>();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce() {
            cubecl_core::runtime_tests::to_client::test_all_reduce::<TestRuntime>();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_sync() {
            cubecl_core::runtime_tests::to_client::test_all_reduce_sync::<TestRuntime>();
        }
    };
}
