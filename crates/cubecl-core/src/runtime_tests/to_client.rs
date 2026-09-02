use alloc::vec::Vec;
use std::println;

use cubecl_common::device::{Device, DeviceId};

use crate::Runtime;
use crate::prelude::*;

pub fn test_to_client<R: Runtime>() {
    // Every device the runtime can reach, not just one type: a machine with a
    // single discrete GPU still has other devices to move data between, and
    // this test is the only cover for the cross-device transfer path.
    let devices = R::enumerate_all_devices();

    if devices.len() < 2 {
        return;
    }

    for (device_0, device_1) in num_combination(&devices) {
        let device_0 = R::Device::from_id(device_0);
        let device_1 = R::Device::from_id(device_1);

        println!("Moving data from {device_0:?} to {device_1:?} ...");

        let mut client_0 = R::client(&device_0);
        let client_1 = R::client(&device_1);

        let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = client_0.create_from_slice(f32::as_bytes(&expected));

        let output = client_0.to_client(
            input,
            &client_1,
            cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32),
        );

        let actual = client_1.read_one_unchecked(output);
        let actual = f32::from_bytes(&actual);

        assert_eq!(actual, expected);
    }
}

fn num_combination(devices: &[DeviceId]) -> Vec<(DeviceId, DeviceId)> {
    let mut results = Vec::new();

    for i in 0..devices.len() {
        for j in i + 1..devices.len() {
            results.push((devices[i], devices[j]));
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
    };
}
