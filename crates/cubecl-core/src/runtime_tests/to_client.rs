use cubecl_runtime::id::DeviceId;

use crate::prelude::*;
use crate::{Device, Runtime};

pub fn test_to_client<R: Runtime>() {
    let type_id = 0;
    let device_count = R::Device::device_count(type_id);

    if device_count < 2 {
        return;
    }

    for (device_0, device_1) in num_combination(type_id, device_count as u32) {
        let device_0 = R::Device::from_id(device_0);
        let device_1 = R::Device::from_id(device_1);
        let client_0 = R::client(&device_0);
        let client_1 = R::client(&device_1);

        let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = client_0.create(f32::as_bytes(&expected));

        let output = client_0.to_client(input, &client_1).handle;

        let actual = client_1.read_one(output);
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

        #[test]
        fn test_to_client() {
            cubecl_core::runtime_tests::to_client::test_to_client::<TestRuntime>();
        }
    };
}
