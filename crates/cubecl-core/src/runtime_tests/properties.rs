use crate::{self as cubecl, as_type};

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_properties(output: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let properties = device_properties();
        output[0] = properties.hardware.plane_size_min;
        output[1] = properties.hardware.plane_size_max;
    }
}

pub fn test_kernel_properties<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(u32::as_bytes(as_type![u32: 0, 0]));

    kernel_properties::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts::<u32>(&handle, 2, 1) },
    )
    .unwrap();

    let actual = client.read_one(handle);
    let actual = u32::from_bytes(&actual);
    let plane_size_min = client.properties().hardware.plane_size_min;
    let plane_size_max = client.properties().hardware.plane_size_max;

    assert_eq!(actual[0], plane_size_min);
    assert_eq!(actual[1], plane_size_max);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_properties {
    () => {
        use super::*;

        #[test]
        fn test_device_properties() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::properties::test_kernel_properties::<TestRuntime>(client);
        }
    };
}
