use alloc::{vec, vec::Vec};

use crate as cubecl;
use crate::prelude::*;

#[cube(launch)]
fn constant_array_kernel<F: Float>(out: &mut [F], #[comptime] data: Vec<u32>) {
    let array = Array::<F>::from_data(data);

    if UNIT_POS == 0 {
        out[0] = array[1];
    }
}

pub fn test_constant_array<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[0.0, 1.0]));

    constant_array_kernel::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
        vec![3, 5, 1],
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_constants {
    () => {
        use super::*;

        #[ignore = "constant arrays are inconsistent across implementations"]
        #[$crate::runtime_tests::test_log::test]
        fn test_constant_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::constants::test_constant_array::<TestRuntime>(client);
        }
    };
}
