use crate as cubecl;
use crate::prelude::*;

#[cube(launch)]
fn constant_array_kernel<F: Float>(out: &mut Array<F>, #[comptime] data: Vec<u32>) {
    let array = Array::<F>::from_data(data);

    if UNIT_POS == 0 {
        out[0] = array[1];
    }
}

pub fn test_constant_array<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client
        .create(f32::as_bytes(&[0.0, 1.0]))
        .expect("Alloc failed");

    let vectorization = 1;

    constant_array_kernel::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
        vec![3, 5, 1],
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_constants {
    () => {
        use super::*;

        #[test]
        fn test_constant_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::constants::test_constant_array::<TestRuntime>(client);
        }
    };
}
