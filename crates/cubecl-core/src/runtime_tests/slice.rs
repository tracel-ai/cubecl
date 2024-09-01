use crate as cubecl;
use cubecl::prelude::*;
use cubecl_macros_2::cube2;

#[cube2(launch)]
pub fn slice_select(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        let slice = &input[2..3];
        output[0] = slice[0];
    }
}

#[cube2(launch)]
pub fn slice_assign(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        let slice_1 = &mut output[2..3];
        slice_1[0] = input[0];
    }
}

#[cube2(launch)]
pub fn slice_len(input: &Array<f32>, output: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let slice = &input[2..4];
        let _tmp = slice[0]; // It must be used at least once, otherwise wgpu isn't happy.
        output[0] = slice.len();
    }
}

pub fn test_slice_select<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(core::mem::size_of::<f32>());

    unsafe {
        slice_select::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts(&input, 5, 1),
            ArrayArg::from_raw_parts(&output, 1, 1),
        )
    };

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 2.0);
}

pub fn test_slice_len<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(core::mem::size_of::<u32>());

    unsafe {
        slice_len::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts(&input, 5, 1),
            ArrayArg::from_raw_parts(&output, 1, 1),
        )
    };

    let actual = client.read(output.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[2]);
}

pub fn test_slice_assign<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[15.0]));
    let output = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));

    unsafe {
        slice_assign::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts(&input, 5, 1),
            ArrayArg::from_raw_parts(&output, 1, 1),
        )
    };

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, &[0.0, 1.0, 15.0, 3.0, 4.0]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_slice {
    () => {
        use super::*;

        #[test]
        fn test_slice_select() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::slice::test_slice_select::<TestRuntime>(client);
        }

        #[test]
        fn test_slice_assign() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::slice::test_slice_assign::<TestRuntime>(client);
        }

        #[test]
        fn test_slice_len() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::slice::test_slice_len::<TestRuntime>(client);
        }
    };
}
