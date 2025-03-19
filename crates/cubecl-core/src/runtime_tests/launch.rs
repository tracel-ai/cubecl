use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[derive(CubeLaunch, CubeType)]
pub struct ComptimeTag {
    array: Array<f32>,
    #[cube(comptime)]
    tag: String,
}

#[cube(launch)]
pub fn kernel_with_comptime_tag(output: &mut ComptimeTag) {
    if UNIT_POS == 0 {
        if comptime![&output.tag == "zero"] {
            output.array[0] = f32::new(0.0);
        } else {
            output.array[0] = f32::new(1.0);
        }
    }
}

#[cube(launch)]
pub fn kernel_with_generics<F: Float>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        output[0] = F::new(5.0);
    }
}

#[cube(launch)]
pub fn kernel_without_generics(output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        output[0] = 5.0;
    }
}

pub fn test_kernel_with_comptime_tag<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[5.0]));
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 1, 1) };

    kernel_with_comptime_tag::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ComptimeTagLaunch::new(array_arg, &"zero".to_string()),
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(0.0));

    let handle = client.create(f32::as_bytes(&[5.0]));
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 1, 1) };

    kernel_with_comptime_tag::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ComptimeTagLaunch::new(array_arg, &"not_zero".to_string()),
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(1.0));
}

pub fn test_kernel_with_generics<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]);

    kernel_with_generics::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_without_generics<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    kernel_without_generics::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_launch {
    () => {
        use super::*;

        #[test]
        fn test_launch_with_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_generics::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_launch_without_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_without_generics::<TestRuntime>(client);
        }

        #[test]
        fn test_launch_with_comptime_tag() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_comptime_tag::<TestRuntime>(
                client,
            );
        }
    };
}
