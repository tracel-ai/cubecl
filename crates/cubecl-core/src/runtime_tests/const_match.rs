use crate as cubecl;

use cubecl::prelude::*;

#[derive(CubeType, Clone, Hash, PartialEq, Eq, Debug)]
pub enum Operation {
    IndexAssign(u32, u32),
}

#[cube(launch)]
pub fn kernel_const_match_simple(output: &mut Array<f32>, #[comptime] op: Operation) {
    match op {
        Operation::IndexAssign(index, value) => {
            output[index.runtime()] = f32::cast_from(value.runtime());
        }
    };
}

pub fn test_kernel_const_match<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    let index = 1;
    let value = 5.0;

    kernel_const_match_simple::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, 1) },
        Operation::IndexAssign(index as u32, value as u32),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[index], value);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_const_match {
    () => {
        use super::*;

        #[test]
        fn test_const_match() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::const_match::test_kernel_const_match::<TestRuntime>(client);
        }
    };
}
