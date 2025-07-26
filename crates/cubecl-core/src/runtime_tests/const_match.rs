use core::hash;
use std::fmt::Debug;

use crate::{self as cubecl, as_bytes};

use cubecl::prelude::*;

#[derive(CubeType, Clone, Hash, PartialEq, Eq, Debug)]
pub enum Operation<U: Int + hash::Hash + Eq + Debug> {
    IndexAssign(u32, U),
}

#[cube(launch)]
pub fn kernel_const_match_simple<F: Float, U: Int + hash::Hash + Eq + Debug>(
    output: &mut Array<F>,
    #[comptime] op: Operation<U>,
) {
    match op {
        Operation::IndexAssign(index, value) => {
            output[index.runtime()] = F::cast_from(value.runtime());
        }
    };
}

pub fn test_kernel_const_match<
    R: Runtime,
    F: Float + CubeElement,
    U: Int + hash::Hash + Eq + Debug,
>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    let index = 1;
    let value = 5.0;

    kernel_const_match_simple::launch::<F, U, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
        Operation::IndexAssign(index as u32, U::new(value as i64)),
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[index], F::new(value));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_const_match {
    () => {
        use super::*;

        #[test]
        fn test_const_match() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::const_match::test_kernel_const_match::<
                TestRuntime,
                FloatType,
                UintType,
            >(client);
        }
    };
}
