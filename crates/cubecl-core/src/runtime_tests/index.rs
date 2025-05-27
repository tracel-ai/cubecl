use crate::{self as cubecl, as_type};

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_assign<F: Float>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let item = F::new(5.0);
        // Assign normally.
        output[0] = item;

        // out of bounds write should not show up in the array.
        output[4] = F::new(10.0);

        // out of bounds read should be read as 0.
        output[1] = output[4];
    }
}

pub fn test_kernel_index_scalar<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(F::as_bytes(as_type![F: 0.0, 1.0, 123.0, 6.0]));
    let handle_slice = handle
        .clone()
        .offset_end(F::as_elem_native_unchecked().size() as u64);
    let vectorization = 1;

    kernel_assign::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle_slice, 3, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
    assert_eq!(actual[1], F::new(0.0));
    assert_eq!(actual[2], F::new(123.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_index {
    () => {
        use super::*;

        #[test]
        fn test_assign_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_index_scalar::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
