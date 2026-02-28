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

pub fn test_kernel_index_scalar<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(F::as_bytes(as_type![F: 0.0, 1.0, 123.0, 6.0]));
    let handle_slice = handle
        .clone()
        .offset_end(F::as_type_native_unchecked().size() as u64);
    let vectorization = 1;

    kernel_assign::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts::<F>(handle_slice, 3, vectorization) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
    assert_eq!(actual[1], F::new(0.0));
    assert_eq!(actual[2], F::new(123.0));
}

#[cube(launch)]
fn shuffle_kernel(orders: &mut Array<u32>) {
    if UNIT_POS < 2 {
        let mut order = Array::<u32>::new(2usize);
        order[0] = 0;
        order[1] = 1;

        let i = ABSOLUTE_POS & 1;
        let tmp = order[i];
        order[i] = 9;
        order[i ^ 1] = tmp;

        let base = ABSOLUTE_POS * 2;
        orders[base] = order[0];
        orders[base + 1] = order[1];
    }
}

// Regression test for invalid `CopyTransform`
pub fn test_kernel_shuffle<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(4 * size_of::<u32>());

    shuffle_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts::<u32>(handle.clone(), 4, 1) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, [9, 0, 1, 9]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_index {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_index_scalar::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_shuffle() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_shuffle::<TestRuntime>(client);
        }
    };
}
