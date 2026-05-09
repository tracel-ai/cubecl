use crate::{self as cubecl};

use cubecl::prelude::*;

#[cube(launch)]
fn shuffle_kernel(orders: &mut [u32]) {
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
        unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
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
        fn test_shuffle() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_shuffle::<TestRuntime>(client);
        }
    };
}
