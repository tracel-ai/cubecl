use crate::prelude::*;
use crate::{self as cubecl, Feature};

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_sync_plane<F: Float>(out: &mut Array<F>) {
    let mut shared_memory = SharedMemory::<F>::new(1);

    if UNIT_POS == 0 {
        shared_memory[0] = F::from_int(1);
    }

    sync_plane();

    out[UNIT_POS] = shared_memory[0];
}

pub fn test_sync_plane<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    if !client.properties().feature_enabled(Feature::SyncPlane) {
        // We can't execute the test, skip.
        return;
    }

    let handle = client
        .empty(64 * core::mem::size_of::<f32>())
        .expect("Alloc failed");

    let vectorization = 1;

    kernel_test_sync_plane::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 2),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);
    let expected = &[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    assert_eq!(&actual[0..32], expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_sync_plane {
    () => {
        use super::*;

        #[test]
        fn test_sync_plane() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::sync_plane::test_sync_plane::<TestRuntime>(client);
        }
    };
}
