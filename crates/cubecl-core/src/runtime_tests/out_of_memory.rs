//! An allocation larger than the device's memory or than its largest single allocation, and
//! what the client reports for it.
//!
//! Both are recoverable: the allocation fails, the buffer reports why when it is read, and the
//! device keeps serving the next allocation.

use alloc::string::ToString;
use cubecl_runtime::{client::Client, runtime::Runtime, server::Handle};

/// A terabyte: more than any device holds.
const TOO_LARGE: usize = 1 << 40;

/// The read of a buffer the device could not allocate fails, naming the allocation, and the
/// device stays usable.
pub fn an_allocation_too_large_for_the_device_fails_at_the_read<R: Runtime>(client: Client) {
    let error = client
        .read_one(client.empty(TOO_LARGE))
        .expect_err("a buffer the device could not allocate must not read clean");
    assert!(
        !error.is_device_poisoned(),
        "running out of memory leaves the device usable, got: {error}"
    );
    assert!(
        error.to_string().contains(&TOO_LARGE.to_string()),
        "the error names the allocation that failed, got: {error}"
    );

    let bytes = client
        .read_one(client.create_from_slice(&[7u8; 4]))
        .expect("the device keeps serving allocations that fit");
    assert_eq!(&bytes[..], &[7u8; 4]);
}

/// The first size past the largest single allocation the device makes.
fn past_the_largest_allocation(client: &Client) -> usize {
    let largest = client.properties().memory.max_allocation_size();
    usize::try_from(largest.saturating_add(1)).unwrap_or(usize::MAX)
}

/// Reads `refused`, which must fail as an allocation the device can never make, then checks
/// the device still serves an allocation that fits.
fn assert_refused_and_device_usable(client: &Client, refused: Handle, size: usize) {
    let error = client
        .read_one(refused)
        .expect_err("a buffer past the largest allocation must not read clean");
    assert!(
        !error.is_device_poisoned(),
        "a refused allocation leaves the device usable, got: {error}"
    );
    // The size named is the one the client asked the pool for, padded to the device's
    // alignment, so it is not `size` itself.
    assert!(
        error.to_string().contains("can't allocate buffer"),
        "the allocation of {size} bytes is refused as too big, got: {error}"
    );

    let bytes = client
        .read_one(client.create_from_slice(&[7u8; 4]))
        .expect("the device keeps serving allocations that fit");
    assert_eq!(&bytes[..], &[7u8; 4]);
}

/// A dedicated allocation past the largest the device makes is refused before the device is
/// asked, and the device stays usable. On wgpu, asking would fail validation and poison it.
pub fn a_dedicated_allocation_past_the_largest_is_refused<R: Runtime>(client: Client) {
    let size = past_the_largest_allocation(&client);
    let refused = client.memory_dedicated_allocation((), |_| client.empty(size));
    assert_refused_and_device_usable(&client, refused, size);
}

/// A persistent allocation past the largest the device makes is refused before the device is
/// asked, and the device stays usable.
pub fn a_persistent_allocation_past_the_largest_is_refused<R: Runtime>(client: Client) {
    let size = past_the_largest_allocation(&client);
    let refused = client.memory_persistent_allocation((), |_| client.empty(size));
    assert_refused_and_device_usable(&client, refused, size);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_out_of_memory {
    () => {
        mod out_of_memory {
            use super::*;

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_dedicated_allocation_past_the_largest_is_refused() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::out_of_memory::a_dedicated_allocation_past_the_largest_is_refused::<
                    TestRuntime,
                >(client);
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_persistent_allocation_past_the_largest_is_refused() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::out_of_memory::a_persistent_allocation_past_the_largest_is_refused::<
                    TestRuntime,
                >(client);
            }

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn an_allocation_too_large_for_the_device_fails_at_the_read() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::out_of_memory::an_allocation_too_large_for_the_device_fails_at_the_read::<
                    TestRuntime,
                >(client);
            }
        }
    };
}
