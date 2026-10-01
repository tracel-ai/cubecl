//! An allocation larger than the device's memory, and what the client reports for it.
//!
//! Running out of memory is recoverable: the allocation fails, the buffer reports why when it
//! is read, and the device keeps serving the next allocation.

use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;

/// A terabyte: more than any device holds.
const TOO_LARGE: usize = 1 << 40;

#[test]
fn an_allocation_too_large_for_the_device_fails_at_the_read() {
    let client = HipRuntime::client(&Default::default());

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
