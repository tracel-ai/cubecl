//! Regression test: reading a zero-size buffer stages through pinned host
//! memory, and a zero-size pinned allocation carries a NULL pointer
//! (`hipHostMalloc(0)` returns success without allocating). Building the
//! staging slice from that pointer used to trip the `from_raw_parts_mut`
//! non-null precondition and SIGABRT the process.

use cubecl_core::prelude::*;
use cubecl_hip::HipRuntime;

#[test]
fn read_empty_buffer() {
    let client = HipRuntime::client(&Default::default());
    let handle = client.empty(0);
    let bytes = client.read_one(handle).unwrap();
    assert!(bytes.is_empty());
}

/// Writing an empty rank-2 tensor (`[0, 1]` — zero rows, non-zero row width)
/// used to reach `hipMemcpy2DAsync` with the row count clamped to 1, copying a
/// full row out of the zero-size staging buffer and segfaulting. The follow-up
/// non-empty round-trip proves the pools survive the empty write.
#[test]
fn write_empty_tensor_then_roundtrip() {
    let client = HipRuntime::client(&Default::default());

    let empty = client.create_tensor_from_slice(&[], [0, 1].into(), 4);

    let values = [1.0f32, 2.0, 3.0];
    let full = client.create_tensor_from_slice(f32::as_bytes(&values), [3, 1].into(), 4);

    let bytes = client.read_one(full.memory).unwrap();
    assert_eq!(f32::from_bytes(&bytes), &values);

    // A shaped read of the empty tensor yields zero bytes.
    let descriptor = cubecl_core::server::CopyDescriptor::new(
        empty.memory.binding(),
        [0, 1].into(),
        empty.strides,
        4,
    );
    let bytes = client.read_tensor(vec![descriptor]).remove(0);
    assert!(bytes.is_empty());
}
