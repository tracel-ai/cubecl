use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::bytes::AllocationProperty;
use cubecl_runtime::server::MemoryLayout;
use cubecl_zspace::shape;

/// A device resource read lazily must defer the device-to-host copy until first access, then
/// return the same data as an eager read.
pub fn test_read_lazy<R: Runtime>(client: ComputeClient<R>) {
    let data = (0i32..1024).collect::<Vec<i32>>();
    let bytes_expected = i32::as_bytes(&data);
    let elem_size = size_of::<i32>();
    let shape = shape![data.len()];

    let MemoryLayout {
        memory: handle,
        strides,
    } = client.create_tensor_from_slice(bytes_expected, shape.clone(), elem_size);

    let descriptor = handle.copy_descriptor(shape, strides, elem_size);
    let lazy = client.read_lazy(descriptor);

    // Before any access the data is still on the device. On wasm `read_lazy` is eager, so the
    // property already reflects the materialized host allocation.
    #[cfg(not(target_family = "wasm"))]
    assert!(
        matches!(lazy.property(), AllocationProperty::Device),
        "lazy bytes must report `Device` before the first access, got {:?}",
        lazy.property()
    );

    // First access materializes through the regular read path and caches the result.
    let bytes_lazy: &[u8] = &lazy;
    assert_eq!(
        bytes_expected, bytes_lazy,
        "lazily read bytes must match the source data"
    );

    // After materialization the property reflects the host allocation, not `Device`.
    assert!(
        !matches!(lazy.property(), AllocationProperty::Device),
        "materialized bytes must no longer report `Device`"
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_read_lazy {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_read_lazy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_read_lazy::<TestRuntime>(client);
        }
    };
}
