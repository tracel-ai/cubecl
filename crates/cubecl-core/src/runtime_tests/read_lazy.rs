use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::bytes::{AccessError, AllocationProperty, Bytes, Reader};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::MemoryLayout;
use cubecl_zspace::shape;

/// A device resource read lazily must defer the device-to-host copy until first access, then
/// return the same data as an eager read.
pub fn test_read_lazy<R: Runtime>(client: Client) {
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
    {
        assert!(
            matches!(lazy.property(), AllocationProperty::Device),
            "lazy bytes must report `Device` before the first access, got {:?}",
            lazy.property()
        );

        // `len()` and `capacity()` must be no-copy: they must not pull the data off the device.
        assert_eq!(lazy.len(), bytes_expected.len());
        assert_eq!(lazy.capacity(), bytes_expected.len());

        // A no-copy read must refuse rather than materialize.
        assert!(
            matches!(
                lazy.read(Reader::new().no_copy()),
                Err(AccessError::WouldCopy)
            ),
            "a no-copy read of an unmaterialized device buffer must refuse"
        );

        // None of the above triggered a device-to-host copy.
        assert!(
            matches!(lazy.property(), AllocationProperty::Device),
            "len()/capacity()/no-copy read must not materialize the device buffer"
        );

        // A device view is itself lazy and reads only its byte sub-range when materialized.
        let (start, end) = (400, 800);
        let view = lazy
            .view(start, end)
            .expect("a contiguous device buffer supports views");
        assert!(
            matches!(view.property(), AllocationProperty::Device),
            "a device view must itself be lazy"
        );
        let view_bytes: &[u8] = &view;
        assert_eq!(
            view_bytes,
            &bytes_expected[start..end],
            "a device view must read its sub-range"
        );
        // Materializing the view must not have materialized the parent.
        assert!(
            matches!(lazy.property(), AllocationProperty::Device),
            "viewing must not materialize the parent device buffer"
        );
    }

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

/// Empty tensors can be read eagerly or lazily, including layouts with zero strides.
pub fn test_read_empty_tensor<R: Runtime>(client: Client) {
    for shape in [
        shape![0],
        shape![0, 2],
        shape![2, 0],
        shape![0, 0],
        shape![2, 0, 3],
    ] {
        let elem_size = size_of::<f32>();
        let MemoryLayout { memory, strides } = client.empty_tensor(shape.clone(), elem_size);
        let lazy = client.read_lazy(memory.clone().copy_descriptor(
            shape.clone(),
            strides.clone(),
            elem_size,
        ));
        // len() alone does not materialize a lazy device read.
        assert!(lazy.read(Reader::new()).unwrap().is_empty());

        let eager =
            client.read_one_unchecked_tensor(memory.copy_descriptor(shape, strides, elem_size));
        assert!(eager.read(Reader::new()).unwrap().is_empty());
    }
}

/// Uploading an empty tensor must not reject its zero strides or attempt a device copy.
pub fn test_create_empty_tensor<R: Runtime>(client: Client) {
    for shape in [
        shape![0],
        shape![0, 2],
        shape![2, 0],
        shape![0, 0],
        shape![2, 0, 3],
    ] {
        let allocation =
            client.create_tensor(Bytes::from_bytes_vec(Vec::new()), shape, size_of::<f32>());
        // Check the upload independently of readback so a read failure cannot mask it.
        client.check([&allocation.memory]).unwrap();
    }
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

        #[$crate::runtime_tests::test_log::test]
        fn test_read_empty_tensor() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_read_empty_tensor::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_create_empty_tensor() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_create_empty_tensor::<TestRuntime>(client);
        }
    };
}
