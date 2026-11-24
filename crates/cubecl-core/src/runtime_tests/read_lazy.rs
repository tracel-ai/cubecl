use crate::{self as cubecl};
use alloc::vec;
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::bytes::{AccessError, AllocationProperty, Bytes, Reader};
use cubecl_environment::future::block_on;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{ElemType, FloatKind};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::{CopyDescriptor, Handle, MemoryLayout, ReduceOperation};
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

    let lazy = client.read_lazy(CopyDescriptor::new(handle, shape, strides, elem_size));

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
        assert!(lazy.may_alias_mutable_storage());

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
            "metadata queries and no-copy reads must not materialize the device buffer"
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

/// A pending view must retain the source independently of its parent lazy read.
pub fn test_read_lazy_holds_its_source<R: Runtime>(client: Client) {
    let data = [1i32, 2, 3, 4];
    let elem_size = size_of::<i32>();
    let shape = shape![data.len()];
    let MemoryLayout {
        memory: handle,
        strides,
    } = client.create_tensor_from_slice(i32::as_bytes(&data), shape.clone(), elem_size);
    assert!(handle.can_mut());

    let lazy = client.read_lazy(CopyDescriptor::new(
        handle.clone(),
        shape,
        strides,
        elem_size,
    ));
    assert!(
        !handle.can_mut(),
        "a pending lazy read must hold its source"
    );

    let view = lazy
        .view(0, elem_size)
        .expect("a contiguous device buffer supports views");

    assert_eq!(i32::from_bytes(&lazy), &data);
    drop(lazy);
    assert!(
        !handle.can_mut(),
        "a pending view must hold its source after the parent is dropped"
    );

    assert_eq!(i32::from_bytes(&view), &data[..1]);
    drop(view);
    assert!(
        handle.can_mut(),
        "dropping the reads must release the source"
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
        let lazy = client.read_lazy(CopyDescriptor::new(
            memory.clone(),
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

const BEFORE: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const AFTER: [f32; 4] = [2.0, 3.0, 4.0, 5.0];

#[cube(launch)]
fn add_one_in_place(buffer: &mut [f32]) {
    if ABSOLUTE_POS < buffer.len() {
        buffer[ABSOLUTE_POS] += 1.0;
    }
}

#[cube(launch)]
fn add_one_into(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

fn lazy(client: &Client, buffer: &Handle) -> Bytes {
    let source = CopyDescriptor::new(
        buffer.clone(),
        [BEFORE.len()].into(),
        [1].into(),
        size_of::<f32>(),
    );
    block_on(client.read_lazy_async(source)).expect("the buffer should be readable")
}

/// A kernel handed the buffer itself writes it in place, whatever `can_mut` says.
fn add_one_to(client: &Client, buffer: &Handle) {
    add_one_in_place::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(BEFORE.len() as u32),
        unsafe { BufferArg::from_raw_parts(buffer.clone(), BEFORE.len()) },
    );
}

/// A kernel that writes a buffer a lazy read holds leaves that read the buffer as it was.
pub fn test_a_lazy_read_is_its_buffer_before_a_kernel_writes_it<R: Runtime>(client: Client) {
    let buffer = client.create_from_slice(f32::as_bytes(&BEFORE));
    let read = lazy(&client, &buffer);

    add_one_to(&client, &buffer);

    assert_eq!(f32::from_bytes(&read), BEFORE);
    assert_eq!(f32::from_bytes(&client.read_one(buffer).unwrap()), AFTER);
}

/// A host write to a buffer a lazy read holds leaves that read the buffer as it was.
pub fn test_a_lazy_read_is_its_buffer_before_a_host_write<R: Runtime>(client: Client) {
    let buffer = client.create_from_slice(f32::as_bytes(&BEFORE));
    let read = lazy(&client, &buffer);

    client.write(
        &buffer,
        Bytes::from_bytes_vec(f32::as_bytes(&AFTER).to_vec()),
    );

    assert_eq!(f32::from_bytes(&read), BEFORE);
}

/// A view of a lazy read takes its bytes from what the write started for the read it views.
pub fn test_a_view_of_a_lazy_read_is_its_buffer_before_a_write<R: Runtime>(client: Client) {
    let buffer = client.create_from_slice(f32::as_bytes(&BEFORE));
    let read = lazy(&client, &buffer);
    let view = read
        .view(size_of::<f32>(), 3 * size_of::<f32>())
        .expect("a contiguous device buffer supports views");

    add_one_to(&client, &buffer);

    assert_eq!(f32::from_bytes(&view), &BEFORE[1..3]);
}

/// A view that outlives the lazy read it came from copies its own bytes before a write.
pub fn test_a_view_outliving_its_lazy_read_is_its_buffer_before_a_write<R: Runtime>(
    client: Client,
) {
    let buffer = client.create_from_slice(f32::as_bytes(&BEFORE));
    let read = lazy(&client, &buffer);
    let view = read
        .view(size_of::<f32>(), 3 * size_of::<f32>())
        .expect("a contiguous device buffer supports views");
    drop(read);

    add_one_to(&client, &buffer);

    assert_eq!(f32::from_bytes(&view), &BEFORE[1..3]);
}

/// A write on another stream than the read's waits for the read it overwrites.
pub fn test_a_lazy_read_is_its_buffer_before_a_write_on_another_stream<R: Runtime>(client: Client) {
    let (reader, writer) = (StreamId { value: 3_300_001 }, StreamId { value: 3_300_002 });
    let buffer = reader.executes(|| client.create_from_slice(f32::as_bytes(&BEFORE)));
    let read = reader.executes(|| lazy(&client, &buffer));

    writer.executes(|| add_one_to(&client, &buffer));

    assert_eq!(f32::from_bytes(&read), BEFORE);
}

/// A replay that writes a buffer a lazy read holds leaves that read the buffer as it was.
///
/// Not part of `testgen_read_lazy_before_write`: a capture holds its stream for its whole window,
/// so a runtime calls this from its graph tests, one capture at a time.
pub fn a_lazy_read_is_its_buffer_before_a_replay<R: Runtime>() {
    let client = R::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&BEFORE));
    let output = client.empty(size_of_val(&BEFORE));
    let (graph, ()) = client
        .capture(|| {
            add_one_into::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(BEFORE.len() as u32),
                unsafe { BufferArg::from_raw_parts(input.clone(), BEFORE.len()) },
                unsafe { BufferArg::from_raw_parts(output.clone(), BEFORE.len()) },
            )
        })
        .expect("the launch should capture");
    unsafe { graph.replay() }.expect("the replay should enqueue");
    let first_replay = BEFORE.map(|value| value + 1.0);
    let read = lazy(&client, &output);

    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&AFTER).to_vec()),
    );
    unsafe { graph.replay() }.expect("the replay should enqueue");

    assert_eq!(f32::from_bytes(&read), first_replay);
}

/// A replayed collective that writes a buffer a lazy read holds leaves that read the buffer as
/// it was.
///
/// Called from a runtime's graph tests, as [`a_lazy_read_is_its_buffer_before_a_replay`].
pub fn a_lazy_read_is_its_buffer_before_a_replayed_collective<R: Runtime>() {
    let _collectives = super::collectives::queue_alone();
    let mut client = R::client(&Default::default());
    if !client.has_device_transport() {
        std::println!("skipped: no device transport");
        return;
    }
    let device = client.service_id().device;
    let input = client.create_from_slice(f32::as_bytes(&BEFORE));
    let output = client.empty(size_of_val(&BEFORE));
    let (graph, ()) = client
        .clone()
        .capture(|| {
            client.all_reduce(
                input.clone(),
                output.clone(),
                ElemType::Float(FloatKind::F32),
                vec![device],
                ReduceOperation::Sum,
            );
            client.sync_collective();
        })
        .expect("the collective should capture");
    unsafe { graph.replay() }.expect("the replay should enqueue");
    let read = lazy(&client, &output);

    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&AFTER).to_vec()),
    );
    unsafe { graph.replay() }.expect("the replay should enqueue");

    assert_eq!(f32::from_bytes(&read), BEFORE);
}

/// A lazy read on a stream recording a graph is refused: its copy would be recorded rather
/// than run.
///
/// Called from a runtime's graph tests, as [`a_lazy_read_is_its_buffer_before_a_replay`].
pub fn a_lazy_read_inside_a_capture_window_is_refused<R: Runtime>() {
    let client = R::client(&Default::default());
    let buffer = client.create_from_slice(f32::as_bytes(&BEFORE));
    client.graph_prepare().expect("the capture should prepare");
    add_one_to(&client, &buffer);
    client.graph_end_run().expect("the planning run should end");
    client.start_capture().expect("the window should open");
    add_one_to(&client, &buffer);

    let source = CopyDescriptor::new(
        buffer.clone(),
        [BEFORE.len()].into(),
        [1].into(),
        size_of::<f32>(),
    );
    let refused = block_on(client.read_lazy_async(source));

    drop(client.stop_capture());
    assert!(refused.is_err());
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
        fn test_read_lazy_holds_its_source() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_read_lazy_holds_its_source::<TestRuntime>(
                client,
            );
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_read_lazy_before_write {
    () => {
        mod read_lazy_before_write {
            use super::*;
            use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_a_lazy_read_is_its_buffer_before_a_kernel_writes_it() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_a_lazy_read_is_its_buffer_before_a_kernel_writes_it::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_lazy_read_is_its_buffer_before_a_host_write() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_a_lazy_read_is_its_buffer_before_a_host_write::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_view_of_a_lazy_read_is_its_buffer_before_a_write() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_a_view_of_a_lazy_read_is_its_buffer_before_a_write::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_view_outliving_its_lazy_read_is_its_buffer_before_a_write() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_a_view_outliving_its_lazy_read_is_its_buffer_before_a_write::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_lazy_read_is_its_buffer_before_a_write_on_another_stream() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::read_lazy::test_a_lazy_read_is_its_buffer_before_a_write_on_another_stream::<TestRuntime>(client);
        }
        }
    };
}
