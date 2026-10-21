//! What an external operation is owed by the write scope and the stream it runs
//! on: earlier queued work is visible to it, its writes are visible to later
//! work, a failure or panic lands on those writes, and the server keeps the
//! bindings and every owner it registered until the work completes.
//!
//! Only a [`NativeRuntime`] can ask for one, so a backend without the native
//! types refuses it at compile time instead of recording a refusal on the
//! writes.

use crate::{self as cubecl};
use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use cubecl::prelude::*;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::runtime::{NativeRuntime, Runtime};
use cubecl_runtime::server::{ExternalOpCallback, Handle, ServerError};
use std::sync::mpsc::channel;
use std::time::Duration;

/// Copy `len` bytes from `src` to `dst` on the backend's native external
/// stream. The tests supply the concrete copy; CUDA uses a device-to-device
/// copy, the host runtime a `memcpy`.
pub type ExternalCopy<R> = fn(
    <R as NativeRuntime>::NativeStream,
    <R as NativeRuntime>::NativeBuffer,
    <R as NativeRuntime>::NativeBuffer,
    u64,
) -> Result<(), ServerError>;

/// Fill `len` bytes at `dst` with the 32-bit pattern `value`, on the backend's
/// native external stream. Used by the write-only cross-stream test.
pub type ExternalFill<R> = fn(
    <R as NativeRuntime>::NativeStream,
    <R as NativeRuntime>::NativeBuffer,
    u64,
    u32,
) -> Result<(), ServerError>;

/// The byte size of one binding in the backend's own terms, for the tests to
/// check the server filled it in.
pub type ExternalSize<R> = fn(&<R as NativeRuntime>::NativeBuffer) -> u64;

/// A drop observer shared between an operation's `retain` and the test.
struct DropObserver {
    dropped: Arc<AtomicBool>,
    notify: std::sync::mpsc::Sender<()>,
}

/// The test's end of a [`DropObserver`], which can be checked at a point and
/// waited on with a bound once the owner should be gone.
struct DropNotification {
    dropped: Arc<AtomicBool>,
    notify: std::sync::mpsc::Receiver<()>,
}

fn drop_observer() -> (DropObserver, DropNotification) {
    let dropped = Arc::new(AtomicBool::new(false));
    let (notify, wait) = channel();
    (
        DropObserver {
            dropped: dropped.clone(),
            notify,
        },
        DropNotification {
            dropped,
            notify: wait,
        },
    )
}

impl Drop for DropObserver {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
        // The receiver may already be gone; the atomic still tells the test.
        let _ = self.notify.send(());
    }
}

impl DropNotification {
    fn is_dropped(&self) -> bool {
        self.dropped.load(Ordering::SeqCst)
    }

    /// The GC thread releases deferred owners after the stream completes, so
    /// the test waits for the drop rather than assuming a device barrier ran
    /// its destructor.
    fn wait(self, message: &str) {
        self.notify
            .recv_timeout(Duration::from_secs(30))
            .expect(message);
    }
}

fn launch_fill(client: &Client, out: &Handle, value: u32) {
    crate::runtime_tests::stream_errors::fill::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
        value,
    );
}

fn launch_copy(client: &Client, input: &Handle, out: &Handle) {
    crate::runtime_tests::stream_errors::copy::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(input.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
    );
}

fn failure(reason: &str) -> ServerError {
    ServerError::Generic {
        reason: alloc::string::String::from(reason),
        backtrace: BackTrace::capture(),
    }
}

fn assert_failed(client: &Client, handle: Handle, reason: &str) {
    let err = client
        .read_one(handle)
        .expect_err("the work that was going to write this buffer failed");
    let message = format!("{err}");
    assert!(
        message.contains(reason),
        "the read must name the failure that stopped the write, got: {message}"
    );
}

/// Earlier queued work is visible to the operation: a launch writes `src`, the
/// operation copies it to `dst`, and reading `dst` must see the launch's value.
pub fn test_external_op_sees_earlier_work<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    let len = core::mem::size_of::<u32>() as u64;
    let src = client.empty(len as usize);
    let dst = client.empty(len as usize);
    launch_fill(&client, &src, 42);

    client.external_op::<R>(
        core::slice::from_ref(&src),
        core::slice::from_ref(&dst),
        move |context| {
            assert_eq!(size(&context.reads()[0]), len, "the read's size");
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            copy(
                context.stream(),
                context.reads()[0],
                context.writes()[0],
                len,
            )?;
            Ok(())
        },
    );

    let read = client
        .read_one(dst)
        .expect("the operation copied the launch's value");
    assert_eq!(u32::from_bytes(&read), &[42]);
}

/// Later work is ordered behind the operation: the operation writes `mid`, a
/// launch then copies `mid` to `out`.
pub fn test_external_op_write_is_seen_by_a_later_launch<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    let len = core::mem::size_of::<u32>() as u64;
    let src = client.empty(len as usize);
    let mid = client.empty(len as usize);
    let out = client.empty(len as usize);
    launch_fill(&client, &src, 7);

    client.external_op::<R>(
        core::slice::from_ref(&src),
        core::slice::from_ref(&mid),
        move |context| {
            assert_eq!(size(&context.reads()[0]), len, "the read's size");
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            copy(
                context.stream(),
                context.reads()[0],
                context.writes()[0],
                len,
            )?;
            Ok(())
        },
    );
    launch_copy(&client, &mid, &out);

    let read = client
        .read_one(out)
        .expect("the later launch ran after the operation's write");
    assert_eq!(u32::from_bytes(&read), &[7]);
}

/// A write-only operation on stream `B` must wait for earlier work queued on
/// the stream `A` that owns the target. Only the declared write can establish
/// that dependency.
pub fn test_external_op_write_resolves_cross_stream<R: NativeRuntime>(
    client: Client,
    fill: ExternalFill<R>,
    size: ExternalSize<R>,
) {
    let owner = StreamId { value: 5_000_001 };
    let writer = StreamId { value: 5_000_002 };
    let len = core::mem::size_of::<f32>() as u64;

    // Allocate on `owner`, so the handle resolves there and the write-only
    // operation has only this list entry to synchronize against.
    let target = owner.executes(|| client.create_from_slice(f32::as_bytes(&[0.0])));

    let input = owner.executes(|| client.create_from_slice(u32::as_bytes(&[1u32])));
    owner.executes(|| {
        // A long-running launch writing 1.0 into `target`, queued on the
        // allocation stream before the operation is submitted on `writer`.
        crate::runtime_tests::stream::big_task::launch::<f32>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(input.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(target.clone(), 1) },
            50_000_000,
        );
    });

    writer.executes(|| {
        client.external_op::<R>(&[], core::slice::from_ref(&target), move |context| {
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            // 2.0 as f32 bits, through the write pointer only.
            fill(context.stream(), context.writes()[0], len, 0x4000_0000)?;
            Ok(())
        });

        let read = client
            .read_one(target.clone())
            .expect("the write-only operation resolved the target's stream");
        assert_eq!(
            f32::from_bytes(&read),
            &[2.0],
            "the operation must run after the queued work on the target's stream"
        );
    });
}

/// An error from the operation taints its writes and is reported by a read of
/// one of them.
pub fn test_external_op_error_taints_writes<R: NativeRuntime>(client: Client) {
    let dst = client.empty(core::mem::size_of::<u32>());

    client.external_op::<R>(&[], core::slice::from_ref(&dst), |_context| {
        Err(failure("external operation root"))
    });

    assert_failed(&client, dst, "external operation root");
}

/// A tainted input skips the operation and passes its failure to the outputs.
pub fn test_external_op_tainted_input_skips_it<R: NativeRuntime>(client: Client) {
    let tainted = client.empty(core::mem::size_of::<u32>());
    client.external_op::<R>(&[], core::slice::from_ref(&tainted), |_context| {
        Err(failure("external operation root"))
    });

    let out = client.empty(core::mem::size_of::<u32>());
    let called = Arc::new(AtomicBool::new(false));
    let observed = called.clone();
    client.external_op::<R>(
        core::slice::from_ref(&tainted),
        core::slice::from_ref(&out),
        move |_context| {
            observed.store(true, Ordering::SeqCst);
            Ok(())
        },
    );

    assert_failed(&client, out, "external operation root");
    assert!(
        !called.load(Ordering::SeqCst),
        "an operation with a tainted input must not run"
    );
}

/// A panic inside the operation settles as an error on its writes, and the
/// device keeps accepting work afterwards.
pub fn test_external_op_panic_is_settled<R: NativeRuntime>(client: Client) {
    let dst = client.empty(core::mem::size_of::<u32>());
    let (owner, dropped) = drop_observer();
    client.external_op::<R>(
        &[],
        core::slice::from_ref(&dst),
        move |context| -> Result<(), ServerError> {
            context.retain(owner);
            panic!("external operation boom")
        },
    );

    assert_failed(&client, dst, "external operation boom");
    future::block_on(client.sync()).expect("the barrier settles the panicked operation");
    dropped.wait("a panic does not leak the owners the operation registered");

    let healthy = client.create_from_slice(u32::as_bytes(&[5u32]));
    let read = client
        .read_one(healthy)
        .expect("the device still works after a callback panicked");
    assert_eq!(u32::from_bytes(&read), &[5]);
}

/// A handle may be declared both read and written, for an in-place operation.
pub fn test_external_op_in_place<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    let len = core::mem::size_of::<u32>() as u64;
    let buffer = client.empty(len as usize);
    launch_fill(&client, &buffer, 3);

    client.external_op::<R>(
        core::slice::from_ref(&buffer),
        core::slice::from_ref(&buffer),
        move |context| {
            assert_eq!(size(&context.reads()[0]), len, "the read's size");
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            copy(
                context.stream(),
                context.reads()[0],
                context.writes()[0],
                len,
            )?;
            Ok(())
        },
    );

    let read = client
        .read_one(buffer)
        .expect("an in-place operation is accepted");
    assert_eq!(u32::from_bytes(&read), &[3]);
}

/// Register an owner before enqueueing work, block until the test has observed
/// it, then return `Ok` or `Err`. The server must keep the owner alive across
/// both the enqueue and the outcome.
fn retained_until_completion<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
    return_error: bool,
) {
    let len = core::mem::size_of::<u32>() as u64;
    let src = client.create_from_slice(u32::as_bytes(&[9u32]));
    let dst = client.empty(len as usize);
    let dst_worker = dst.clone();
    let (owner, dropped) = drop_observer();
    let (started_tx, started_rx) = channel();
    let (release_tx, release_rx) = channel();
    let client_worker = client.clone();

    let worker = std::thread::spawn(move || {
        client_worker.external_op::<R>(&[src], &[dst_worker], move |context| {
            // Retained before the enqueue below, as a vendor call must.
            context.retain(owner);
            assert_eq!(size(&context.reads()[0]), len, "the read's size");
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            copy(
                context.stream(),
                context.reads()[0],
                context.writes()[0],
                len,
            )?;
            started_tx.send(()).expect("the test is listening");
            release_rx.recv().expect("the test releases the callback");
            if return_error {
                Err(failure("external partial enqueue"))
            } else {
                Ok(())
            }
        });
    });

    started_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the callback started");
    assert!(
        !dropped.is_dropped(),
        "the owner is alive while the operation and its queued work are in flight"
    );
    release_tx.send(()).expect("the callback is waiting");

    // The worker thread also waits for a blocking transport to accept the
    // submission, so join before the barrier that waits for completion.
    worker.join().expect("the submitting thread");

    if return_error {
        future::block_on(client.sync()).expect("plain sync waits for completion");
    } else {
        future::block_on(client.sync_buffers([&dst]))
            .expect("the completion barrier and the write check succeed");
    }
    dropped.wait("the server releases the owner once the work has completed");
}

/// The owner stays retained after the callback returns while its queued work is
/// still blocked on the device. A callback that returns is not completion: only
/// the barrier after the queued work is.
pub fn test_external_op_retained_after_callback_returns<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    let len = core::mem::size_of::<u32>() as u64;
    let src = client.create_from_slice(u32::as_bytes(&[11u32]));
    let dst = client.empty(len as usize);
    let dst_worker = dst.clone();

    // A long launch ahead of the operation on the same stream keeps the
    // operation's own work queued after its callback has already returned.
    let blocker_in = client.create_from_slice(u32::as_bytes(&[1u32]));
    let blocker_out = client.empty(core::mem::size_of::<f32>());
    crate::runtime_tests::stream::big_task::launch::<f32>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(blocker_in, 1) },
        unsafe { BufferArg::from_raw_parts(blocker_out.clone(), 1) },
        50_000_000,
    );

    let (owner, dropped) = drop_observer();
    let client_worker = client.clone();
    let worker = std::thread::spawn(move || {
        client_worker.external_op::<R>(&[src], &[dst_worker], move |context| {
            context.retain(owner);
            assert_eq!(size(&context.reads()[0]), len, "the read's size");
            assert_eq!(size(&context.writes()[0]), len, "the write's size");
            copy(
                context.stream(),
                context.reads()[0],
                context.writes()[0],
                len,
            )?;
            Ok(())
        });
    });

    worker.join().expect("the submitting thread");
    assert!(
        !dropped.is_dropped(),
        "the owner stays retained after the callback returns while its queued work is blocked"
    );

    future::block_on(client.sync_buffers([&dst]))
        .expect("the completion barrier and the write check succeed");
    dropped.wait("the server releases the owner once the blocked work completes");
}

/// The owner registered by a successful operation outlives the enqueue.
pub fn test_external_op_retained_until_completion<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    retained_until_completion::<R>(client, copy, size, false);
}

/// A partial enqueue followed by `Err` keeps the registered owner until the
/// queue drains too.
pub fn test_external_op_retained_on_error<R: NativeRuntime>(
    client: Client,
    copy: ExternalCopy<R>,
    size: ExternalSize<R>,
) {
    retained_until_completion::<R>(client, copy, size, true);
}

/// Asking for another runtime's external operation reports the mismatch on the
/// local writes and never runs the callback.
pub fn test_external_op_runtime_mismatch(client: Client) {
    let out = client.empty(core::mem::size_of::<u32>());
    let called = Arc::new(AtomicBool::new(false));
    let observed = called.clone();

    client.external_op::<OtherRuntime>(&[], core::slice::from_ref(&out), move |_context| {
        observed.store(true, Ordering::SeqCst);
        Ok(())
    });

    let err = client
        .read_one(out)
        .expect_err("a runtime mismatch leaves an error on the writes");
    let message = format!("{err}");
    assert!(
        message.contains("reaches"),
        "the read must report the refusal, got: {message}"
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "a mismatched runtime's operation must not run"
    );
}

/// A capture window refuses the operation and dooms the recording instead of
/// sealing a graph that silently omits it.
pub fn test_external_op_refused_during_capture<R: NativeRuntime>(client: Client) {
    let out = client.empty(core::mem::size_of::<u32>());
    client.graph_prepare().expect("graph_prepare");
    client.start_capture().expect("start_capture");

    let called = Arc::new(AtomicBool::new(false));
    let observed = called.clone();
    client.external_op::<R>(&[], &[out], move |_context| {
        observed.store(true, Ordering::SeqCst);
        Ok(())
    });

    let err = client
        .stop_capture()
        .expect_err("an external operation dooms the capture");
    let message = format!("{err}");
    assert!(
        message.contains("external"),
        "the capture must fail naming the external operation, got: {message}"
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "the operation must not run inside a capture"
    );
    assert!(!client.is_capturing(), "the failed capture ends");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_external_op {
    ($copy:path, $fill:path, $size:path) => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn external_op_sees_earlier_work() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_sees_earlier_work::<
                TestRuntime,
            >(client, $copy, $size);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_write_is_seen_by_a_later_launch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_write_is_seen_by_a_later_launch::<
                TestRuntime,
            >(client, $copy, $size);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_write_resolves_cross_stream() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_write_resolves_cross_stream::<
                TestRuntime,
            >(client, $fill, $size);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_error_taints_writes() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_error_taints_writes::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_tainted_input_skips_it() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_tainted_input_skips_it::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_panic_is_settled() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_panic_is_settled::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_in_place() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_in_place::<TestRuntime>(
                client, $copy, $size,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_retained_until_completion() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_retained_until_completion::<
                TestRuntime,
            >(client, $copy, $size);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_retained_on_error() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_retained_on_error::<
                TestRuntime,
            >(client, $copy, $size);
        }

        #[$crate::runtime_tests::test_log::test]
        fn external_op_runtime_mismatch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::external_op::test_external_op_runtime_mismatch(client);
        }
    };
}

// A second runtime with a different server type, only to ask a client for an
// external operation it must refuse before submitting. Nothing here is meant
// to run: the mismatch is caught on the client.
#[derive(Clone, Debug)]
struct OtherRuntime;

#[derive(Clone, Debug, Default)]
struct OtherDevice;

impl cubecl_common::device::Device for OtherDevice {
    fn from_id(_device_id: cubecl_common::device::DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> cubecl_common::device::DeviceId {
        cubecl_common::device::DeviceId::new(0, 0)
    }
}

#[derive(Debug)]
struct OtherServer;

#[derive(Debug)]
struct OtherStorage;

impl cubecl_common::device::DeviceService for OtherServer {
    fn init(_device_id: cubecl_common::device::DeviceId) -> Self {
        unimplemented!("the mismatch test never starts this server")
    }

    fn utilities(&self) -> cubecl_common::device::ServerUtilitiesHandle {
        unimplemented!("the mismatch test never asks for these")
    }
}

impl cubecl_runtime::server::ServerCommunication for OtherServer {}

impl cubecl_runtime::storage::ComputeStorage for OtherStorage {
    type Resource = ();

    fn alignment(&self) -> usize {
        1
    }

    fn get(
        &mut self,
        _handle: &cubecl_runtime::storage::StorageHandle,
    ) -> Result<Self::Resource, cubecl_runtime::server::IoError> {
        unimplemented!("the mismatch test never resolves storage")
    }

    fn alloc(
        &mut self,
        _size: u64,
    ) -> Result<cubecl_runtime::storage::StorageHandle, cubecl_runtime::server::IoError> {
        unimplemented!("the mismatch test never allocates")
    }

    fn dealloc(&mut self, _id: cubecl_runtime::storage::StorageId) {}

    fn flush(&mut self) {}

    fn bytes_allocated(&self) -> u64 {
        0
    }
}

impl cubecl_runtime::server::Server for OtherServer {
    fn initialize_memory(
        &mut self,
        _memory: cubecl_runtime::memory_management::ManagedMemoryHandle,
        _size: u64,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<(), ServerError> {
        unimplemented!("the mismatch test never initializes memory")
    }

    fn logger(&self) -> Arc<cubecl_runtime::logging::ServerLogger> {
        unimplemented!("the mismatch test never logs")
    }

    fn utilities(&self) -> Arc<cubecl_runtime::server::ServerUtilities> {
        unimplemented!("the mismatch test never asks for these")
    }

    fn read(
        &mut self,
        _descriptors: Vec<cubecl_runtime::server::CopyDescriptor>,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> cubecl_environment::future::DynFut<Result<Vec<cubecl_common::bytes::Bytes>, ServerError>>
    {
        unimplemented!("the mismatch test never reads")
    }

    fn write(
        &mut self,
        _descriptors: Vec<(
            cubecl_runtime::server::CopyDescriptor,
            cubecl_common::bytes::Bytes,
        )>,
        _stream_id: cubecl_environment::stream::StreamId,
    ) {
        unimplemented!("the mismatch test never writes")
    }

    fn sync(
        &mut self,
        _handles: Vec<cubecl_runtime::server::BufferBinding>,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> cubecl_environment::future::DynFut<Result<(), ServerError>> {
        unimplemented!("the mismatch test never syncs")
    }

    fn check(
        &mut self,
        _handles: Vec<cubecl_runtime::server::BufferBinding>,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<(), ServerError> {
        unimplemented!("the mismatch test never checks")
    }

    unsafe fn launch(
        &mut self,
        _kernel: Box<dyn cubecl_runtime::kernel::CubeKernel>,
        _count: CubeCount,
        _bindings: cubecl_runtime::server::KernelArguments,
        _stream_id: cubecl_environment::stream::StreamId,
        _launch_mode: cubecl_runtime::execution::LaunchMode,
    ) {
        unimplemented!("the mismatch test never launches")
    }

    fn external_op(
        &mut self,
        _reads: Vec<cubecl_runtime::server::BufferBinding>,
        _writes: Vec<cubecl_runtime::server::BufferBinding>,
        _op: ExternalOpCallback,
        _stream_id: cubecl_environment::stream::StreamId,
    ) {
        unimplemented!("the mismatch test never reaches this server")
    }

    fn flush(
        &mut self,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<(), ServerError> {
        unimplemented!("the mismatch test never flushes")
    }

    fn memory_report(
        &mut self,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Option<cubecl_runtime::memory_management::StreamMemoryReport> {
        unimplemented!("the mismatch test never reports memory")
    }

    fn memory_cleanup(
        &mut self,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<(), ServerError> {
        unimplemented!("the mismatch test never cleans memory")
    }

    fn start_profile(
        &mut self,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<cubecl_runtime::server::ProfilingToken, ServerError> {
        unimplemented!("the mismatch test never profiles")
    }

    fn end_profile(
        &mut self,
        _stream_id: cubecl_environment::stream::StreamId,
        _token: cubecl_runtime::server::ProfilingToken,
    ) -> Result<cubecl_common::profile::ProfileDuration, cubecl_runtime::server::ProfileError> {
        unimplemented!("the mismatch test never profiles")
    }

    fn allocation_mode(
        &mut self,
        _mode: cubecl_runtime::memory_management::MemoryAllocationMode,
        _stream_id: cubecl_environment::stream::StreamId,
    ) {
        unimplemented!("the mismatch test never changes the allocation mode")
    }
}

impl cubecl_runtime::server::ServerStorage for OtherServer {
    type Storage = OtherStorage;

    fn get_resource(
        &mut self,
        _binding: cubecl_runtime::server::BufferBinding,
        _stream_id: cubecl_environment::stream::StreamId,
    ) -> Result<cubecl_runtime::storage::ManagedResource<()>, ServerError> {
        unimplemented!("the mismatch test never resolves a resource")
    }
}

impl Runtime for OtherRuntime {
    type Server = OtherServer;
    type Device = OtherDevice;

    fn can_read_tensor(_shape: &cubecl::zspace::Shape, _strides: &cubecl::zspace::Strides) -> bool {
        unimplemented!("the mismatch test never reads a tensor")
    }

    fn target_properties() -> cubecl::ir::TargetProperties {
        unimplemented!("the mismatch test never compiles")
    }

    fn enumerate_devices(_type_id: u16) -> Vec<cubecl_common::device::DeviceId> {
        unimplemented!("the mismatch test never enumerates devices")
    }
}

impl NativeRuntime for OtherRuntime {
    type NativeStream = ();
    type NativeBuffer = ();
}
