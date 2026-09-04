//! What a backend supplies so the shared [`Command`](super::Command) can run.
//!
//! Two traits, because they answer different questions. [`Driver`] is the four
//! device calls the runtime cannot make itself. [`DeviceStream`] is where a
//! backend's stream keeps the state those calls move in step with — a command
//! reaches all of it, and none of it is the driver's to reach.

use crate::allocator::Pitch;
use crate::id::KernelId;
use crate::memory_management::drop_queue::{Fence, PendingDropQueue};
use crate::memory_management::{ManagedMemoryBinding, MemoryManagement};
use crate::metadata_cache::MetadataInfoCache;
use crate::server::{Handle, IoError, LaunchError};
use crate::storage::ComputeStorage;
use crate::stream::{EventStreamBackend, StreamCapture};
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_zspace::{Shape, Strides, striding::has_pitched_row_major_strides};

/// The state a backend's stream keeps beside its driver handle.
///
/// All of it moves in step with the work queued on that stream, which is why
/// it is the stream's and not the device's: the deferred frees a fenced flush
/// releases, the capture window that forbids allocating, the per-launch info
/// buffers a capture may not evict, and the memory the stream's kernels see.
pub trait DeviceStream {
    /// The fence this backend records on a stream. Fencing is how the drop
    /// queue knows the device is done with what it is holding.
    type Fence: Fence + Send + 'static;
    /// The storage backing the memory this stream's kernels address.
    type DeviceStorage: ComputeStorage;
    /// The storage backing the host buffers that stage transfers to it.
    type HostStorage: ComputeStorage;

    /// The memory this stream's kernels see. Allocations are per stream, so a
    /// buffer resolves to the stream that created it and nowhere else.
    fn device_memory(&mut self) -> &mut MemoryManagement<Self::DeviceStorage>;

    /// The pinned host memory this stream stages transfers through.
    fn host_memory(&mut self) -> &mut MemoryManagement<Self::HostStorage>;

    /// Frees deferred until the device is known to be done with them.
    fn drop_queue(&mut self) -> &mut PendingDropQueue<Self::Fence>;

    /// Where this stream sits in the graph-capture lifecycle.
    fn capturing(&mut self) -> &mut StreamCapture;

    /// The per-launch metadata buffers this stream reuses.
    fn info_cache(&mut self) -> &mut MetadataInfoCache<Handle>;

    /// A cheap, copyable identifier for this stream.
    ///
    /// It exists so a fence can be recorded while the stream's own fields are
    /// borrowed: draining the drop queue needs a fresh fence per rotation, and
    /// the queue is reached through `&mut self`.
    type Signal: Copy;

    /// This stream's signal.
    fn signal(&self) -> Self::Signal;

    /// Record a fence on the stream `signal` names, signalled once everything
    /// already enqueued on it has run.
    fn fence(signal: Self::Signal) -> Self::Fence;
}

/// The layout of the device side of a copy.
///
/// The pitch is computed once, by [`of`](Self::of), because whether a buffer's
/// rows are padded is the same question whichever driver performs the copy —
/// and getting it wrong scrambles the rows rather than failing.
///
/// Which is why a driver cannot build one. The fields are readable, since a
/// driver needs all four, but only `of` puts them together: a hand-built
/// layout could carry a pitch its strides do not agree with, and nothing
/// downstream would notice.
#[non_exhaustive]
pub struct CopyLayout<'a> {
    /// The extent of each dimension.
    pub shape: &'a Shape,
    /// The stride of each dimension, in elements.
    pub strides: &'a Strides,
    /// The size of one element, in bytes.
    pub elem_size: usize,
    /// The 2D geometry when the rows are padded; `None` when the whole buffer
    /// is one contiguous span and a linear copy is both correct and faster.
    pub pitch: Option<Pitch>,
}

impl<'a> CopyLayout<'a> {
    /// The layout of a copy over this shape, refusing anything the drivers
    /// cannot express.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major. A driver copies either one contiguous span or a stack of
    /// evenly-spaced rows; nothing else has a call to make.
    pub fn of(shape: &'a Shape, strides: &'a Strides, elem_size: usize) -> Result<Self, IoError> {
        if !has_pitched_row_major_strides(shape, strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }
        Ok(Self {
            shape,
            strides,
            elem_size,
            pitch: Pitch::of(shape, strides, elem_size),
        })
    }
}

/// The device calls a [`Command`](super::Command) cannot make itself.
///
/// Four, because everything else a command does — deciding what to stage, when
/// to reclaim, whether a layout needs a 2D copy, when the drop queue may be
/// flushed — is the same whichever driver is underneath.
pub trait Driver: Sized {
    /// The multi-stream backend whose streams this driver drives.
    type Backend: EventStreamBackend<Stream = Self::Stream>;
    /// The backend's stream.
    type Stream: DeviceStream;
    /// Whatever the backend needs to launch a compiled kernel — its loaded
    /// modules, its profiling clocks.
    type Context;
    /// What this backend hands a kernel at launch.
    ///
    /// Not simply the buffers: CUDA passes tensor-map descriptors alongside
    /// them and HIP has none, so what a launch is given is the backend's to
    /// say. The command only carries it from the server to [`launch`].
    ///
    /// [`launch`]: Self::launch
    type LaunchArgs: ?Sized;

    /// Hand out `size` bytes of the pinned host allocation `binding` names,
    /// released back to the pool when the [`Bytes`] drop.
    ///
    /// # Safety
    ///
    /// `binding` names initialized host memory of at least `size` bytes, and
    /// `resource` resolves it.
    unsafe fn pinned_bytes(
        binding: ManagedMemoryBinding,
        resource: HostResource<Self>,
        size: usize,
    ) -> Bytes;

    /// Enqueue a copy from device memory into `bytes` on `stream`.
    ///
    /// # Safety
    ///
    /// `resource` is a live device allocation of at least `bytes.len()`
    /// readable bytes, `bytes` has room for the copy, and the caller
    /// synchronizes `stream` before reading it back.
    ///
    /// # Errors
    ///
    /// The driver's refusal to copy.
    unsafe fn copy_to_host(
        resource: &DeviceResource<Self>,
        layout: &CopyLayout<'_>,
        bytes: &mut Bytes,
        stream: &Self::Stream,
    ) -> Result<(), IoError>;

    /// Enqueue a copy of `data` into device memory on `stream`.
    ///
    /// # Safety
    ///
    /// `resource` is a live device allocation big enough for `data`, and
    /// `data` stays alive until `stream` is synchronized — which is what the
    /// caller's drop queue guarantees.
    ///
    /// # Errors
    ///
    /// The driver's refusal to copy.
    unsafe fn copy_to_device(
        resource: &DeviceResource<Self>,
        layout: &CopyLayout<'_>,
        data: &[u8],
        stream: &Self::Stream,
    ) -> Result<(), IoError>;

    /// Enqueue an already-compiled kernel on `stream`.
    ///
    /// Always a compiled kernel: the server compiles before entering its write
    /// scope, and a skipped launch stops there, before any resource is
    /// resolved.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the launch.
    fn launch(
        ctx: &mut Self::Context,
        stream: &mut Self::Stream,
        kernel: KernelId,
        count: (u32, u32, u32),
        args: &mut Self::LaunchArgs,
    ) -> Result<(), LaunchError>;
}

/// The device allocation a driver's buffers resolve to.
pub type DeviceResource<D> =
    <<<D as Driver>::Stream as DeviceStream>::DeviceStorage as ComputeStorage>::Resource;

/// The host allocation a driver's staging buffers resolve to.
pub type HostResource<D> =
    <<<D as Driver>::Stream as DeviceStream>::HostStorage as ComputeStorage>::Resource;
