use super::Handle;
use crate::kernel::BufferIOAttr;
use crate::{
    client::ComputeClient,
    compiler::CompilationError,
    config::{CubeClRuntimeConfig, RuntimeConfig, compilation::BoundsCheckMode},
    dry_run::LaunchMode,
    id::GraphId,
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{
        InstallMemoryPoolsError, ManagedMemoryHandle, ManagedMemoryId, MemoryAllocationMode,
        MemoryConfiguration, MemoryReport, MemoryUsage,
    },
    runtime::Runtime,
    server::{BufferBinding, KernelResource},
    storage::{ComputeStorage, ManagedResource},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use ahash::AHasher;
use alloc::boxed::Box;
#[cfg(feature = "profile-tracy")]
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{
    fmt::Debug,
    hash::{Hash, Hasher},
};
use cubecl_common::{
    bytes::Bytes,
    device::{self, DeviceId},
    profile::ProfileDuration,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashSet;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::RwLock;
use cubecl_ir::{DeviceProperties, ElemType, settings::Dim3};
use cubecl_zspace::{Shape, Strides, metadata::Metadata};
use derive_more::{Deref, DerefMut, From};
use itertools::Itertools;
use thiserror::Error;

#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
/// An error during profiling.
pub enum ProfileError {
    /// An unknown error happened during profiling
    #[error(
        "An unknown error happened during profiling\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Unknown {
        /// The caused of the error
        reason: String,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// No profiling was registered
    #[error("No profiling registered\nBacktrace:\n{backtrace}")]
    NotRegistered {
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Server(#[from] Box<ServerError>),
}

/// A failure during a profiling window invalidates the measurement, whatever
/// the failure was. Every backend answers a launch, write or replay failure
/// this way, so the conversion lives here rather than five times over.
impl From<&ServerError> for ProfileError {
    fn from(error: &ServerError) -> Self {
        ProfileError::Server(Box::new(error.clone()))
    }
}

impl core::fmt::Debug for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Contains many different types that are useful for server implementations and compute clients.
pub struct ServerUtilities<Server: ComputeServer> {
    /// The time when `profile-tracy` is activated.
    #[cfg(feature = "profile-tracy")]
    pub epoch_time: cubecl_environment::time::Instant,
    /// The GPU client when `profile-tracy` is activated.
    #[cfg(feature = "profile-tracy")]
    pub gpu_client: tracy_client::GpuContext,
    /// Information shared between all servers.
    pub properties: DeviceProperties,
    /// Stable hash of the device properties
    pub properties_hash: u64,
    /// Information specific to the current server.
    pub info: Server::Info,
    /// The logger based on global cubecl configs.
    pub logger: Arc<ServerLogger>,
    /// How to create the allocation.
    pub layout_policy: Server::MemoryLayoutPolicy,
    /// How to enforce bounds checking on kernels.
    pub check_mode: BoundsCheckMode,
    /// A set containing the ids for which the inter-device communication has already been initialized.
    pub initialized_comms: RwLock<HashSet<CommunicationId>>,
}

/// Defines how the memory layout is determined.
pub trait MemoryLayoutPolicy: Send + Sync + 'static {
    /// Applies the memory layout policy to a list of descriptors.
    ///
    /// Returns a vector of `MemoryLayout`, one per descriptor, with layouts that share a
    /// single `Binding`.
    fn apply(
        &self,
        stream_id: StreamId,
        descriptors: &[MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>);
}

impl<Server: core::fmt::Debug> core::fmt::Debug for ServerUtilities<Server>
where
    Server: ComputeServer,
    Server::Info: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ServerUtilities")
            .field("properties", &self.properties)
            .field("info", &self.info)
            .field("logger", &self.logger)
            .finish()
    }
}

impl<S: ComputeServer> ServerUtilities<S> {
    /// Creates a new server utilities.
    pub fn new(
        properties: DeviceProperties,
        logger: Arc<ServerLogger>,
        info: S::Info,
        allocator: S::MemoryLayoutPolicy,
    ) -> Self {
        // Start a tracy client if needed.
        #[cfg(feature = "profile-tracy")]
        let client = tracy_client::Client::start();

        Self {
            properties_hash: properties.checksum(),
            properties,
            logger,
            // Create the GPU client if needed.
            #[cfg(feature = "profile-tracy")]
            gpu_client: client
                .clone()
                .new_gpu_context(
                    Some(&format!("{info:?}")),
                    // In the future should ask the server what makes sense here. 'Invalid' atm is a generic stand-in (Tracy doesn't have CUDA/RocM atm anyway).
                    tracy_client::GpuContextType::Invalid,
                    0,   // Timestamps are manually aligned to this epoch so start at 0.
                    1.0, // Timestamps are manually converted to be nanoseconds so period is 1.
                )
                .unwrap(),
            #[cfg(feature = "profile-tracy")]
            epoch_time: cubecl_environment::time::Instant::now(),
            info,
            layout_policy: allocator,
            check_mode: CubeClRuntimeConfig::get().compilation.check_mode,
            initialized_comms: RwLock::new(HashSet::default()),
        }
    }
}

/// Kernel Launch Errors.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum LaunchError {
    /// The given kernel can't be compiled.
    #[error("A compilation error happened during launch\nCaused by:\n  {0}")]
    CompilationError(#[from] CompilationError),

    /// The server is out of memory.
    #[error(
        "An out-of-memory error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    OutOfMemory {
        /// The caused of the memory error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Too many resources were requested
    #[error("Too many resources were requested during launch\n{0}")]
    TooManyResources(#[from] ResourceLimitError),

    /// Unknown launch error.
    #[error(
        "An unknown error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    Unknown {
        /// The caused of the unknown error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

/// Resource limit errors.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceLimitError {
    /// Shared memory exceeds maximum
    #[error(
        "Too much shared memory requested.\nRequested {requested} bytes, maximum {max} bytes available.\nBacktrace\n{backtrace}"
    )]
    SharedMemory {
        /// Value requested
        requested: usize,
        /// Maximum value
        max: usize,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
    /// Total units exceeds maximum
    #[error(
        "Total unit count exceeds maximum.\nRequested {requested} units, max units is {max}.\nBacktrace\n{backtrace}"
    )]
    Units {
        /// Requested value
        requested: u32,
        /// Maximum value
        max: u32,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
    /// `CubeDim` exceeds maximum
    #[error(
        "Cube dim exceeds maximum bounds.\nRequested {requested:?}, max is {max:?}.\nBacktrace\n{backtrace}"
    )]
    CubeDim {
        /// Requested value
        requested: (u32, u32, u32),
        /// Maximum value
        max: (u32, u32, u32),
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl core::fmt::Debug for ResourceLimitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Error that can happen asynchronously while executing registered kernels.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ServerError {
    /// A runtime validation error
    #[error(
        "A validation error happened during execution\nCaused by:\n  {message}\nBacktrace:\n{backtrace}"
    )]
    Validation {
        /// The details of the validation error.
        message: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A generic runtime error.
    #[error("An error happened during execution\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    Generic {
        /// The details of the generic error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened
    #[error("A launch error happened\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An IO error happened
    #[error("An IO error happened\nCaused by:\n  {0}")]
    Io(#[from] IoError),

    /// The work writing this buffer was torn down before it could say what
    /// went wrong: its write scope never reached the exit that names the real
    /// failure, which a panic mid-launch explains.
    ///
    /// This is the provisional error every write scope enters with, so it
    /// carries no payload and captures no backtrace — a launch that succeeds
    /// mints one and drops it again, and paying a `String` and a stack walk
    /// per launch for the message nobody normally reads is the whole reason
    /// it is a variant rather than a [`Generic`](Self::Generic).
    #[error(
        "The work writing this buffer was torn down before it could say what went wrong: its \
         write scope never reached the exit that names the real failure, which a panic \
         mid-launch explains"
    )]
    TornDown,

    /// The bytes asked about were never written: the work that was going to
    /// write them failed, or was skipped downstream of a failure. `chain`
    /// walks from the buffer asked about back toward the root, newest skip
    /// first, and `root` is the failure that started it.
    #[error(
        "The bytes were never written (failure #{failure}, still claiming {claimed} buffer(s))\n{}Caused by:\n  {root}\nAsked at:\n{backtrace}",
        chain.iter().map(|hop| alloc::format!("  {hop}\n")).collect::<String>()
    )]
    Unwritten {
        /// The failure's id in the device's error store, as printed by every
        /// other read that trips over the same failure.
        failure: u64,
        /// How many buffers the failure still claims.
        claimed: u32,
        /// The skip chain from the buffer asked about back toward the root.
        chain: Vec<String>,
        /// The failure that started it, backtrace included.
        root: Box<ServerError>,
        /// Where the question was asked, so the lazy report and the read that
        /// tripped over it can be tied together.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// The work did not run, because an input it needed carried a failure.
    ///
    /// The report is on the buffers: the work's outputs claim the failure its
    /// inputs did, so a read of one of them names the root cause and the path
    /// back to it. This variant says only *that* the caller's work was
    /// skipped, which is why it carries no payload — the failure the inputs
    /// held is not the caller's to receive here, and minting a formatted
    /// message per skip would cost the loop that skips on every iteration.
    #[error(
        "The work was skipped: an input carried a failure, and the work's outputs claim it now \
         — a read of one of them names the root cause"
    )]
    Skipped,

    /// More than one thing went wrong at once, and the caller is owed all of
    /// them: a read naming buffers that several distinct failures claim, or a
    /// capture that was both doomed and abandoned.
    #[error("Several failures at once\nCaused by:\n  {}", errors.iter().join("\n"))]
    Several {
        /// The failures, in the order they were found.
        errors: Vec<Self>,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

impl Debug for ServerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl ServerError {
    /// Whether this is the kernel being refused before it ran, rather than
    /// something going wrong while running it.
    ///
    /// The distinction a test harness or an autotuner needs: a kernel a
    /// backend cannot build at this configuration is a candidate to drop or a
    /// case to skip, while a fault, an out-of-memory or an IO failure is a
    /// defect that has to be reported. Answering it by reading the message is
    /// how a harness ends up accepting the second as the first.
    ///
    /// Walks [`Several`](Self::Several) and [`Unwritten`](Self::Unwritten) to
    /// the roots, because a read of an unwritten buffer reports the failure
    /// that stopped its writer and that is where the distinction lives. A
    /// group answers yes only when every root does: one real failure among
    /// refusals is still a real failure, and an empty group refuses nothing.
    pub fn is_refusal(&self) -> bool {
        match self {
            Self::Launch(LaunchError::CompilationError(_) | LaunchError::TooManyResources(_)) => {
                true
            }
            Self::Unwritten { root, .. } => root.is_refusal(),
            Self::Several { errors, .. } => {
                !errors.is_empty() && errors.iter().all(Self::is_refusal)
            }
            _ => false,
        }
    }

    /// A graph-capture call the stream's lifecycle does not allow — a
    /// `begin_capture` without `graph_prepare`, a second overlapping capture, a
    /// replay of an unknown graph, or an operation a capture window cannot
    /// record. `reason` names the call and what was wrong with it.
    pub fn graph_state(reason: impl Into<String>) -> Self {
        Self::Generic {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        }
    }

    /// The error the default (unsupported) graph-capture methods return, for a
    /// backend that has no graph support at all.
    pub fn graph_capture_unsupported() -> Self {
        Self::graph_state("graph capture is not supported by this backend")
    }
}

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [`ComputeClient`] for thread safety.
pub trait ComputeServer:
    Send + core::fmt::Debug + ServerCommunication + device::DeviceService + 'static
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: KernelMetadata;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// Manages how allocations are performed for a server.
    type MemoryLayoutPolicy: MemoryLayoutPolicy;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;

    /// Initializes [memory](ManagedMemoryHandle) on the given [stream](StreamId) with the given size.
    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId);

    /// Reserves N [Bytes] of the provided sizes to be used as staging to load data.
    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Retrieve the server utilities.
    fn utilities(&self) -> Arc<ServerUtilities<Self>>;

    /// Given bindings, returns the owned resources as bytes.
    ///
    /// # Errors
    ///
    /// [`ServerError::Several`] when the work that was supposed to
    /// write one of these buffers failed, whichever stream it ran on — copying
    /// bytes out would hand back whatever was in memory before. Every
    /// implementation asks
    /// [`FailureStore::ensure_written`](crate::stream::FailureStore::ensure_written)
    /// before it copies anything.
    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>>;

    /// Writes the specified bytes into the buffers given
    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId);

    /// Wait for the completion of every task in the server, then answer for
    /// `handles`: the barrier first, so device faults count, and then the
    /// claim check a read would have made — a read without the read.
    ///
    /// An empty `handles` is the plain barrier plus the device fault, which
    /// is the only failure left that no buffer can report.
    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>>;

    /// Whether the bytes the handles name can be trusted, right now and with
    /// no barrier: the claim check a read makes, without the read. Instant —
    /// enqueue-time failures only. A device fault needs [`sync`](Self::sync),
    /// which drains first.
    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> Result<(), ServerError>;

    /// Given a resource handle, returns the storage resource.
    ///
    /// The same claim check a read makes guards this too: a buffer a failed
    /// launch never filled reports the failure rather than handing back a
    /// pointer to whatever was there before. It costs a field read on a slice
    /// the resolution walks anyway.
    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// `launch_mode` says whether the kernel actually runs. On
    /// [`LaunchMode::Skip`] the server must still do everything a first launch
    /// does short of dispatching — expand, compile, validate, fill its caches —
    /// and then drop the launch; skipping the compilation instead would defeat
    /// the whole point of a [dry run](crate::dry_run).
    ///
    /// # Safety
    ///
    /// When executing with mode [`ExecutionMode::Unchecked`], out-of-bound reads and writes can happen.
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    );

    /// Flush all outstanding tasks in the server.
    ///
    /// # Errors
    ///
    /// The device fault, when the context itself is broken — a launch failure
    /// is not the flush's to report: it lives on the buffers the launch left
    /// unwritten, and surfaces on any read, sync or check of them.
    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError>;

    /// Prepare `stream_id` for an upcoming graph capture: route allocations
    /// into a stable pool and snapshot it, so every buffer allocated between
    /// here and [`end_capture`](ComputeServer::end_capture) can be pinned for
    /// the graph's lifetime. Call this **before** the warmup run so the capture
    /// window reuses the slices warmup left in the pool rather than allocating
    /// its own — which a hardware-graph backend cannot do at all (a device
    /// malloc inside the capture is illegal there), and which on any backend
    /// would grow the memory a graph pins beyond what it replays against.
    ///
    /// Prefer having kernels already **autotuned before** this call: any
    /// transient benchmark buffers autotune allocates while the window is armed
    /// are forced into the persistent pool and pinned to the graph, so a graph
    /// captured over a cold autotune cache retains more device memory than it
    /// replays against. Warm the autotune cache first, then `graph_prepare` and
    /// warm up only to populate the pool.
    ///
    /// A no-op by default (harmless on backends without graph support); a
    /// backend with graph support enables its persistent pool + capture
    /// recording.
    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let _ = stream_id;
        Ok(())
    }

    /// Begin recording the launches issued on `stream_id` into a graph instead
    /// of executing them, so the sequence can later be
    /// [replayed](ComputeServer::replay) without paying the launch path again.
    /// Call [`graph_prepare`](ComputeServer::graph_prepare) and warm up first.
    ///
    /// Between this call and [`end_capture`](ComputeServer::end_capture) the
    /// stream must not synchronize — a read, a sync or a profile either aborts
    /// the capture or is refused — and should not allocate fresh device memory,
    /// which `graph_prepare` plus a warmup run is what avoids. Whether an
    /// operation the window cannot record fails the call or fails
    /// `end_capture`, and whether a mid-window allocation is fatal, is the
    /// backend's to say; see [`StreamCapture`](crate::stream::StreamCapture).
    ///
    /// The default is unsupported. Two shapes of backend override it: a
    /// **hardware graph** (CUDA, HIP), where the driver records a replayable
    /// graph object, and a **software graph** (wgpu), where the runtime records
    /// fully-resolved dispatches and re-encodes them on replay.
    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let _ = stream_id;
        Err(ServerError::graph_capture_unsupported())
    }

    /// Stop recording (see [`begin_capture`](ComputeServer::begin_capture)),
    /// store the captured graph in the backend's registry, and return its
    /// [`GraphId`], ready to [replay](ComputeServer::replay).
    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let _ = stream_id;
        Err(ServerError::graph_capture_unsupported())
    }

    /// Replay the graph identified by `graph` on `stream_id`, re-running the
    /// whole recorded launch sequence against its original buffers. A hardware
    /// graph replays as a single dispatch; a software graph re-encodes the
    /// recorded dispatches, which is still far cheaper than the launch path but
    /// stays O(n) in recorded launches.
    ///
    /// The call enqueues the dispatch and returns without waiting for the
    /// device; what it reports is the enqueue — an unknown or destroyed
    /// graph, a refusal — since a caller replaying a graph is standing right
    /// there. A failure also leaves the graph's write set carrying it, so a
    /// read of those buffers fails until a replay lands. Unsupported by
    /// default: a [`GraphId`] can only come from
    /// [`end_capture`](ComputeServer::end_capture).
    fn replay(&mut self, graph: GraphId, stream_id: StreamId) -> Result<(), ServerError> {
        let _ = (graph, stream_id);
        Err(ServerError::graph_capture_unsupported())
    }

    /// Release the graph identified by `graph`, destroying whatever it recorded
    /// and unpinning the buffers it retained. Replay returns at enqueue time,
    /// so the backend must guarantee no in-flight replay can still read those
    /// buffers once they return to the pool — by syncing `stream_id` where
    /// nothing weaker will do (CUDA, HIP), or by relying on the queue ordering
    /// that already places a submitted replay ahead of any later write (wgpu).
    /// A no-op by default and for an unknown id.
    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        let _ = (graph, stream_id);
    }

    /// Memory usage of the given stream.
    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage;

    /// Structured per-pool report of the given stream's **main GPU** memory:
    /// each pool's shape, usage, and high-water marks, in allocation-routing
    /// order. The read side of a measured memory plan — see
    /// [`MemoryManagement::memory_report`](crate::memory_management::MemoryManagement::memory_report).
    fn memory_report(&mut self, stream_id: StreamId) -> MemoryReport;

    /// Stream ids the client should iterate to aggregate across the device.
    ///
    /// Default is just the calling stream, which is correct for
    /// non-multi-stream backends; multi-stream backends override to
    /// return one id per initialized stream pool slot.
    fn stream_ids(&self) -> Vec<StreamId> {
        Vec::from([StreamId::current()])
    }

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self, stream_id: StreamId);

    /// Install a new dynamic-pool layout for the device's **main GPU** memory.
    ///
    /// The calling stream's pools are rebuilt in place (see
    /// [`MemoryManagement::install_pools`](crate::memory_management::MemoryManagement::install_pools)
    /// — a rebuild only happens when nothing is live in them), and the layout
    /// becomes the one every stream created afterwards is built with. Pool
    /// layouts are a purely programmatic, runtime setting — there is no
    /// config-file pathway — so callers size them per workload (e.g. per model,
    /// just before loading it).
    ///
    /// # Errors
    ///
    /// [`PoolsInUse`](InstallMemoryPoolsError::PoolsInUse) when the calling
    /// stream kept its old layout because something was still live in its
    /// pools — e.g. a garbage-collection task that has not released its
    /// cross-stream pins yet, which can lag behind an explicit
    /// [`memory_cleanup`](Self::memory_cleanup). The layout still applies to
    /// streams created afterwards; retry to rebuild the calling stream too.
    ///
    /// [`Unsupported`](InstallMemoryPoolsError::Unsupported) from servers
    /// without configurable pools, which is the default implementation.
    fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        stream_id: StreamId,
    ) -> Result<(), InstallMemoryPoolsError> {
        let _ = (config, stream_id);
        Err(InstallMemoryPoolsError::Unsupported)
    }

    /// Enable collecting timestamps.
    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError>;

    /// Disable collecting timestamps.
    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError>;

    /// Update the memory mode of allocation in the server.
    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId);
}

/// An ID unique to any unordered combination of devices.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct CommunicationId {
    /// The ID as a `String`.
    pub id: u64,
}

impl From<Vec<DeviceId>> for CommunicationId {
    fn from(mut value: Vec<DeviceId>) -> Self {
        // Make sure that device ids are sorted so that any combination of the same devices uses the same communicator.
        value.sort();
        let mut hasher = AHasher::default();
        value.hash(&mut hasher);
        CommunicationId {
            id: hasher.finish(),
        }
    }
}

/// Different reduce operations.
pub enum ReduceOperation {
    /// Sum.
    Sum,
    /// Mean.
    Mean,
}

/// Defines functions for optimized data transfer between servers, supporting custom communication
/// mechanisms such as peer-to-peer communication or specialized implementations.
///
/// # Inside the tainted-buffer rules
///
/// A collective reads a source buffer and produces a destination one, and owes
/// the same two answers the rest of the server gives: ask whether the source
/// carries a failure on the way in (as [`read`](ComputeServer::read) does
/// through
/// [`FailureStore::ensure_written`](crate::stream::FailureStore::ensure_written)),
/// and taint the destination on the way out when the operation fails (as a
/// failed [`launch`](ComputeServer::launch) does). Skipping either lets a
/// collective reduce stale bytes across every device in the group, or leave a
/// destination that reads back clean when nothing wrote it.
pub trait ServerCommunication {
    /// Indicates whether server-to-server communication is enabled for this implementation.
    const SERVER_COMM_ENABLED: bool;

    /// Ensure that all queued collective operations have been executed.
    ///
    /// # Arguments
    ///
    /// * `stream_id` - The [`StreamId`] of the stream waiting for the sync.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        todo!() // For backends other than cuda.
    }

    /// Initialize the communication between the devices in `device_ids`.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - The IDs of the devices that need communication.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Performs an `all_reduce` operation on the input data and writes it to the output buffer.
    /// see <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce>
    ///
    /// # Arguments
    ///
    /// * `src` - The data to be reduced.
    /// * `dst` - Where to write the result.
    /// * `dtype` - The element type of the data being reduced
    /// * `stream_id` - The data's stream id.
    /// * `op` - The reduce's aggregation operation e.g. mean, sum, etc.
    /// * `device_ids` - The list of device ids from which to `all_reduce`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn all_reduce(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Sends data from this server to a destination server.
    ///
    /// # Arguments
    ///
    /// * `desc` - A descriptor specifying the data to be sent, including shape, strides, and binding.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_dst` - ID of the device receiving the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    ///
    /// # Known limitation
    ///
    /// Send and recv are posted fire-and-forget on two devices and block for
    /// each other, so a send that refuses — a source whose writer failed,
    /// above all — leaves the peer's already-posted recv waiting on its
    /// communication stream with no way to recall it from here. The refusal
    /// is still right: completing the send would launder stale bytes onto a
    /// handle that carries no claim on the other device. Cross-device
    /// failure propagation needs a design pass of its own.
    #[allow(unused_variables)]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Receive data from another server.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle in which the received data is written.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_src` - ID of the device sending the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Profiling identification so that the server can support recursive and overlapping profilings.
pub struct ProfilingToken {
    /// The token value.
    pub id: u64,
}

/// Type of allocation, either contiguous or optimized (row-aligned when possible)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MemoryLayoutStrategy {
    /// Contiguous layout, with no padding
    Contiguous,
    /// Optimized for access speed. In practice this means row-aligned with padding for runtimes
    /// that support it.
    Optimized,
}

/// Descriptor for a new tensor allocation
#[derive(new, Debug, Clone)]
pub struct MemoryLayoutDescriptor {
    /// Strategy used to create the memory layout.
    pub strategy: MemoryLayoutStrategy,
    /// Shape of the tensor
    pub shape: Shape,
    /// Size of each element in the tensor (used for conversion of shape to bytes)
    pub elem_size: usize,
}

impl MemoryLayoutDescriptor {
    /// Create an optimized allocation descriptor
    pub fn optimized(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size)
    }

    /// Create a contiguous allocation descriptor
    pub fn contiguous(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, elem_size)
    }
}

/// An allocation with associated strides. Strides depend on tensor layout.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// The handle for the memory resource
    pub memory: Handle,
    /// TODO: `Strides` should become `Layout`.
    ///
    /// The strides of the tensor
    pub strides: Strides,
}

impl MemoryLayout {
    /// Create a new memory layout.
    pub fn new(handle: Handle, strides: impl Into<Strides>) -> Self {
        MemoryLayout {
            memory: handle,
            strides: strides.into(),
        }
    }
}

/// A reason for an error.
#[derive(Default, Clone)]
pub struct Reason {
    inner: ReasonInner,
}

#[cfg(std_io)]
mod _reason_serde {
    use super::*;

    use alloc::string::ToString;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Reason {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Use the Display implementation (via to_string) to flatten the enum
            serializer.serialize_str(&self.to_string())
        }
    }

    impl<'de> Deserialize<'de> for Reason {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            // Deserialize into a standard String first
            let s = String::deserialize(deserializer)?;

            // Wrap it in the Dynamic variant since we can't safely
            // reconstruct a 'static str from a runtime string.
            Ok(Reason {
                inner: ReasonInner::Dynamic(Arc::new(s)),
            })
        }
    }
}

#[derive(Default, Clone)]
enum ReasonInner {
    Static(&'static str),
    Dynamic(Arc<String>),
    #[default]
    NotProvided,
}

impl core::fmt::Display for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.inner {
            ReasonInner::Static(content) => f.write_str(content),
            ReasonInner::Dynamic(content) => f.write_str(content),
            ReasonInner::NotProvided => f.write_str("No reason provided for the error"),
        }
    }
}

impl core::fmt::Debug for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self, f)
    }
}

impl From<&'static str> for Reason {
    fn from(value: &'static str) -> Self {
        Self {
            inner: ReasonInner::Static(value),
        }
    }
}

impl From<String> for Reason {
    fn from(value: String) -> Self {
        Self {
            inner: ReasonInner::Dynamic(Arc::new(value)),
        }
    }
}

/// Error returned from `create`/`read`/`write` functions. Due to async execution not all errors
/// are able to be caught, so some IO errors will still panic.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum IoError {
    /// Buffer size exceeds the max available
    #[error("can't allocate buffer of size: {size}\n{backtrace}")]
    BufferTooBig {
        /// The size of the buffer in bytes.
        size: u64,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// The device had no memory left for this allocation.
    ///
    /// Unlike [`IoError::BufferTooBig`] (the allocation can *never* fit), this
    /// describes the device at one moment: pool pages whose slices have all
    /// been dropped are still resident, and the frees that would release them
    /// may not have reached the driver yet. Reclaiming and retrying is a
    /// reasonable response, which is why a storage backend must not report a
    /// driver out-of-memory as `BufferTooBig`: that tells every caller the
    /// allocation is hopeless when it is merely untimely.
    #[error("out of device memory allocating {size} bytes\n{backtrace}")]
    OutOfMemory {
        /// The size of the failed allocation in bytes.
        size: u64,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A memory pool with a fixed capacity cap is exhausted.
    ///
    /// Unlike [`IoError::BufferTooBig`] (the allocation can *never* fit), this
    /// means the working set exceeded the configured budget. Server execution
    /// paths treat it as fatal — the budget is a hard contract, so failing
    /// early beats silently growing — but callers that manage their own
    /// working set may free pool memory and retry.
    #[error(
        "memory pool capacity exceeded: failed to reserve {size} bytes, pool is capped at {capacity} bytes ({in_use} bytes in use)\n{backtrace}"
    )]
    PoolCapacityExceeded {
        /// The size of the failed reservation in bytes.
        size: u64,
        /// The configured pool capacity in bytes (whole pages).
        capacity: u64,
        /// Bytes currently in use in the pool.
        in_use: u64,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Strides aren't supported for this copy operation on this runtime
    #[error("the provided strides are not supported for this operation\n{backtrace}")]
    UnsupportedStrides {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Memory wasn't found in the memory pool
    #[error("couldn't find resource for that handle: {reason}\n{backtrace}")]
    NotFound {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
        /// The reason the handle is invalid.
        reason: Reason,
    },

    /// The storage backend holds no allocation for a handle's storage id.
    ///
    /// One layer below [`IoError::NotFound`]: there the memory manager could
    /// not route a binding to a slice, here the routing succeeded and the
    /// allocation the slice names is gone. A handle outliving its page, or a
    /// storage id a deallocation retired, reaches the storage this way.
    #[error("the storage holds no allocation for that handle: {reason}\n{backtrace}")]
    StorageHandleNotFound {
        /// Which id was looked up, and in which storage.
        reason: Reason,
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// An allocation carved lazily under a [`DryRun`](crate::dry_run::DryRun)
    /// could not be given real device backing when it was finally resolved.
    ///
    /// Distinct from the same failure at reservation time, and the distinction
    /// is what a caller acts on: the memory was promised earlier, by a pass
    /// that measured a plan without paying for it, and is only now being
    /// charged for. A plan whose replay hits this was measured against more
    /// device memory than the replay has — warm the tune caches first, or
    /// measure a smaller one.
    #[error(
        "couldn't map storage for a deferred allocation of {size} bytes\nCaused by:\n  {source}"
    )]
    StorageMappingFailed {
        /// The size of the allocation that could not be backed, in bytes.
        size: u64,
        /// Why the device allocation failed.
        source: Box<IoError>,
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Unknown error happened during execution
    #[error("Unknown error happened during execution: {description}\n{backtrace}")]
    Unknown {
        /// Details of the error
        description: String,
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// The current IO operation is not supported
    #[error("The current IO operation is not supported\n{backtrace}")]
    UnsupportedIoOperation {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

impl IoError {
    /// Whether reclaiming memory could still make this allocation succeed.
    ///
    /// Out of memory *right now* is not out of memory for good: pool pages
    /// whose slices have all been dropped are still resident, and the frees
    /// that would release them may sit in a deferred drop queue. A transient
    /// peak — a model build holding float weights while their quantized copies
    /// allocate, an autotune sample on a full device — is rescued by a reclaim
    /// and a second attempt.
    ///
    /// A buffer larger than any page the device can hold is the exception. It
    /// never fits, so reclaiming would only spend the time.
    pub fn may_succeed_after_reclaim(&self) -> bool {
        !matches!(self, IoError::BufferTooBig { .. })
    }
}

impl core::fmt::Debug for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Arguments to execute a kernel.
#[derive(Debug, Default)]
pub struct KernelArguments {
    /// Kernel bindings
    pub resources: Vec<KernelResource>,
    /// What the caller declared each resource is for, indexed like
    /// `resources`.
    ///
    /// The compiled kernel's own answer is better when it exists — the
    /// visibility analysis can prove a buffer write-only or dead, which a
    /// caller cannot — but it only exists once the kernel compiles. This one
    /// is stamped at the launch site from what the caller can see (a launch
    /// generated from `&Tensor` versus `&mut Tensor` knows it statically), so
    /// it survives the compile failing, which is exactly when it is needed: a
    /// launch that never ran must not taint the buffers it was only going to
    /// read. Missing entries read as [`ReadWrite`](BufferIOAttr::ReadWrite),
    /// so a caller that declares nothing keeps the loud fallback.
    pub declared_io: Vec<BufferIOAttr>,
    /// Packed scalars and metadata. First scalars sorted by type, then static metadata,
    /// then dynamic metadata.
    pub info: MetadataBindingInfo,
}

impl core::fmt::Display for KernelArguments {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KernelArguments")?;
        for b in self.resources.iter() {
            f.write_fmt(format_args!("\n - buffer: {b:?}\n"))?;
        }

        Ok(())
    }
}

impl KernelArguments {
    /// Create a new bindings struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a buffer binding
    pub fn with_buffer(mut self, binding: BufferBinding) -> Self {
        self.resources.push(KernelResource::Buffer(binding));
        self
    }

    /// Add a buffer binding, declaring what the kernel does with it.
    ///
    /// The declaration is what a launch that fails before running — a kernel
    /// that does not compile above all — falls back on: only declared-writable
    /// buffers take the failure, so the ones the kernel was only going to read
    /// stay readable. Resources added without a declaration read as
    /// [`ReadWrite`](BufferIOAttr::ReadWrite), and mixing the two keeps every
    /// declaration on the resource it was made for.
    pub fn with_buffer_io(mut self, binding: BufferBinding, io: BufferIOAttr) -> Self {
        self.declared_io
            .resize(self.resources.len(), BufferIOAttr::ReadWrite);
        self.resources.push(KernelResource::Buffer(binding));
        self.declared_io.push(io);
        self
    }

    /// Extend the buffers with `bindings`
    pub fn with_buffers(mut self, bindings: Vec<BufferBinding>) -> Self {
        let bindings = bindings.into_iter().map(KernelResource::Buffer);
        self.resources.extend(bindings);
        self
    }

    /// Set the info to `info`
    pub fn with_info(mut self, info: MetadataBindingInfo) -> Self {
        self.info = info;
        self
    }

    /// Extend the tensor maps with `bindings`
    pub fn with_tensor_maps(mut self, bindings: Vec<TensorMapBinding>) -> Self {
        let bindings = bindings.into_iter().map(KernelResource::TensorMap);
        self.resources.extend(bindings);
        self
    }

    /// The buffers this launch was given.
    pub fn buffers(&self) -> impl Iterator<Item = &BufferBinding> {
        self.resources.iter().map(|resource| match resource {
            KernelResource::Buffer(binding) => binding,
            KernelResource::TensorMap(tensor_map) => &tensor_map.binding,
        })
    }

    /// The memory this launch was given.
    pub fn memory_ids(&self) -> impl Iterator<Item = ManagedMemoryId> + '_ {
        self.buffers().map(|binding| binding.memory.id())
    }

    /// The buffers this launch was given that the kernel writes, per the
    /// compiled kernel's own answer — the ones a launch that fails taints,
    /// and nothing else.
    ///
    /// `io` is what the compiler recorded from its visibility analysis,
    /// indexed like `resources` (see
    /// [`BufferIOAttr`](crate::kernel::BufferIOAttr)). An index it has no
    /// answer for falls back to the caller's declaration in `declared_io` —
    /// which is how a kernel that never compiled still taints only its
    /// outputs — and an index neither answers reads as written: naming a
    /// buffer the kernel only read fails a read that would have been fine,
    /// loudly; missing one it writes hands back the bytes that were there
    /// before, silently — so the last-resort fallback over-names.
    pub fn buffers_written<'a>(
        &'a self,
        io: Option<&'a [BufferIOAttr]>,
    ) -> impl Iterator<Item = &'a BufferBinding> {
        self.buffers()
            .enumerate()
            .filter_map(move |(index, binding)| {
                let written = self
                    .io_attr(io, index)
                    .map(|io| io.is_writable())
                    .unwrap_or(true);
                written.then_some(binding)
            })
    }

    /// The buffers this launch was given that the kernel reads — the ones
    /// whose contents have to be trustworthy before the launch runs, and the
    /// only ones checked: a pure output is not read, so a relaunch into a
    /// tainted buffer is exactly how the buffer gets repaired.
    ///
    /// The same fallback chain as [`buffers_written`](Self::buffers_written):
    /// compiled answer, then the caller's declaration, then read — so a
    /// kernel nobody kept an answer for is checked on everything rather than
    /// checked on nothing.
    pub fn buffers_read<'a>(
        &'a self,
        io: Option<&'a [BufferIOAttr]>,
    ) -> impl Iterator<Item = &'a BufferBinding> {
        self.buffers()
            .enumerate()
            .filter_map(move |(index, binding)| {
                let read = self
                    .io_attr(io, index)
                    .map(|io| io.is_readable())
                    .unwrap_or(true);
                read.then_some(binding)
            })
    }

    /// The answer for one resource: the compiled kernel's when it kept one,
    /// the caller's declaration otherwise, `None` when neither answered.
    fn io_attr(&self, compiled: Option<&[BufferIOAttr]>, index: usize) -> Option<BufferIOAttr> {
        compiled
            .and_then(|io| io.get(index))
            .or_else(|| self.declared_io.get(index))
            .copied()
    }
}

/// Binding of a set of scalars of the same type to execute a kernel.
///
/// The [`ComputeServer`] is responsible to convert those info into actual [`Binding`] when launching
/// kernels.
#[derive(new, Debug, Default)]
pub struct MetadataBindingInfo {
    /// Scalar and metadata values
    pub data: Vec<u64>,
    /// Start of the dynamically sized portion of the metadata, relative to the entire info buffer
    pub dynamic_metadata_offset: usize,
}

impl MetadataBindingInfo {
    /// Create a new binding info for custom data, for externally compiled kernels.
    pub fn custom(data: Vec<u64>) -> Self {
        Self::new(data, 0)
    }
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug)]
pub struct CopyDescriptor {
    /// Binding for the memory resource
    pub handle: BufferBinding,
    /// Shape of the resource
    pub shape: Shape,
    /// Strides of the resource
    pub strides: Strides,
    /// Size of each element in the resource
    pub elem_size: usize,
}

/// A tensor map used with TMA ops
#[derive(new, Clone, Debug)]
pub struct TensorMapBinding {
    /// The binding for the backing tensor
    pub binding: BufferBinding,
    /// The tensormap metadata
    pub map: TensorMapMeta,
}

/// `TensorMap` metadata for the opaque proxy used in TMA copies
#[derive(Debug, Clone)]
pub struct TensorMapMeta {
    /// Tensormap format (tiled or im2col)
    pub format: TensorMapFormat,
    /// Metadata of the backing tensor
    pub metadata: Metadata,
    /// Element stride, usually 1 but may be 2 for complex tensors
    /// For im2col, this is equivalent to the kernel stride
    pub elem_stride: Strides,
    /// Interleave mode
    pub interleave: TensorMapInterleave,
    /// Swizzle mode
    pub swizzle: TensorMapSwizzle,
    /// Prefetch settings
    pub prefetch: TensorMapPrefetch,
    /// OOB fill value
    pub oob_fill: OobFill,
    /// Element type
    pub elem_ty: ElemType,
}

/// Specifieds the number of cubes to be dispatched for a kernel.
///
/// This translates to eg. a grid for CUDA, or to `num_workgroups` for wgsl.
#[allow(clippy::large_enum_variant)]
pub enum CubeCount {
    /// Dispatch a known count of x, y, z cubes.
    Static(u32, u32, u32),
    /// Dispatch an amount based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(BufferBinding),
}

/// Defines how to select cube count based on the number of cubes required.
pub enum CubeCountSelection {
    /// If the number of cubes is the same as required.
    Exact(CubeCount),
    /// If the number of cubes isn't the same as required.
    ///
    /// This can happen based on the hardware limit, requiring the kernel to perform OOB checks.
    Approx(CubeCount, u32),
}

impl CubeCountSelection {
    /// Creates a [`CubeCount`] while respecting the hardware limits.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, num_cubes: u32) -> Self {
        let cube_count = cube_count_spread(&client.properties().hardware.max_cube_count, num_cubes);

        let num_cubes_actual = cube_count[0] * cube_count[1] * cube_count[2];
        let cube_count = CubeCount::Static(cube_count[0], cube_count[1], cube_count[2]);

        match num_cubes_actual == num_cubes {
            true => CubeCountSelection::Exact(cube_count),
            false => CubeCountSelection::Approx(cube_count, num_cubes_actual),
        }
    }

    /// If some cubes will be idle.
    pub fn has_idle(&self) -> bool {
        matches!(self, Self::Approx(..))
    }

    /// Converts into [`CubeCount`].
    pub fn cube_count(self) -> CubeCount {
        match self {
            CubeCountSelection::Exact(cube_count) => cube_count,
            CubeCountSelection::Approx(cube_count, _) => cube_count,
        }
    }
}

impl From<CubeCountSelection> for CubeCount {
    fn from(value: CubeCountSelection) -> Self {
        value.cube_count()
    }
}

impl CubeCount {
    /// Create a new static cube count with the given x = y = z = 1.
    pub fn new_single() -> Self {
        CubeCount::Static(1, 1, 1)
    }

    /// Create a new static cube count with the given x, and y = z = 1.
    pub fn new_1d(x: u32) -> Self {
        CubeCount::Static(x, 1, 1)
    }

    /// Create a new static cube count with the given x and y, and z = 1.
    pub fn new_2d(x: u32, y: u32) -> Self {
        CubeCount::Static(x, y, 1)
    }

    /// Create a new static cube count with the given x, y and z.
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        CubeCount::Static(x, y, z)
    }

    /// Checks whether the cube count is definitely empty, i.e. has 0 dispatches.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Static(x, y, z) => *x == 0 || *y == 0 || *z == 0,
            Self::Dynamic(_) => false,
        }
    }
}

impl Debug for CubeCount {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CubeCount::Static(x, y, z) => f.write_fmt(format_args!("({x}, {y}, {z})")),
            CubeCount::Dynamic(_) => f.write_str("binding"),
        }
    }
}

impl Clone for CubeCount {
    fn clone(&self) -> Self {
        match self {
            Self::Static(x, y, z) => Self::Static(*x, *y, *z),
            Self::Dynamic(binding) => Self::Dynamic(binding.clone()),
        }
    }
}

#[derive(Debug, From, PartialEq, Eq, Clone, Copy, Hash, Deref, DerefMut)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
/// The number of units across all 3 axis totalling to the number of working units in a cube.
pub struct CubeDim(pub Dim3);

impl CubeDim {
    /// Creates a new [`CubeDim`] based on the maximum number of tasks that can be parellalized by units, in other words,
    /// by the maximum number of working units.
    ///
    /// # Notes
    ///
    /// For complex problems, you probably want to have your own logic function to create the
    /// [`CubeDim`], but for simpler problems such as elemwise-operation, this is a great default.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, working_units: usize) -> Self {
        let properties = client.properties();
        let plane_size = properties.hardware.plane_size_max;
        let plane_count = Self::calculate_plane_count_per_cube(
            working_units as u32,
            plane_size,
            properties.hardware.num_cpu_cores,
        );

        // Make sure it respects the max units per cube (especially on wasm)
        let limit = properties.hardware.max_units_per_cube / plane_size;

        // Ensure at least 1 plane so CubeDim is always valid (num_elems() > 0).
        Self::new_2d(plane_size, u32::min(limit, plane_count).max(1))
    }

    fn calculate_plane_count_per_cube(
        working_units: u32,
        plane_dim: u32,
        num_cpu_cores: Option<u32>,
    ) -> u32 {
        match num_cpu_cores {
            Some(num_cores) => core::cmp::min(num_cores, working_units),
            None => {
                let plane_count_max = core::cmp::max(1, working_units / plane_dim);

                // Ensures `plane_count` is a power of 2.
                const NUM_PLANE_MAX: u32 = 8u32;
                const NUM_PLANE_MAX_LOG2: u32 = NUM_PLANE_MAX.ilog2();
                let plane_count_max_log2 =
                    core::cmp::min(NUM_PLANE_MAX_LOG2, u32::ilog2(plane_count_max));
                2u32.pow(plane_count_max_log2)
            }
        }
    }

    /// Create a new cube dim with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self(Dim3::new_single())
    }

    /// Create a new cube dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self(Dim3::new_1d(x))
    }

    /// Create a new cube dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self(Dim3::new_2d(x, y))
    }

    /// Create a new cube dim with the given x, y and z.
    /// This is equivalent to the [new](CubeDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self(Dim3::new_3d(x, y, z))
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.0.num_elems()
    }

    /// Whether this `CubeDim` can fully contain `other`
    pub const fn can_contain(&self, other: CubeDim) -> bool {
        self.0.can_contain(other.0)
    }
}

impl From<(u32, u32, u32)> for CubeDim {
    fn from(value: (u32, u32, u32)) -> Self {
        CubeDim::new_3d(value.0, value.1, value.2)
    }
}

impl From<CubeDim> for (u32, u32, u32) {
    fn from(val: CubeDim) -> Self {
        (val.x, val.y, val.z)
    }
}

impl From<CubeDim> for Dim3 {
    fn from(value: CubeDim) -> Self {
        value.0
    }
}

fn cube_count_spread(max: &(u32, u32, u32), num_cubes: u32) -> [u32; 3] {
    let max_cube_counts = [max.0, max.1, max.2];
    let mut num_cubes = [num_cubes, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_cubes[i] <= max_cube_counts[i] {
            return true;
        }

        loop {
            num_cubes[i] = num_cubes[i].div_ceil(base);
            num_cubes[i + 1] *= base;

            if num_cubes[i] <= max_cube_counts[i] {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_cubes
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test_log::test]
    fn safe_num_cubes_even() {
        let max = (32, 32, 32);
        let required = 2048;

        let actual = cube_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test_log::test]
    fn safe_num_cubes_odd() {
        let max = (48, 32, 16);
        let required = 3177;

        let actual = cube_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }

    /// The compiled kernel's answer drives both sets exactly, in resource
    /// order.
    #[test_log::test]
    fn buffer_io_drives_the_read_and_write_sets() {
        use crate::kernel::BufferIOAttr;
        use cubecl_environment::stream::StreamId;

        let stream = StreamId { value: 0 };
        let args = KernelArguments::new().with_buffers(vec![
            Handle::new(stream, 8).binding(),
            Handle::new(stream, 8).binding(),
            Handle::new(stream, 8).binding(),
            Handle::new(stream, 8).binding(),
        ]);
        let io = [
            BufferIOAttr::ReadOnly,
            BufferIOAttr::WriteOnly,
            BufferIOAttr::ReadWrite,
            BufferIOAttr::Dead,
        ];

        let written: Vec<_> = args.buffers_written(Some(&io)).collect();
        assert_eq!(written.len(), 2, "WriteOnly and ReadWrite are written");
        assert!(core::ptr::eq(written[0], args.buffers().nth(1).unwrap()));
        assert!(core::ptr::eq(written[1], args.buffers().nth(2).unwrap()));

        let read: Vec<_> = args.buffers_read(Some(&io)).collect();
        assert_eq!(read.len(), 2, "ReadOnly and ReadWrite are read");
        assert!(core::ptr::eq(read[0], args.buffers().next().unwrap()));
        assert!(core::ptr::eq(read[1], args.buffers().nth(2).unwrap()));
    }

    /// A refusal is the kernel being turned down, and nothing else is.
    ///
    /// The direction that matters is the false positive: a harness that takes
    /// a device fault for a refusal reports a broken run as a skipped one, and
    /// the test goes green. So a group answers yes only when every root does.
    #[test_log::test]
    fn only_a_refused_kernel_reads_as_a_refusal() {
        use crate::server::{LaunchError, ResourceLimitError};

        let refused =
            ServerError::Launch(LaunchError::CompilationError(CompilationError::Generic {
                reason: "no such intrinsic on this target".into(),
                backtrace: Default::default(),
            }));
        let over_budget = ServerError::Launch(LaunchError::TooManyResources(
            ResourceLimitError::SharedMemory {
                requested: 1 << 20,
                max: 1 << 15,
                backtrace: Default::default(),
            },
        ));
        let fault = ServerError::Generic {
            reason: "the device faulted".into(),
            backtrace: Default::default(),
        };

        assert!(refused.is_refusal());
        assert!(over_budget.is_refusal());
        assert!(!fault.is_refusal(), "a fault is not a refusal");

        // A read reports the failure that stopped the buffer's writer, so the
        // question has to reach through the report to the root.
        let unwritten = |root: &ServerError| ServerError::Unwritten {
            failure: 1,
            claimed: 1,
            chain: Vec::new(),
            root: alloc::boxed::Box::new(root.clone()),
            backtrace: Default::default(),
        };
        assert!(unwritten(&refused).is_refusal());
        assert!(!unwritten(&fault).is_refusal());

        let group = |errors: Vec<ServerError>| ServerError::Several {
            errors,
            backtrace: Default::default(),
        };
        assert!(group(vec![unwritten(&refused), unwritten(&over_budget)]).is_refusal());
        assert!(
            !group(vec![unwritten(&refused), unwritten(&fault)]).is_refusal(),
            "one real failure among refusals is still a real failure"
        );
        assert!(
            !group(Vec::new()).is_refusal(),
            "an empty group refuses nothing"
        );
    }

    /// Every fallback over-names: a kernel the compiler kept no answer for,
    /// and a resource past what the answer covers, read as both read and
    /// written. Naming a buffer the kernel only read fails a read that would
    /// have been fine, loudly; missing one it writes hands back the bytes
    /// that were there before, silently.
    #[test_log::test]
    fn missing_io_reads_as_everything_read_and_written() {
        use crate::kernel::BufferIOAttr;
        use cubecl_environment::stream::StreamId;

        let stream = StreamId { value: 0 };
        let args = KernelArguments::new().with_buffers(vec![
            Handle::new(stream, 8).binding(),
            Handle::new(stream, 8).binding(),
        ]);

        assert_eq!(args.buffers_written(None).count(), 2);
        assert_eq!(args.buffers_read(None).count(), 2);

        let short = [BufferIOAttr::Dead];
        assert_eq!(
            args.buffers_written(Some(&short)).count(),
            1,
            "the uncovered resource reads as written"
        );
        assert_eq!(args.buffers_read(Some(&short)).count(), 1);
    }

    /// The caller's declaration answers when the compiled kernel kept none —
    /// which is what a launch that fails to compile falls back on, so it
    /// taints only its declared outputs — and the compiled answer still wins
    /// where it exists, since only the visibility analysis can prove a buffer
    /// write-only or dead.
    #[test_log::test]
    fn declared_io_answers_when_the_compiled_kernel_kept_none() {
        use crate::kernel::BufferIOAttr;
        use cubecl_environment::stream::StreamId;

        let stream = StreamId { value: 0 };
        let args = KernelArguments::new()
            .with_buffer_io(Handle::new(stream, 8).binding(), BufferIOAttr::ReadOnly)
            .with_buffer_io(Handle::new(stream, 8).binding(), BufferIOAttr::ReadOnly)
            .with_buffer_io(Handle::new(stream, 8).binding(), BufferIOAttr::WriteOnly);

        // No compiled answer: the declaration decides. The inputs are not
        // written, so a failed compile leaves them readable.
        let written: Vec<_> = args.buffers_written(None).collect();
        assert_eq!(written.len(), 1, "only the declared output is written");
        assert!(core::ptr::eq(written[0], args.buffers().nth(2).unwrap()));
        assert_eq!(args.buffers_read(None).count(), 2);

        // A compiled answer overrides the declaration where it has one and
        // falls back to it where it does not.
        let compiled = [BufferIOAttr::ReadWrite];
        let written: Vec<_> = args.buffers_written(Some(&compiled)).collect();
        assert_eq!(written.len(), 2, "compiled ReadWrite plus declared output");
        assert!(core::ptr::eq(written[0], args.buffers().next().unwrap()));
        assert!(core::ptr::eq(written[1], args.buffers().nth(2).unwrap()));
    }

    /// Declarations stay on the resource they were made for when declared and
    /// undeclared resources mix, and the undeclared ones keep the loud
    /// fallback.
    #[test_log::test]
    fn an_undeclared_resource_among_declared_ones_over_names() {
        use crate::kernel::BufferIOAttr;
        use cubecl_environment::stream::StreamId;

        let stream = StreamId { value: 0 };
        let args = KernelArguments::new()
            .with_buffer(Handle::new(stream, 8).binding())
            .with_buffer_io(Handle::new(stream, 8).binding(), BufferIOAttr::ReadOnly);

        let written: Vec<_> = args.buffers_written(None).collect();
        assert_eq!(written.len(), 1, "the undeclared resource reads as written");
        assert!(core::ptr::eq(written[0], args.buffers().next().unwrap()));
        assert_eq!(args.buffers_read(None).count(), 2);
    }
}
