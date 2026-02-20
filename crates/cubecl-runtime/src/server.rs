use crate::{
    client::ComputeClient,
    compiler::CompilationError,
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    storage::{BindingResource, ComputeStorage},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
#[cfg(feature = "profile-tracy")]
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{
    fmt::Debug,
    sync::atomic::{AtomicU64, Ordering},
};
use cubecl_common::{
    backtrace::BackTrace, bytes::Bytes, device, future::DynFut, profile::ProfileDuration,
    stream_id::StreamId,
};
use cubecl_ir::{DeviceProperties, StorageType};
use cubecl_zspace::{Shape, Strides, metadata::Metadata};
use serde::{Deserialize, Serialize};
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

impl core::fmt::Debug for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Contains many different types that are useful for server implementations and compute clients.
pub struct ServerUtilities<Server: ComputeServer> {
    /// The time when `profile-tracy` is activated.
    #[cfg(feature = "profile-tracy")]
    pub epoch_time: web_time::Instant,
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
}

pub trait MemoryLayoutPolicy: Send + Sync + 'static {
    fn apply(&self, stream_id: StreamId, descriptor: &MemoryLayoutDescriptor) -> MemoryLayout;
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
            epoch_time: web_time::Instant::now(),
            info,
            layout_policy: allocator,
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

    /// Can't launch because of an IO Error.
    #[error("An io error happened during launch\nCaused by:\n  {0}")]
    IoError(#[from] IoError),
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
#[derive(Error, Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ServerError {
    /// A generic runtime error.
    #[error("An error happened during execution\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    Generic {
        /// The details of the generic error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Profile(#[from] ProfileError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Io(#[from] IoError),

    /// The server is an invalid state.
    #[error("The server is in an invalid state\nCaused by:\n  {reason}")]
    ServerUnHealty {
        /// The details of the generic error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
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

    /// Binds current [memory handle](Handle) to managed memory on the given [stream](StreamId).
    fn bind(&mut self, handles: Vec<Handle>, stream_id: StreamId);

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

    /// Clear the errors from the server as well as flushing all pending tasks.
    ///
    /// This essentially clear the server state.
    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError>;

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Retrieve the server utilities.
    fn utilities(&self) -> Arc<ServerUtilities<Self>>;

    /// Given bindings, returns the owned resources as bytes.
    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>>;

    /// Writes the specified bytes into the buffers given
    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>>;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Handle,
        stream_id: StreamId,
    ) -> Result<BindingResource<<Self::Storage as ComputeStorage>::Resource>, ServerError>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// # Safety
    ///
    /// When executing with mode [`ExecutionMode::Unchecked`], out-of-bound reads and writes can happen.
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        stream_id: StreamId,
    );

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError>;

    /// The current memory usage of the server.
    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError>;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self, stream_id: StreamId);

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

/// Defines functions for optimized data transfer between servers, supporting custom communication
/// mechanisms such as peer-to-peer communication or specialized implementations.
pub trait ServerCommunication {
    /// Indicates whether server-to-server communication is enabled for this implementation.
    const SERVER_COMM_ENABLED: bool;

    /// Copies data from a source server to a destination server.
    ///
    /// # Arguments
    ///
    /// * `server_src` - A mutable reference to the source server from which data is copied.
    /// * `server_dst` - A mutable reference to the destination server receiving the data.
    /// * `src` - A descriptor specifying the data to be copied, including shape, strides, and binding.
    /// * `stream_id_src` - The stream ID associated with the source server's operation.
    /// * `stream_id_dst` - The stream ID associated with the destination server's operation.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `Allocation` on success, or an `IoError` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if server communication is not enabled (`SERVER_COMM_ENABLED` is `false`) or if the
    /// trait is incorrectly implemented by the server.
    #[allow(unused_variables)]
    fn copy(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<MemoryLayout, IoError> {
        if !Self::SERVER_COMM_ENABLED {
            panic!("Server-to-server communication is not supported by this server.");
        } else {
            panic!(
                "[Internal Error] The `ServerCommunication` trait is incorrectly implemented by the server."
            );
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Profiling identification so that the server can support recursive and overlapping profilings.
pub struct ProfilingToken {
    /// The token value.
    pub id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HandleId {
    value: u64,
    count: Arc<()>,
}

static HANDLE_COUNT: AtomicU64 = AtomicU64::new(0);

impl HandleId {
    pub fn new() -> Self {
        let value = HANDLE_COUNT.fetch_add(1, Ordering::Acquire);
        Self {
            value,
            count: Arc::new(()),
        }
    }
    pub fn can_mut(&self) -> bool {
        // One reference by the server/queue.
        Arc::strong_count(&self.count) <= 2
    }
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.count) == 1
    }
}

/// Server handle containing the [memory handle](crate::server::Handle).
#[derive(new, Debug, PartialEq, Eq)]
pub struct Handle {
    /// Memory handle.
    pub id: HandleId,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    // /// The stream position when the tensor became available.
    // pub cursor: u64,
    /// Length of the underlying buffer ignoring offsets
    size: u64,
}

/// Defines how a block of [managed memory](ManagedMemoryHandle) can be viewed.
#[derive(new, Debug, PartialEq, Eq, Clone)]
pub struct MemorySlot {
    /// Memory handle.
    pub memory: ManagedMemoryHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream position when the tensor became available.
    pub cursor: u64,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
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
    pub handle: Handle,
    /// The strides of the tensor
    pub strides: Strides,
}

impl MemoryLayout {
    /// Create a new allocation.
    pub fn new(handle: Handle, strides: impl Into<Strides>) -> Self {
        MemoryLayout {
            handle,
            strides: strides.into(),
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

    /// Strides aren't supported for this copy operation on this runtime
    #[error("the provided strides are not supported for this operation\n{backtrace}")]
    UnsupportedStrides {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Handle wasn't found in the memory pool
    #[error("couldn't find resource for that handle\n{backtrace}")]
    InvalidHandle {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Unknown error happened during execution
    #[error("Unknown error happened during execution\n{backtrace}")]
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

    /// Can't perform the IO operation because of a runtime error.
    #[error("Can't perform the IO operation because of a runtime error: {0}")]
    Execution(#[from] Box<ServerError>),
}

impl core::fmt::Debug for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl Handle {
    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }
}

impl MemorySlot {
    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: Option<u64>) -> Self {
        let offset = match offset {
            Some(offset) => offset,
            None => return self,
        };
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: Option<u64>) -> Self {
        let offset = match offset {
            Some(offset) => offset,
            None => return self,
        };
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }
}

/// Bindings to execute a kernel.
#[derive(Debug, Default)]
pub struct Bindings {
    /// Buffer bindings
    pub handles: Vec<Handle>,
    /// Packed metadata for tensor bindings (len, shape, stride, etc).
    /// Ordered by inputs, then outputs, then tensormaps
    pub metadata: MetadataBinding,
    /// Scalar bindings
    pub scalars: BTreeMap<StorageType, ScalarBinding>,
    /// Tensor map bindings
    pub tensor_maps: Vec<TensorMapBinding>,
}

impl Bindings {
    /// Create a new bindings struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a buffer binding
    pub fn with_buffer(mut self, binding: Handle) -> Self {
        self.handles.push(binding);
        self
    }

    /// Extend the buffers with `bindings`
    pub fn with_buffers(mut self, bindings: Vec<Handle>) -> Self {
        self.handles.extend(bindings);
        self
    }

    /// Add a scalar parameter
    pub fn with_scalar(mut self, ty: StorageType, length: usize, data: Vec<u64>) -> Self {
        self.scalars
            .insert(ty, ScalarBinding::new(ty, length, data));
        self
    }

    /// Extend the scalars with `bindings`
    pub fn with_scalars(mut self, bindings: Vec<ScalarBinding>) -> Self {
        self.scalars
            .extend(bindings.into_iter().map(|binding| (binding.ty, binding)));
        self
    }

    /// Set the metadata to `meta`
    pub fn with_metadata(mut self, meta: MetadataBinding) -> Self {
        self.metadata = meta;
        self
    }

    /// Extend the tensor maps with `bindings`
    pub fn with_tensor_maps(mut self, bindings: Vec<TensorMapBinding>) -> Self {
        self.tensor_maps.extend(bindings);
        self
    }
}

/// Binding of a set of scalars of the same type to execute a kernel.
#[derive(new, Debug, Default)]
pub struct MetadataBinding {
    /// Metadata values
    pub data: Vec<u64>,
    /// Length of the static portion (rank, len, `buffer_len`, `shape_offsets`, `stride_offsets`).
    pub static_len: usize,
}

/// Binding of a set of scalars of the same type to execute a kernel.
#[derive(new, Debug, Clone)]
pub struct ScalarBinding {
    /// Type of the scalars
    pub ty: StorageType,
    /// Unpadded length of the underlying data
    pub length: usize,
    /// Type-erased data of the scalars. Padded and represented by u64 to prevent misalignment.
    pub data: Vec<u64>,
}

impl ScalarBinding {
    /// Get data as byte slice
    pub fn data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug, Clone)]
pub struct CopyDescriptor {
    /// Binding for the memory resource
    pub handle: Handle,
    /// Shape of the resource
    pub shape: Shape,
    /// Strides of the resource
    pub strides: Strides,
    /// Size of each element in the resource
    pub elem_size: usize,
}

/// A tensor map used with TMA ops
#[derive(new, Debug, Clone)]
pub struct TensorMapBinding {
    /// The binding for the backing tensor
    pub binding: Handle,
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
    /// Storage type
    pub storage_ty: StorageType,
}

impl Handle {
    /// If the tensor handle can be reused inplace.
    pub fn can_mut(&self) -> bool {
        self.id.can_mut() && self.stream == StreamId::current()
    }
}

impl Handle {
    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn copy_descriptor(
        self,
        shape: Shape,
        strides: Strides,
        elem_size: usize,
    ) -> CopyDescriptor {
        CopyDescriptor {
            shape,
            strides,
            elem_size,
            handle: self,
        }
    }
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            size: self.size,
            stream: self.stream,
        }
    }
}

/// Specifieds the number of cubes to be dispatched for a kernel.
///
/// This translates to eg. a grid for CUDA, or to `num_workgroups` for wgsl.
#[allow(clippy::large_enum_variant)]
pub enum CubeCount {
    /// Dispatch a known count of x, y, z cubes.
    Static(u32, u32, u32),
    /// Dispatch an amount based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Handle),
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
            Self::Dynamic(handle) => Self::Dynamic(handle.clone()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
/// The number of units across all 3 axis totalling to the number of working units in a cube.
pub struct CubeDim {
    /// The number of units in the x axis.
    pub x: u32,
    /// The number of units in the y axis.
    pub y: u32,
    /// The number of units in the z axis.
    pub z: u32,
}

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

        Self::new_2d(plane_size, u32::min(limit, plane_count))
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
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new cube dim with the given x, y and z.
    /// This is equivalent to the [new](CubeDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `CubeDim` can fully contain `other`
    pub const fn can_contain(&self, other: CubeDim) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
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

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Unchecked kernels are unsafe.
    Unchecked,
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
}
