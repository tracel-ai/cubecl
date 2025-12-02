use crate::{
    DeviceProperties,
    compiler::CompilationError,
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryHandle, MemoryUsage,
        memory_pool::{SliceBinding, SliceHandle},
    },
    storage::{BindingResource, ComputeStorage},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use cubecl_common::{
    ExecutionMode, backtrace::BackTrace, bytes::Bytes, device, future::DynFut,
    profile::ProfileDuration, stream_id::StreamId,
};
use cubecl_ir::StorageType;
use thiserror::Error;

#[derive(Error, Clone)]
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
        backtrace: BackTrace,
    },

    /// No profiling was registered
    #[error("No profiling registered\nBacktrace:\n{backtrace}")]
    NotRegistered {
        /// The captured backtrace.
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Execution(#[from] ExecutionError),
}

impl core::fmt::Debug for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

#[derive(Debug)]
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
    /// Information specific to the current server.
    pub info: Server::Info,
    /// The logger based on global cubecl configs.
    pub logger: Arc<ServerLogger>,
}

impl<S: ComputeServer> ServerUtilities<S> {
    /// Creates a new server utilities.
    pub fn new(properties: DeviceProperties, logger: Arc<ServerLogger>, info: S::Info) -> Self {
        // Start a tracy client if needed.
        #[cfg(feature = "profile-tracy")]
        let client = tracy_client::Client::start();

        Self {
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
        }
    }
}

/// Error that can happen when calling [ComputeServer::execute];
///
/// # Notes
///
/// Not all errors are going to be catched when calling [ComputeServer::execute] only the one that
/// won't block the compute queue.
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

impl core::fmt::Debug for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Error that can happen asynchronously while executing registered kernels.
#[derive(Error, Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ExecutionError {
    /// A generic runtime error.
    #[error("An error happened during execution\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    Generic {
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
/// [compute channel](crate::channel::ComputeChannel) for thread safety.
pub trait ComputeServer:
    Send + core::fmt::Debug + ServerCommunication + device::DeviceState + 'static
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: KernelMetadata;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError>;

    /// Reserves N [Bytes] of the provided sizes to be used as staging to load data.
    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        })
    }

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Retrieve the server utilities.
    fn utilities(&self) -> Arc<ServerUtilities<Self>>;

    /// Utility to create a new buffer and immediately copy contiguous data into it
    fn create_with_data(&mut self, data: &[u8], stream_id: StreamId) -> Result<Handle, IoError> {
        let alloc = self
            .create(
                vec![AllocationDescriptor::new(
                    AllocationKind::Contiguous,
                    &[data.len()],
                    1,
                )],
                stream_id,
            )?
            .remove(0);
        self.write(
            vec![(
                CopyDescriptor::new(
                    alloc.handle.clone().binding(),
                    &[data.len()],
                    &alloc.strides,
                    1,
                ),
                Bytes::from_bytes_vec(data.to_vec()),
            )],
            stream_id,
        )?;
        Ok(alloc.handle)
    }

    /// Utility to create a new buffer and immediately copy contiguous data into it
    fn create_with_bytes(&mut self, data: Bytes, stream_id: StreamId) -> Result<Handle, IoError> {
        let alloc = self
            .create(
                vec![AllocationDescriptor::new(
                    AllocationKind::Contiguous,
                    &[data.len()],
                    1,
                )],
                stream_id,
            )?
            .remove(0);
        self.write(
            vec![(
                CopyDescriptor::new(
                    alloc.handle.clone().binding(),
                    &[data.len()],
                    &alloc.strides,
                    1,
                ),
                data,
            )],
            stream_id,
        )?;
        Ok(alloc.handle)
    }

    /// Given bindings, returns the owned resources as bytes.
    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>>;

    /// Writes the specified bytes into the buffers given
    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError>;

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>>;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// # Safety
    ///
    /// When executing with mode [ExecutionMode::Unchecked], out-of-bound reads and writes can happen.
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError>;

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self, stream_id: StreamId);

    /// The current memory usage of the server.
    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self, stream_id: StreamId);

    /// Enable collecting timestamps.
    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken;

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
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
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

/// Server handle containing the [memory handle](crate::server::Handle).
#[derive(new, Debug, PartialEq, Eq)]
pub struct Handle {
    /// Memory handle.
    pub memory: SliceHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: cubecl_common::stream_id::StreamId,
    /// The stream position when the tensor became available.
    pub cursor: u64,
    /// Length of the underlying buffer ignoring offsets
    size: u64,
}

/// Type of allocation, either contiguous or optimized (row-aligned when possible)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AllocationKind {
    /// Contiguous layout, with no padding
    Contiguous,
    /// Optimized for access speed. In practice this means row-aligned with padding for runtimes
    /// that support it.
    Optimized,
}

/// Descriptor for a new tensor allocation
#[derive(new, Debug, Clone, Copy)]
pub struct AllocationDescriptor<'a> {
    /// Layout for the tensor
    pub kind: AllocationKind,
    /// Shape of the tensor
    pub shape: &'a [usize],
    /// Size of each element in the tensor (used for conversion of shape to bytes)
    pub elem_size: usize,
}

impl<'a> AllocationDescriptor<'a> {
    /// Create an optimized allocation descriptor
    pub fn optimized(shape: &'a [usize], elem_size: usize) -> Self {
        AllocationDescriptor::new(AllocationKind::Optimized, shape, elem_size)
    }

    /// Create a contiguous allocation descriptor
    pub fn contiguous(shape: &'a [usize], elem_size: usize) -> Self {
        AllocationDescriptor::new(AllocationKind::Contiguous, shape, elem_size)
    }
}

/// An allocation with associated strides. Strides depend on tensor layout.
#[derive(new, Debug)]
pub struct Allocation {
    /// The handle for the memory resource
    pub handle: Handle,
    /// The strides of the tensor
    pub strides: Vec<usize>,
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
    #[error("Can't perform the IO operation because of a runtime error")]
    Execution(#[from] ExecutionError),
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

    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
}

/// Bindings to execute a kernel.
#[derive(Debug, Default)]
pub struct Bindings {
    /// Buffer bindings
    pub buffers: Vec<Binding>,
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
    pub fn with_buffer(mut self, binding: Binding) -> Self {
        self.buffers.push(binding);
        self
    }

    /// Extend the buffers with `bindings`
    pub fn with_buffers(mut self, bindings: Vec<Binding>) -> Self {
        self.buffers.extend(bindings);
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
    pub data: Vec<u32>,
    /// Length of the static portion (rank, len, buffer_len, shape_offsets, stride_offsets).
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

/// Binding of a [tensor handle](Handle) to execute a kernel.
#[derive(new, Debug)]
pub struct Binding {
    /// Memory binding.
    pub memory: SliceBinding,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: cubecl_common::stream_id::StreamId,
    /// The stream position when the tensor became available.
    pub cursor: u64,
    /// Size in bytes
    size: u64,
}

impl Binding {
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug, Clone)]
pub struct CopyDescriptor<'a> {
    /// Binding for the memory resource
    pub binding: Binding,
    /// Shape of the resource
    pub shape: &'a [usize],
    /// Strides of the resource
    pub strides: &'a [usize],
    /// Size of each element in the resource
    pub elem_size: usize,
}

/// A tensor map used with TMA ops
#[derive(new, Debug, Clone)]
pub struct TensorMapBinding {
    /// The binding for the backing tensor
    pub binding: Binding,
    /// The tensormap metadata
    pub map: TensorMapMeta,
}

/// TensorMap metadata for the opaque proxy used in TMA copies
#[derive(Debug, Clone)]
pub struct TensorMapMeta {
    /// Tensormap format (tiled or im2col)
    pub format: TensorMapFormat,
    /// Rank of the backing tensor
    pub rank: usize,
    /// Shape of the backing tensor
    pub shape: Vec<usize>,
    /// Strides of the backing tensor
    pub strides: Vec<usize>,
    /// Element stride, usually 1 but may be 2 for complex tensors
    /// For im2col, this is equivalent to the kernel stride
    pub elem_stride: Vec<usize>,
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
        self.memory.can_mut() && self.stream == StreamId::current()
    }
}

impl Handle {
    /// Convert the [handle](Handle) into a [binding](Binding).
    pub fn binding(self) -> Binding {
        Binding {
            memory: MemoryHandle::binding(self.memory),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            size: self.size,
            stream: self.stream,
            cursor: self.cursor,
        }
    }

    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn copy_descriptor<'a>(
        &'a self,
        shape: &'a [usize],
        strides: &'a [usize],
        elem_size: usize,
    ) -> CopyDescriptor<'a> {
        CopyDescriptor {
            shape,
            strides,
            elem_size,
            binding: self.clone().binding(),
        }
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            size: self.size,
            stream: self.stream,
            cursor: self.cursor,
        }
    }
}

impl Clone for Binding {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            size: self.size,
            stream: self.stream,
            cursor: self.cursor,
        }
    }
}

/// Specifieds the number of cubes to be dispatched for a kernel.
///
/// This translates to eg. a grid for CUDA, or to num_workgroups for wgsl.
#[allow(clippy::large_enum_variant)]
pub enum CubeCount {
    /// Dispatch a known count of x, y, z cubes.
    Static(u32, u32, u32),
    /// Dispatch an amount based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding),
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
