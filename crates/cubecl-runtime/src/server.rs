use crate::{
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
use cubecl_common::{ExecutionMode, future::DynFut, profile::ProfileDuration};
use cubecl_ir::StorageType;
use thiserror::Error;

#[derive(Debug, Clone)]
/// An error during profiling.
pub enum ProfileError {
    /// Unknown error.
    Unknown(String),
    /// When no profiling has been registered.
    NotRegistered,
}

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [compute channel](crate::channel::ComputeChannel) for thread safety.
pub trait ComputeServer: Send + core::fmt::Debug
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: KernelMetadata;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The type of the features supported by the server.
    type Feature: Ord + Copy + Debug + Send + Sync;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError>;

    /// Utility to create a new buffer and immediately copy contiguous data into it
    fn create_with_data(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        let alloc = self
            .create(vec![AllocationDescriptor::new(
                AllocationKind::Contiguous,
                &[data.len()],
                1,
            )])?
            .remove(0);
        self.write(vec![(
            CopyDescriptor::new(
                alloc.handle.clone().binding(),
                &[data.len()],
                &alloc.strides,
                1,
            ),
            data,
        )])?;
        Ok(alloc.handle)
    }

    /// Given bindings, returns the owned resources as bytes.
    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
    ) -> DynFut<Result<Vec<Vec<u8>>, IoError>>;

    /// Writes the specified bytes into the buffers given
    fn write(&mut self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError>;

    /// Wait for the completion of every task in the server.
    fn sync(&mut self) -> DynFut<()>;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// # Safety
    ///
    /// When executing with mode [ExecutionMode::Unchecked], out-of-bound reads and writes can happen.
    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        logger: Arc<ServerLogger>,
    );

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self);

    /// The current memory usage of the server.
    fn memory_usage(&self) -> MemoryUsage;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self);

    /// Enable collecting timestamps.
    fn start_profile(&mut self) -> ProfilingToken;

    /// Disable collecting timestamps.
    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError>;

    /// Update the memory mode of allocation in the server.
    fn allocation_mode(&mut self, mode: MemoryAllocationMode);
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Profiling identification so that the server can support recursive and overlapping profilings.
pub struct ProfilingToken {
    /// The token value.
    pub id: u64,
}

/// Server handle containing the [memory handle](crate::server::Handle).
#[derive(new, Debug)]
pub struct Handle {
    /// Memory handle.
    pub memory: SliceHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
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
#[derive(Debug, Error)]
pub enum IoError {
    /// Buffer size exceeds the max available
    #[error("can't allocate buffer of size")]
    BufferTooBig(usize),
    /// Strides aren't supported for this copy operation on this runtime
    #[error("the provided strides are not supported for this operation")]
    UnsupportedStrides,
    /// Handle wasn't found in the memory pool
    #[error("couldn't find resource for that handle")]
    InvalidHandle,
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
        self.memory.can_mut()
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
