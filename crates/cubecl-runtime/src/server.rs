use crate::{
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{
        MemoryHandle, MemoryUsage,
        memory_pool::{SliceBinding, SliceHandle},
    },
    storage::{BindingResource, ComputeStorage},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Debug;
use cubecl_common::{ExecutionMode, benchmark::ProfileDuration, future::DynFut};
use cubecl_ir::Elem;

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

    /// Given bindings, returns the owned resources as bytes.
    fn read(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>>;

    /// Given tensor handles, returns the owned resources as bytes.
    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>>;

    /// Wait for the completion of every task in the server.
    fn sync(&mut self) -> DynFut<()>;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource>;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> Handle;

    /// Given a resource as bytes with `shape`, stores it and returns the tensor handle.
    /// May or may not be contiguous, depending on what's best for the given runtime. Always use
    /// strides to index.
    /// For example, in CUDA, this will allocate a padded tensor where the last dimension is padded
    /// to the cache lines, so row access is faster.
    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)>;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> Handle;

    /// Reserves `shape` bytes in the storage, and returns a handle to it.
    fn empty_tensors(
        &mut self,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)>;

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
    pub scalars: BTreeMap<Elem, ScalarBinding>,
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
    pub fn with_scalar(mut self, elem: Elem, length: usize, data: Vec<u64>) -> Self {
        self.scalars
            .insert(elem, ScalarBinding::new(elem, length, data));
        self
    }

    /// Extend the scalars with `bindings`
    pub fn with_scalars(mut self, bindings: Vec<ScalarBinding>) -> Self {
        self.scalars
            .extend(bindings.into_iter().map(|binding| (binding.elem, binding)));
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
#[derive(new, Debug)]
pub struct ScalarBinding {
    /// Type of the scalars
    pub elem: Elem,
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
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug)]
pub struct BindingWithMeta {
    /// Binding for the memory resource
    pub binding: Binding,
    /// Shape of the resource
    pub shape: Vec<usize>,
    /// Strides of the resource
    pub strides: Vec<usize>,
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
    /// Element type
    pub elem: Elem,
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
        }
    }

    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn binding_with_meta(
        self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    ) -> BindingWithMeta {
        BindingWithMeta {
            shape,
            strides,
            elem_size,
            binding: self.binding(),
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
