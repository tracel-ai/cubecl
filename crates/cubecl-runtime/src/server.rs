use crate::{
    memory_management::{
        memory_pool::{SliceBinding, SliceHandle},
        MemoryHandle, MemoryUsage,
    },
    storage::{BindingResource, ComputeStorage},
};
use alloc::vec::Vec;
use core::{fmt::Debug, future::Future};
use cubecl_common::{
    benchmark::TimestampsResult, ExecutionMode, OobFill, TensorMapFormat, TensorMapInterleave,
    TensorMapPrefetch, TensorMapSwizzle,
};
use cubecl_ir::Elem;

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [compute channel](crate::channel::ComputeChannel) for thread safety.
pub trait ComputeServer: Send + core::fmt::Debug
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: Send;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The type of the features supported by the server.
    type Feature: Ord + Copy + Debug + Send + Sync;

    /// Given bindings, returns the owned resources as bytes.
    fn read(
        &mut self,
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource>;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> Handle;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> Handle;

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
        constants: Vec<ConstBinding>,
        bindings: Vec<Binding>,
        kind: ExecutionMode,
    );

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self) -> impl Future<Output = ()> + Send + 'static;

    /// Wait for the completion of every task in the server.
    ///
    /// Returns the (approximate) total amount of GPU work done since the last sync.
    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + Send + 'static;

    /// The current memory usage of the server.
    fn memory_usage(&self) -> MemoryUsage;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self);

    /// Enable collecting timestamps.
    fn enable_timestamps(&mut self);

    /// Disable collecting timestamps.
    fn disable_timestamps(&mut self);
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

/// Binding of a grid constant to execute a kernel.
#[derive(new, Debug, Clone)]
pub enum ConstBinding {
    /// A tensor map used for TMA loading ops
    TensorMap {
        /// The binding for the backing tensor
        binding: Binding,
        /// The tensormap metadata
        map: TensorMap,
    },
}

/// TensorMap metadata for the opaque proxy used in TMA copies
#[derive(Debug, Clone)]
pub struct TensorMap {
    /// Tensormap format (tiled or im2col)
    pub format: TensorMapFormat,
    /// Rank of the backing tensor
    pub rank: usize,
    /// Shape of the backing tensor
    pub shape: Vec<u64>,
    /// Strides of the backing tensor
    pub strides: Vec<u64>,
    /// Element stride, usually 1 but may be 2 for complex tensors
    pub elem_stride: Vec<u32>,
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
