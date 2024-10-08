use crate::{
    memory_management::{
        memory_pool::{SliceBinding, SliceHandle},
        MemoryHandle, MemoryUsage,
    },
    storage::{BindingResource, ComputeStorage},
    ExecutionMode,
};
use alloc::vec::Vec;
use core::fmt::Debug;
use cubecl_common::{reader::Reader, sync_type::SyncType};

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
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The type of the features supported by the server.
    type Feature: Ord + Copy + Debug + Send + Sync;

    /// Given a handle, returns the owned resource as bytes.
    fn read(&mut self, binding: Binding) -> Reader;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(&mut self, binding: Binding) -> BindingResource<Self>;

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
        bindings: Vec<Binding>,
        kind: ExecutionMode,
    );

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, command: SyncType);

    /// The current memory usage of the server.
    fn memory_usage(&self) -> MemoryUsage;
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new, Debug)]
pub struct Handle {
    /// Memory handle.
    pub memory: SliceHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<usize>,
    /// Memory offset in bytes.
    pub offset_end: Option<usize>,
}

impl Handle {
    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: usize) -> Self {
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: usize) -> Self {
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }
}

/// Binding of a [tensor handle](Handle) to execute a kernel.
#[derive(new, Debug)]
pub struct Binding {
    /// Memory binding.
    pub memory: SliceBinding,
    /// Memory offset in bytes.
    pub offset_start: Option<usize>,
    /// Memory offset in bytes.
    pub offset_end: Option<usize>,
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

/// Provides launch information specifying the number of work groups to be used by a compute shader.
pub enum CubeCount {
    /// Dispatch x,y,z work groups.
    Static(u32, u32, u32),
    /// Dispatch work groups based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding),
}

impl Debug for CubeCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
