use crate::{
    memory_management::{MemoryHandle, MemoryManagement, MemoryUsage},
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
    /// Options when dispatching the kernel, eg. the number of executions.
    type DispatchOptions: Send;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;
    /// The [memory management](MemoryManagement) type defines strategies for allocation in the [storage](ComputeStorage) type.
    type MemoryManagement: MemoryManagement<Self::Storage>;
    /// The type of the features supported by the server.
    type Feature: Ord + Copy + Debug + Send + Sync;

    /// Given a handle, returns the owned resource as bytes.
    fn read(&mut self, binding: Binding<Self>) -> Reader;

    /// Given a resource handle, returns the storage resource.
    fn get_resource(&mut self, binding: Binding<Self>) -> BindingResource<Self>;

    /// Given a resource as bytes, stores it and returns the memory handle.
    fn create(&mut self, data: &[u8]) -> Handle<Self>;

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    fn empty(&mut self, size: usize) -> Handle<Self>;

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
        count: Self::DispatchOptions,
        bindings: Vec<Binding<Self>>,
        kind: ExecutionMode,
    );

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, command: SyncType);

    /// The current memory usage of the server.
    fn memory_usage(&self) -> MemoryUsage;
}

/// Server handle containing the [memory handle](MemoryManagement::Handle).
#[derive(new, Debug)]
pub struct Handle<Server: ComputeServer> {
    /// Memory handle.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::Handle,
    /// Memory offset in bytes.
    pub offset_start: Option<usize>,
    /// Memory offset in bytes.
    pub offset_end: Option<usize>,
}

impl<Server: ComputeServer> Handle<Server> {
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
pub struct Binding<Server: ComputeServer> {
    /// Memory binding.
    pub memory: <Server::MemoryManagement as MemoryManagement<Server::Storage>>::Binding,
    /// Memory offset in bytes.
    pub offset_start: Option<usize>,
    /// Memory offset in bytes.
    pub offset_end: Option<usize>,
}

impl<Server: ComputeServer> Handle<Server> {
    /// If the tensor handle can be reused inplace.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }
}

impl<Server: ComputeServer> Handle<Server> {
    /// Convert the [handle](Handle) into a [binding](Binding).
    pub fn binding(self) -> Binding<Server> {
        Binding {
            memory: MemoryHandle::binding(self.memory),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
        }
    }
}

impl<Server: ComputeServer> Clone for Handle<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
        }
    }
}

impl<Server: ComputeServer> Clone for Binding<Server> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
        }
    }
}
