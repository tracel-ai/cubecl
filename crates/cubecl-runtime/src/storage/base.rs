use core::fmt::Debug;

use crate::{
    server::{Binding, IoError},
    storage_id_type,
};

// This ID is used to map a handle to its actual data.
storage_id_type!(StorageId);

impl core::fmt::Display for StorageId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StorageId({})", self.value))
    }
}

/// Defines if data uses a full memory chunk or a slice of it.
#[derive(Clone, Debug)]
pub struct StorageUtilization {
    /// The offset in bytes from the chunk start.
    pub offset: u64,
    /// The size of the slice in bytes.
    pub size: u64,
}

/// Contains the [storage id](StorageId) of a resource and the way it is used.
#[derive(new, Clone, Debug)]
pub struct StorageHandle {
    /// Storage id.
    pub id: StorageId,
    /// How the storage is used.
    pub utilization: StorageUtilization,
}

impl StorageHandle {
    /// Returns the size the handle is pointing to in memory.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Returns the offset of the handle.
    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }

    /// Increase the current offset with the given value in bytes.
    pub fn offset_start(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset() + offset_bytes,
            size: self.size() - offset_bytes,
        };

        Self {
            id: self.id,
            utilization,
        }
    }

    /// Reduce the size of the memory handle..
    pub fn offset_end(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset(),
            size: self.size() - offset_bytes,
        };

        Self {
            id: self.id,
            utilization,
        }
    }
}

/// Storage types are responsible for allocating and deallocating memory.
pub trait ComputeStorage: Send {
    /// The resource associated type determines the way data is implemented and how
    /// it can be accessed by kernels.
    type Resource: Send;

    /// The alignment memory is allocated with in this storage.
    fn alignment(&self) -> usize;

    /// Returns the underlying resource for a specified storage handle
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Allocates `size` units of memory and returns a handle to it
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError>;

    /// Deallocates the memory pointed by the given storage id.
    ///
    /// These deallocations might need to be flushed with [`Self::flush`].
    fn dealloc(&mut self, id: StorageId);

    /// Flush deallocations when required.
    fn flush(&mut self) {}
}

/// Access to the underlying resource for a given binding.
#[derive(new, Debug)]
pub struct BindingResource<Resource: Send> {
    // This binding is here just to keep the underlying allocation alive.
    // If the underlying allocation becomes invalid, someone else might
    // allocate into this resource which could lead to bad behaviour.
    #[allow(unused)]
    binding: Binding,
    resource: Resource,
}

impl<Resource: Send> BindingResource<Resource> {
    /// access the underlying resource. Note: The resource might be bigger
    /// than the part required by the binding (e.g. a big buffer where the binding only
    /// refers to a slice of it). Only the part required by the binding is guaranteed to remain,
    /// other parts of this resource *will* be re-used.
    pub fn resource(&self) -> &Resource {
        &self.resource
    }
}
