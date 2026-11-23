use crate::memory_management::ManagedMemoryBinding;

/// A [`StorageAllocator`](super::StorageAllocator)'s resource for one managed
/// allocation, which keeps that allocation in place while it lives.
///
/// The resource may reach further than the allocation, such as a whole buffer
/// of which the allocation is one slice. Only the allocation's own part stays
/// put: the rest of the resource keeps serving other allocations.
#[derive(new, Debug)]
pub struct ManagedResource<Resource: Send> {
    // Keeps the allocation alive and in place while the resource is held: a
    // bound allocation is never freed, handed out again or relocated, so the
    // raw address the resource carries stays valid.
    #[allow(unused)]
    binding: ManagedMemoryBinding,
    resource: Resource,
}

impl<Resource: Send> ManagedResource<Resource> {
    /// The resource, to read the allocation through.
    pub fn resource(&self) -> &Resource {
        &self.resource
    }

    /// The resource, to write the allocation through.
    pub fn resource_mut(&mut self) -> &mut Resource {
        &mut self.resource
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;

    #[test]
    fn a_managed_resource_keeps_its_allocation_reserved() {
        let handle = ManagedMemoryHandle::new();

        let _resource = ManagedResource::new(handle.clone().binding(), ());

        assert!(!handle.is_free());
    }

    #[test]
    fn the_resource_is_written_in_place() {
        let mut resource = ManagedResource::new(ManagedMemoryHandle::new().binding(), 1);

        *resource.resource_mut() = 2;

        assert_eq!(*resource.resource(), 2);
    }
}
