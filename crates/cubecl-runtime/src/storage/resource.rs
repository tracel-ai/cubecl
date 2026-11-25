use crate::memory_management::ManagedMemoryBinding;

/// A [`StorageAllocator`](super::StorageAllocator)'s resource for one managed
/// allocation, which keeps that allocation alive while it lives.
///
/// The resource may reach further than the allocation, such as a whole buffer
/// of which the allocation is one slice. Only the allocation's own part is
/// kept: the rest of the resource keeps serving other allocations.
///
/// # Guarantees
///
/// - While the resource lives, its allocation is never freed or handed out
///   again.
/// - Its allocation stays in place, so the raw address the resource carries
///   stays valid, unless a graph capture keeps that allocation apart from
///   what its replays write: opening the capture's window may then move it,
///   and the address names its old place from then on.
#[derive(new, Debug)]
pub struct ManagedResource<Resource: Send> {
    // Keeps the allocation alive while the resource is held: a bound
    // allocation is never freed or handed out again.
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
