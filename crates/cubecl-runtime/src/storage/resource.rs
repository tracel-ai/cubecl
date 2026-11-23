use crate::memory_management::ManagedMemoryBinding;

/// Access to the underlying resource.
#[derive(new, Debug)]
pub struct ManagedResource<Resource: Send> {
    /// Keeps the allocation alive and in place while the resource is held:
    /// a bound allocation is never freed, handed out again or relocated, so
    /// the raw address the resource carries stays valid. The rest of its page
    /// keeps serving other allocations.
    #[allow(unused)]
    binding: ManagedMemoryBinding,
    resource: Resource,
}

impl<Resource: Send> ManagedResource<Resource> {
    /// access the underlying resource.
    ///
    /// # Note
    ///
    /// The resource might be bigger than the part required.
    /// (e.g. a big buffer where the handle only refers to a slice of it).
    /// Only the part required by the handle is guaranteed to remain,
    /// other parts of this resource *will* be re-used.
    pub fn resource(&self) -> &Resource {
        &self.resource
    }

    /// Mutable access to the underlying resource.
    pub fn resource_mut(&mut self) -> &mut Resource {
        &mut self.resource
    }
}
