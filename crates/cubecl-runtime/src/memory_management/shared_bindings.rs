use super::ManagedMemoryBinding;
use alloc::vec::Vec;

/// Bindings held only to keep their memory reserved while streams that share
/// it may still read it, released together.
#[derive(Debug, Default)]
pub struct SharedMemoryBindings {
    /// The bindings held, only ever added at the end and released all at
    /// once: nothing reads them one by one.
    bindings: Vec<ManagedMemoryBinding>,
}

impl SharedMemoryBindings {
    /// Releases every binding held.
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Whether no binding is held.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Holds `binding` until the next [`clear`](Self::clear).
    pub fn push(&mut self, binding: ManagedMemoryBinding) {
        self.bindings.push(binding)
    }

    /// Takes over every binding `other` holds, leaving it empty.
    pub fn append(&mut self, other: &mut Self) {
        self.bindings.append(&mut other.bindings)
    }
}

impl cubecl_common::pool::Reclaim for SharedMemoryBindings {
    fn reclaim(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;

    fn holding_one() -> SharedMemoryBindings {
        let mut shared = SharedMemoryBindings::default();
        shared.push(ManagedMemoryHandle::new().binding());
        shared
    }

    #[test]
    fn new_bindings_are_empty() {
        assert!(SharedMemoryBindings::default().is_empty());
    }

    #[test]
    fn a_pushed_binding_is_held() {
        let mut shared = SharedMemoryBindings::default();

        shared.push(ManagedMemoryHandle::new().binding());

        assert!(!shared.is_empty());
    }

    #[test]
    fn a_pushed_binding_keeps_its_memory_reserved() {
        let handle = ManagedMemoryHandle::new();
        let mut shared = SharedMemoryBindings::default();

        shared.push(handle.clone().binding());

        assert!(!handle.is_free());
    }

    #[test]
    fn clear_releases_the_memory() {
        let handle = ManagedMemoryHandle::new();
        let mut shared = SharedMemoryBindings::default();
        shared.push(handle.clone().binding());

        shared.clear();

        assert!(handle.is_free());
    }

    #[test]
    fn append_empties_the_other() {
        let (mut shared, mut other) = (SharedMemoryBindings::default(), holding_one());

        shared.append(&mut other);

        assert!(other.is_empty() && !shared.is_empty());
    }

    #[test]
    fn reclaim_releases_every_binding() {
        let mut shared = holding_one();

        cubecl_common::pool::Reclaim::reclaim(&mut shared);

        assert!(shared.is_empty());
    }
}
