use super::{ManagedMemoryDescriptor, ManagedMemoryHandle};
use crate::{
    memory_management::{ErrorGraph, Taint},
    server::IoError,
    storage::{StorageAllocator, StorageHandle},
};
use cubecl_environment::backtrace::BackTrace;

#[derive(Debug)]
/// Slice of data with its associated storage.
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: ManagedMemoryHandle,
    pub padding: u64,
    pub cursor: u64,
    /// Whether `storage.id` is backed by a real device allocation. Pools whose
    /// slices own their whole buffer (the persistent and direct pools) track
    /// laziness here; sliced pools track it on the page, whose id every slice
    /// shares.
    pub mapped: bool,
    /// The tainted regions of this allocation, if any.
    ///
    /// Whether these bytes can be trusted is a property of the memory, so it
    /// lives here rather than in any stream's bookkeeping. What a failure id
    /// means is the [`ErrorGraph`]'s to say; the slice only carries the
    /// claims, byte-ranged so a partial write releases exactly what it
    /// covers, and sheds them the moment the allocation it describes ends —
    /// see [`bind`](Self::bind) and the clear calls on every path that drops
    /// a slice.
    pub tainted: Taint,
}

impl Slice {
    pub fn new(storage: StorageHandle, padding: u64) -> Self {
        Self {
            storage,
            handle: ManagedMemoryHandle::new(),
            padding,
            cursor: 0,
            mapped: true,
            tainted: Taint::default(),
        }
    }

    /// Take on a new allocation. Whatever tainted the last one says nothing
    /// about the next, so every claim is released before the handle changes
    /// hands.
    pub(crate) fn bind(&mut self, handle: ManagedMemoryHandle, failures: &mut ErrorGraph) {
        self.tainted.clear(failures);
        self.handle = handle;
    }

    /// Give this slice's allocation to `target`, which was reserved to
    /// receive it and already holds a copy of its bytes: the handle, its
    /// cursor and its taint move, the handle's location is rewritten to the
    /// target's so every owner resolves there from now on, and this slice is
    /// left free. Whatever the target's last allocation left tainted is
    /// released first, as [`bind`](Self::bind) would.
    pub(crate) fn hand_over(&mut self, target: &mut Slice, failures: &mut ErrorGraph) {
        target.tainted.clear(failures);
        let location = target.handle.descriptor().location();
        let source_location = self.handle.descriptor().location();

        let freed = ManagedMemoryHandle::new();
        freed.descriptor().update_location(source_location);
        let allocation = core::mem::replace(&mut self.handle, freed);
        allocation.descriptor().update_location(location);

        target.handle = allocation;
        target.cursor = self.cursor;
        target.tainted = core::mem::take(&mut self.tainted);
    }

    /// If the slice is free to be reused.
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    /// The total size of the slice including padding.
    pub(crate) fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    /// The description of the slice.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        self.handle.descriptor()
    }

    /// Install real device backing behind a slice that owns its whole buffer
    /// and was allocated [`PageMapping::Lazy`](super::PageMapping::Lazy): allocate for real and retire
    /// the minted id, which never reached the driver. The caller checks the
    /// slice is unmapped and that the binding has a claim on it.
    pub(crate) fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        let effective_size = self.effective_size();
        let real = storage.allocate_bytes(effective_size).map_err(|err| {
            IoError::StorageMappingFailed {
                size: effective_size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            }
        })?;
        self.storage.id = real.id;
        self.mapped = true;
        Ok(())
    }
}

/// `size` bytes rounded up to `memory_alignment`, what an allocation of them
/// takes: `None` past the largest size a `u64` holds, which no device
/// allocates.
pub(crate) fn padded_size(size: u64, memory_alignment: u64) -> Option<u64> {
    size.checked_add(calculate_padding(size, memory_alignment))
}

/// Calculates the padding required to store the given size in a buffer given the memory alignment.
pub(crate) fn calculate_padding(size: u64, memory_alignment: u64) -> u64 {
    let remainder = size % memory_alignment;
    if remainder != 0 {
        memory_alignment - remainder
    } else {
        0
    }
}
