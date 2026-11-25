use super::{ManagedMemoryDescriptor, ManagedMemoryHandle};
use crate::{
    memory_management::{MemoryLocation, MemoryUsage, Taint},
    server::IoError,
    storage::StorageHandle,
};
use alloc::format;
use cubecl_environment::backtrace::BackTrace;

/// The page index a pool's first location carries.
const FIRST_PAGE_INDEX: u16 = 0;
/// The slice index a page's first location carries.
const FIRST_SLICE_INDEX: u32 = 0;

/// One stretch of a device allocation, and the allocation on it if any.
#[derive(Debug)]
pub(crate) struct Slice {
    /// The bytes the allocation on it uses, its padding excluded.
    pub storage: StorageHandle,
    /// The handle the allocation is reached through, free while no owner
    /// holds it.
    pub handle: ManagedMemoryHandle,
    /// The bytes past `storage` that alignment adds.
    pub padding_bytes: u64,
    /// The stream cursor the allocation was bound at, which a cross-stream
    /// wait compares against.
    pub cursor: u64,
    /// Whether `storage.id` is backed by a real device allocation. Pools whose
    /// slices own their whole buffer (the persistent and direct pools) track
    /// laziness here; paged pools track it on the page, whose id every slice
    /// shares.
    pub mapped: bool,
    /// The bytes of the allocation no writer filled.
    ///
    /// Whether these bytes can be trusted is a property of the memory, so it
    /// lives here rather than in any stream's bookkeeping. What a failure id
    /// means is the [`ErrorGraph`]'s to say; the slice only carries the
    /// claims, byte-ranged so a partial write releases exactly what it
    /// covers, and sheds them the moment the allocation it describes ends.
    pub tainted: Taint,
}

impl Slice {
    /// A free slice over `storage`, `padding_bytes` of alignment past it.
    pub fn new(storage: StorageHandle, padding_bytes: u64) -> Self {
        Self {
            storage,
            handle: ManagedMemoryHandle::new(),
            padding_bytes,
            cursor: 0,
            mapped: true,
            tainted: Taint::default(),
        }
    }

    /// Whether no owner holds the allocation, so the slice can serve another.
    pub fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    /// The bytes the slice takes on its page, its padding included.
    pub fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding_bytes
    }

    /// The descriptor every owner of the allocation resolves it through.
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        self.handle.descriptor()
    }
}

/// The first location of the pool `pool_index` names.
pub(crate) fn pool_origin(pool_index: u8) -> MemoryLocation {
    MemoryLocation::new(pool_index, FIRST_PAGE_INDEX, FIRST_SLICE_INDEX)
}

/// `size` bytes rounded up to `memory_alignment`, what an allocation of them
/// takes: `None` past the largest size a `u64` holds, which no device
/// allocates.
pub(crate) fn padded_size(size: u64, memory_alignment: u64) -> Option<u64> {
    size.checked_add(calculate_padding(size, memory_alignment))
}

/// The bytes alignment adds past `size`, for a buffer aligned to `memory_alignment`.
pub(crate) fn calculate_padding(size: u64, memory_alignment: u64) -> u64 {
    (memory_alignment - size % memory_alignment) % memory_alignment
}

/// The error a lookup of a `kind` that a pool does not hold at `index` answers.
pub(crate) fn missing(kind: &str, index: usize) -> IoError {
    IoError::NotFound {
        backtrace: BackTrace::capture(),
        reason: format!("Memory {kind} {index} doesn't exist").into(),
    }
}

/// What `slices` hold: the live ones' count, bytes and padding, and every
/// slice's padded size as the bytes reserved.
pub(crate) fn slices_usage<'a>(slices: impl Iterator<Item = &'a Slice>) -> MemoryUsage {
    slices.fold(MemoryUsage::default(), |usage, slice| {
        usage.combine(slice_usage(slice))
    })
}

/// What one slice holds: its padded size reserved, and its allocation when
/// one is live on it.
fn slice_usage(slice: &Slice) -> MemoryUsage {
    let reserved = MemoryUsage {
        reserved_bytes: slice.effective_size(),
        ..MemoryUsage::default()
    };
    if slice.is_free() {
        return reserved;
    }
    MemoryUsage {
        allocation_count: 1,
        in_use_bytes: slice.storage.size(),
        padding_bytes: slice.padding_bytes,
        ..reserved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageId;
    use crate::storage::StorageUtilization;

    fn slice(size: u64) -> Slice {
        Slice::new(
            StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: size,
                },
            ),
            0,
        )
    }

    #[test]
    fn padding_rounds_up_to_the_alignment() {
        assert_eq!(calculate_padding(100, 32), 28);
        assert_eq!(calculate_padding(128, 32), 0);
        assert_eq!(padded_size(100, 32), Some(128));
        assert_eq!(padded_size(u64::MAX, 32), None);
    }

    /// The usage of slices counts live ones only, and every slice's padded
    /// size as reserved.
    #[test]
    fn the_usage_counts_live_slices_and_every_reserved_byte() {
        let live = slice(64);
        let _owner = live.handle.clone();
        let free = Slice::new(slice(32).storage, 32);

        let usage = slices_usage([&live, &free].into_iter());

        assert_eq!(
            (
                usage.allocation_count,
                usage.in_use_bytes,
                usage.reserved_bytes
            ),
            (1, 64, 128)
        );
    }
}
