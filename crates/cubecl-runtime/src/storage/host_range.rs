use crate::server::IoError;
use alloc::format;
use alloc::sync::Arc;
use core::ops::Range;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::{AllocationProperty, Bytes};

/// The alignment of host memory a device imports and of every range imported from it: the largest
/// page a runtime imports host memory at (Metal's 16 KiB, Vulkan's
/// `minImportedHostPointerAlignment`), rounded up.
pub const HOST_MEMORY_ALIGNMENT: usize = 64 * 1024;

/// A range of host memory that a device may import
/// ([`Client::import_host_memory`](crate::client::Client::import_host_memory)) and its kernels read
/// in place, with no copy to the device.
///
/// The memory is shared behind an [`Arc`] and so read-only: a kernel reads it while the host could
/// otherwise write it, and nothing orders the two. Several devices may import the same memory; a
/// runtime pins it once for all of them.
///
/// # Guarantees
///
/// The memory is resident, allocated by the host's allocator, and starts at
/// [`HOST_MEMORY_ALIGNMENT`]; the range is non-empty, inside it, and aligned to
/// [`HOST_MEMORY_ALIGNMENT`] at both ends.
///
/// # Examples
///
/// ```
/// use cubecl_runtime::storage::{HOST_MEMORY_ALIGNMENT, HostRange};
/// use cubecl_environment::bytes::Bytes;
/// use std::sync::Arc;
///
/// let mut bytes = Bytes::zeroed_aligned(2 * HOST_MEMORY_ALIGNMENT, HOST_MEMORY_ALIGNMENT).unwrap();
/// bytes[0] = 7;
/// let chunk = HOST_MEMORY_ALIGNMENT as u64;
/// let range = HostRange::new(Arc::new(bytes), chunk..2 * chunk).unwrap();
/// assert_eq!(range.len(), chunk);
/// ```
#[derive(Clone, Debug)]
pub struct HostRange {
    memory: Arc<Bytes>,
    range: Range<u64>,
}

impl HostRange {
    /// `range` of `memory`, such as [`Bytes::zeroed_aligned`] at [`HOST_MEMORY_ALIGNMENT`] allocates.
    ///
    /// # Errors
    ///
    /// [`IoError::HostMemoryRefused`] unless the memory and the range are as the type guarantees.
    pub fn new(memory: Arc<Bytes>, range: Range<u64>) -> Result<Self, IoError> {
        let refused = |reason: &str| IoError::HostMemoryRefused {
            reason: format!("range {range:?} of {} bytes: {reason}", memory.len()).into(),
            backtrace: BackTrace::capture(),
        };
        // Any other memory may move, be read lazily, or already be pinned by a runtime.
        if memory.property() != AllocationProperty::Native {
            return Err(refused(
                "the memory must be allocated by the host's allocator",
            ));
        }
        // The device reads it while the import lives, so no other owner may write it meanwhile.
        if memory.may_alias_mutable_storage() {
            return Err(refused("no other owner may write the memory"));
        }
        let alignment = HOST_MEMORY_ALIGNMENT as u64;
        if !(memory.as_ptr() as u64).is_multiple_of(alignment) {
            return Err(refused("the memory must start aligned to 64 KiB"));
        }
        let inside = !range.is_empty() && range.end <= memory.len() as u64;
        let aligned = range.start.is_multiple_of(alignment) && range.end.is_multiple_of(alignment);
        if !(inside && aligned) {
            return Err(refused(
                "it must be non-empty, inside and aligned to 64 KiB",
            ));
        }
        Ok(Self { memory, range })
    }

    /// The memory the range is of, whole: what a runtime pins.
    pub fn memory(&self) -> &Arc<Bytes> {
        &self.memory
    }

    /// The range's first byte, from the start of the memory.
    pub fn offset(&self) -> u64 {
        self.range.start
    }

    /// The range's bytes.
    pub fn len(&self) -> u64 {
        self.range.end - self.range.start
    }

    /// Always false: an empty range is refused.
    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::string::{String, ToString};
    use core::mem::MaybeUninit;
    use cubecl_environment::bytes::{AccessError, AccessPolicy, AllocationController};

    const CHUNK: u64 = HOST_MEMORY_ALIGNMENT as u64;

    fn aligned(chunks: usize) -> Arc<Bytes> {
        let size = chunks * HOST_MEMORY_ALIGNMENT;
        Arc::new(Bytes::zeroed_aligned(size, HOST_MEMORY_ALIGNMENT).unwrap())
    }

    #[test]
    fn an_aligned_range_inside_is_importable() {
        let range = HostRange::new(aligned(2), CHUNK..2 * CHUNK).unwrap();
        assert_eq!((range.offset(), range.len()), (CHUNK, CHUNK));
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        assert!(HostRange::new(aligned(1), 0..2 * CHUNK).is_err());
    }

    #[test]
    fn an_unaligned_range_is_refused() {
        assert!(HostRange::new(aligned(1), 0..1).is_err());
    }

    #[test]
    fn an_empty_range_is_refused() {
        assert!(HostRange::new(aligned(1), 0..0).is_err());
    }

    #[test]
    fn memory_off_the_alignment_is_refused() {
        let memory = Bytes::zeroed_aligned(2 * HOST_MEMORY_ALIGNMENT, HOST_MEMORY_ALIGNMENT)
            .unwrap()
            .shared();
        let unaligned = memory.view(1, HOST_MEMORY_ALIGNMENT + 1).unwrap();
        let reason = refusal(unaligned);
        assert!(reason.contains("start aligned"), "{reason}");
    }

    /// Memory this crate did not allocate, holding no bytes.
    struct Foreign {
        property: AllocationProperty,
        aliases: bool,
    }

    impl AllocationController for Foreign {
        fn alloc_align(&self) -> usize {
            1
        }

        fn property(&self) -> AllocationProperty {
            self.property
        }

        fn may_alias_mutable_storage(&self) -> bool {
            self.aliases
        }

        unsafe fn memory_mut(
            &mut self,
            _policy: AccessPolicy,
        ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
            Ok(&mut [])
        }

        fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
            Ok(&[])
        }
    }

    fn foreign(property: AllocationProperty, aliases: bool) -> Bytes {
        let controller = Foreign { property, aliases };
        // SAFETY: the controller holds no bytes.
        unsafe { Bytes::from_controller(Box::new(controller), 0) }
    }

    fn refusal(memory: Bytes) -> String {
        match HostRange::new(Arc::new(memory), 0..CHUNK) {
            Err(IoError::HostMemoryRefused { reason, .. }) => reason.to_string(),
            other => panic!("the memory should be refused, got {other:?}"),
        }
    }

    #[test]
    fn memory_not_allocated_by_the_host_is_refused() {
        let reason = refusal(foreign(AllocationProperty::File, false));
        assert!(reason.contains("allocated by the host"), "{reason}");
    }

    #[test]
    fn memory_another_owner_writes_is_refused() {
        let reason = refusal(foreign(AllocationProperty::Native, true));
        assert!(reason.contains("no other owner"), "{reason}");
    }
}
