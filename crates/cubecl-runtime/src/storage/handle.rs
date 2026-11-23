use crate::storage_id_type;

// This ID is used to map a handle to its actual data.
storage_id_type!(StorageId);

impl core::fmt::Display for StorageId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StorageId({})", self.value))
    }
}

/// The part of a storage allocation a [`StorageHandle`] covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageUtilization {
    /// Where the part starts, from the start of the allocation.
    pub offset_bytes: u64,
    /// How long the part is.
    pub size_bytes: u64,
}

/// A part of one allocation of a [`StorageAllocator`](super::StorageAllocator).
#[derive(new, Clone, Debug)]
pub struct StorageHandle {
    /// The allocation the part belongs to.
    pub id: StorageId,
    /// Which part of the allocation.
    pub utilization: StorageUtilization,
}

impl StorageHandle {
    /// How long the part is, in bytes.
    pub fn size(&self) -> u64 {
        self.utilization.size_bytes
    }

    /// Where the part starts in its allocation, in bytes.
    pub fn offset(&self) -> u64 {
        self.utilization.offset_bytes
    }

    /// The same allocation without its first `bytes` bytes.
    pub fn trim_start_bytes(&self, bytes: u64) -> Self {
        Self {
            id: self.id,
            utilization: StorageUtilization {
                offset_bytes: self.offset() + bytes,
                size_bytes: self.size() - bytes,
            },
        }
    }

    /// The same allocation without its last `bytes` bytes.
    pub fn trim_end_bytes(&self, bytes: u64) -> Self {
        Self {
            id: self.id,
            utilization: StorageUtilization {
                offset_bytes: self.offset(),
                size_bytes: self.size() - bytes,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(offset_bytes: u64, size_bytes: u64) -> StorageHandle {
        StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes,
                size_bytes,
            },
        )
    }

    #[test]
    fn size_reads_the_utilization() {
        assert_eq!(handle(8, 24).size(), 24);
    }

    #[test]
    fn offset_reads_the_utilization() {
        assert_eq!(handle(8, 24).offset(), 8);
    }

    #[test]
    fn trimming_the_start_moves_the_offset_and_shrinks_the_size() {
        assert_eq!(
            handle(8, 24).trim_start_bytes(4).utilization,
            handle(12, 20).utilization
        );
    }

    #[test]
    fn trimming_the_end_keeps_the_offset() {
        assert_eq!(
            handle(8, 24).trim_end_bytes(4).utilization,
            handle(8, 20).utilization
        );
    }

    #[test]
    fn two_storage_ids_differ() {
        assert_ne!(StorageId::new(), StorageId::new());
    }
}
