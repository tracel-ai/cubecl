use crate::storage_id_type;

// This ID is used to map a handle to its actual data.
storage_id_type!(StorageId);

impl core::fmt::Display for StorageId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StorageId({})", self.value))
    }
}

/// Defines if data uses a full memory chunk or a slice of it.
#[derive(Clone, Debug, PartialEq, Eq)]
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
