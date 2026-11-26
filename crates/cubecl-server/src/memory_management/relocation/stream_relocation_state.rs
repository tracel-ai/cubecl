use super::RelocationNeed;

/// What a relocation reads of one stream's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRelocationState {
    /// Whether the memory wants a relocation, before the bytes the device
    /// holds are known (see
    /// [`MemoryManagement::relocation_need`](crate::memory_management::MemoryManagement::relocation_need)).
    pub need: RelocationNeed,
    /// The bytes the memory holds from the device.
    pub allocated_bytes: u64,
}
