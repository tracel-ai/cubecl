/// A byte count.
///
/// A thin newtype over `u64` so pool settings like `page_size` are
/// unambiguously in bytes at the type level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemorySize(pub u64);

impl MemorySize {
    /// The size in bytes.
    pub const fn bytes(self) -> u64 {
        self.0
    }
}
