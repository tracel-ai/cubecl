/// Work required by a problem, specified in minimum compute operations and byte transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Work {
    /// Compute operations required.
    pub compute_ops: usize,
    /// Memory bytes transferred (reads and writes).
    pub bytes: usize,
}
