/// Whose memory a relocation moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationScope {
    /// The memory of the stream the relocation runs for.
    Stream,
    /// The memory of every stream of the device.
    Device,
}
