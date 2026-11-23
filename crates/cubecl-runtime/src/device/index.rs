/// A device index as a [`DeviceId`](super::DeviceId) carries it.
///
/// Kept at the largest the id holds rather than wrapped: wrapped, an index
/// past the end lands on a device that exists; kept, it names one no machine
/// has, and naming it is refused.
pub fn index_id(index: usize) -> u16 {
    index.min(u16::MAX as usize) as u16
}
