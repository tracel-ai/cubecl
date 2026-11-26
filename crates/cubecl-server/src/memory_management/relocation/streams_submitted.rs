/// Proof handed to a [`StreamRelocator`](super::StreamRelocator), so it may
/// build a copier whose wait covers only what was submitted.
///
/// # Guarantees
///
/// Every stream of the device has submitted its queued work, and on a
/// runtime whose flush waits for its launches, they have run. Nothing but
/// the relocations' own copies is issued until the relocation holding it
/// returns.
#[derive(Debug)]
pub struct StreamsSubmitted {
    _minted: (),
}

impl StreamsSubmitted {
    /// # Safety
    ///
    /// Every stream of the device has just been flushed, submitting its
    /// queued work (and, on a runtime whose flush waits for its launches,
    /// running it), and nothing but the relocations' own copies is issued
    /// until whatever the proof is handed to returns.
    pub(crate) unsafe fn new() -> Self {
        Self { _minted: () }
    }
}
