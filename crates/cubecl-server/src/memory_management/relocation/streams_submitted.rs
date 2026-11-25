/// Proof that every stream of the device has submitted its queued work (and,
/// on a runtime that runs work as it is submitted, run it), and that none is
/// issued until the relocation holding it returns.
///
/// Only the crate mints it, right where it submits every stream's work, so a
/// [`StreamRelocator`](super::StreamRelocator) handed one may build a copier
/// whose wait covers only what was submitted.
#[derive(Debug)]
pub struct StreamsSubmitted {
    _minted: (),
}

impl StreamsSubmitted {
    /// The proof, for a caller that just submitted every stream's work.
    ///
    /// # Safety
    ///
    /// Every stream of the device has submitted its queued work (for a
    /// runtime that runs work on submission, run it), and none is issued
    /// until whatever the proof is handed to returns.
    pub(crate) unsafe fn new() -> Self {
        Self { _minted: () }
    }
}
