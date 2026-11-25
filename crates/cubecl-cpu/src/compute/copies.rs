//! Moving a relocation's bytes on the CPU.

use crate::compute::threadpool::completion_counter::LaunchProgress;
use cubecl_core::server::{IoError, ServerError};
use cubecl_server::memory_management::relocation::{
    RelocationCopier, StorageCopy, StreamsSubmitted,
};
use cubecl_server::storage::{BytesStorage, StorageAllocator};

/// The CPU's copy: a `memcpy` between host allocations, made once the
/// stream's launches have run.
pub(crate) struct CpuCopies<'a> {
    /// The launches of the stream whose memory moves, waited on before the
    /// first copy.
    progress: &'a mut LaunchProgress,
}

impl<'a> CpuCopies<'a> {
    /// The copies of a relocation running after every stream's work was
    /// submitted, as `_submitted` proves, waiting on `progress` before the
    /// first one.
    pub(crate) fn new(progress: &'a mut LaunchProgress, _submitted: &StreamsSubmitted) -> Self {
        Self { progress }
    }
}

// SAFETY: the `StreamsSubmitted` `new` takes proves the scheduler flushed every
// stream, and a CPU stream's flush waits for its launch progress: every launch
// of every stream issued before the relocation has run. A copy is a `memcpy`
// that has landed by the time `copy` returns, so `wait_copies` has nothing
// left to wait for.
unsafe impl RelocationCopier<BytesStorage> for CpuCopies<'_> {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        self.progress.wait();
        Ok(())
    }

    unsafe fn copy(
        &mut self,
        storage: &mut BytesStorage,
        copy: &StorageCopy,
    ) -> Result<(), IoError> {
        let source = storage.get(&copy.source);
        let mut target = storage.get(&copy.target);
        target.write().copy_from_slice(source.read());
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        Ok(())
    }
}
