//! Moving a relocation's bytes on the CPU.

use cubecl_server::memory_management::relocation::{CopyQueue, StorageCopy};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::{BytesStorage, ComputeStorage};

/// The CPU's device-to-device copy: the memory is the host's, so a copy is a
/// `memcpy` and it has landed by the time it returns.
pub(crate) struct CpuCopies;

impl CopyQueue<BytesStorage> for CpuCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        Ok(())
    }

    fn copy(&mut self, storage: &mut BytesStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let mut target = storage.get(&copy.target)?;
        target.write().copy_from_slice(source.read());
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        Ok(())
    }
}
