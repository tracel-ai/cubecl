use super::{CopyQueue, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::{BytesStorage, StorageAllocator};

/// Copies between the allocations of a [`BytesStorage`]: host memory, so a
/// copy is a `memcpy` and has landed by the time it returns.
///
/// Waits on nothing: whoever runs work against the memory waits for it before
/// handing the relocation over.
#[derive(Debug, Default)]
pub struct HostCopies;

impl CopyQueue<BytesStorage> for HostCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        Ok(())
    }

    fn copy(&mut self, storage: &mut BytesStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source);
        let mut target = storage.get(&copy.target);
        target.write().copy_from_slice(source.read());
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        Ok(())
    }
}
