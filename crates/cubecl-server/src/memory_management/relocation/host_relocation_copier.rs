use super::{RelocationCopier, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::{BytesStorage, StorageAllocator};

/// Copies between the allocations of a [`BytesStorage`]: host memory, so a
/// copy is a `memcpy` and has landed by the time it returns.
///
/// Waits on nothing: whoever runs work against the memory waits for it before
/// handing the relocation over.
#[derive(Debug, Default)]
pub(crate) struct HostRelocationCopier;

impl RelocationCopier<BytesStorage> for HostRelocationCopier {
    fn wait_device(&mut self) -> Result<(), ServerError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy lands the source's bytes in the target by the time it returns.
    #[test]
    fn a_copy_lands_before_it_returns() {
        let mut storage = BytesStorage::default();
        let source = storage
            .allocate_bytes(4)
            .expect("the host should have 4 bytes");
        let target = storage
            .allocate_bytes(4)
            .expect("the host should have 4 bytes");
        storage.get(&source).write().fill(9);

        // SAFETY: two distinct 4-byte host allocations, neither touched until
        // the copy returns.
        unsafe {
            HostRelocationCopier.copy(
                &mut storage,
                &StorageCopy {
                    source,
                    target: target.clone(),
                },
            )
        }
        .expect("a host copy should land");

        assert!(storage.get(&target).read().iter().all(|byte| *byte == 9));
    }
}
