//! A stream as the queue a relocation copies on.

use super::{DeviceStream, Driver};
use crate::memory_management::relocation::{CopyQueue, StorageCopy};
use crate::server::IoError;
use crate::storage::ComputeStorage;

/// The stream a relocation's copies are enqueued on. Its own device memory
/// resolves the storages: a relocation only moves allocations within one
/// stream's pools.
pub(crate) struct StreamCopies<'a, D: Driver> {
    stream: &'a mut D::Stream,
}

impl<'a, D: Driver> StreamCopies<'a, D> {
    pub(crate) fn new(stream: &'a mut D::Stream) -> Self {
        Self { stream }
    }
}

impl<D: Driver> CopyQueue for StreamCopies<'_, D> {
    type Fence = <D::Stream as DeviceStream>::Fence;

    fn copy(&mut self, copy: &StorageCopy) -> Result<(), IoError> {
        let storage = self.stream.device_memory().storage();
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;
        // SAFETY: a relocation's copy is between two live slices of the same
        // size on distinct pages, and nothing reads the target or writes the
        // source until the relocation waits on this stream's fence.
        unsafe { D::copy_on_device(&source, &target, self.stream) }
    }

    fn fence(&mut self) -> Self::Fence {
        D::Stream::fence(self.stream.signal())
    }
}
