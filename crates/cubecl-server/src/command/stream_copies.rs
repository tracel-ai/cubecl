//! The streams a relocation copies on.

use super::{DeviceStream, Driver};
use crate::memory_management::drop_queue::DeviceWaiter;
use crate::memory_management::relocation::{RelocationCopier, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::StorageAllocator;
use alloc::vec::Vec;

/// A command's streams, as the queue a relocation copies on: the current one
/// carries the copies, and the wait covers every stream the pool has created
/// plus whatever the driver runs outside them.
pub(crate) struct StreamCopies<'a, D: Driver> {
    /// Every stream the pool has created, to wait on before a byte moves.
    signals: Vec<<D::Stream as DeviceStream>::Signal>,
    /// The stream the copies are enqueued on.
    queue: <D::Stream as DeviceStream>::Signal,
    ctx: &'a mut D::Context,
}

impl<'a, D: Driver> StreamCopies<'a, D> {
    /// The queue `queue` names, waiting on the streams `signals` name.
    ///
    /// # Safety
    ///
    /// `signals` holds a signal of every stream of the device: the copier's
    /// `wait_device` waits on those alone.
    pub(crate) unsafe fn new(
        signals: Vec<<D::Stream as DeviceStream>::Signal>,
        queue: <D::Stream as DeviceStream>::Signal,
        ctx: &'a mut D::Context,
    ) -> Self {
        Self {
            signals,
            queue,
            ctx,
        }
    }
}

// SAFETY: `wait_device` waits on a fence of every stream of the device and on
// what the driver runs outside them, and `wait_copies` waits on a fence of the
// queue every copy went on, recorded after the last of them.
unsafe impl<D: Driver> RelocationCopier<<D::Stream as DeviceStream>::DeviceStorage>
    for StreamCopies<'_, D>
{
    fn wait_device(&mut self) -> Result<(), ServerError> {
        for signal in self.signals.iter() {
            D::Stream::fence(*signal).wait()?;
        }
        D::wait_outside_streams(self.ctx)
    }

    unsafe fn copy(
        &mut self,
        storage: &mut <D::Stream as DeviceStream>::DeviceStorage,
        copy: &StorageCopy,
    ) -> Result<(), IoError> {
        let source = storage.get(&copy.source);
        let target = storage.get(&copy.target);
        // SAFETY: the caller upholds `RelocationCopier::copy`'s contract: the
        // two storages are of the same size on distinct allocations, and
        // nothing reads the target or writes the source until `wait_copies`
        // returns.
        unsafe { D::copy_on_device(&source, &target, self.queue) }
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        D::Stream::fence(self.queue).wait()
    }
}
