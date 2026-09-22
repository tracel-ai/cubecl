//! The streams a relocation copies on.

use super::{DeviceStream, Driver};
use crate::memory_management::drop_queue::Fence;
use crate::memory_management::relocation::{CopyQueue, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::ComputeStorage;
use alloc::vec::Vec;

/// A command's streams, as the queue a relocation copies on: the current one
/// carries the copies, and the wait covers every stream the command resolved
/// plus whatever the driver runs outside them.
pub(crate) struct StreamCopies<'a, D: Driver> {
    /// Every stream the command resolved, to wait on before a byte moves.
    signals: Vec<<D::Stream as DeviceStream>::Signal>,
    /// The stream the copies are enqueued on.
    queue: <D::Stream as DeviceStream>::Signal,
    ctx: &'a mut D::Context,
}

impl<'a, D: Driver> StreamCopies<'a, D> {
    pub(crate) fn new(
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

impl<D: Driver> CopyQueue<<D::Stream as DeviceStream>::DeviceStorage> for StreamCopies<'_, D> {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        for signal in self.signals.iter() {
            D::Stream::fence(*signal).wait()?;
        }
        D::wait_outside_streams(self.ctx)
    }

    fn copy(
        &mut self,
        storage: &mut <D::Stream as DeviceStream>::DeviceStorage,
        copy: &StorageCopy,
    ) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;
        // SAFETY: a relocation's copy is between two live slices of the same
        // size on distinct pages, and nothing reads the target or writes the
        // source until `wait_copies` returns.
        unsafe { D::copy_on_device(&source, &target, self.queue) }
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        D::Stream::fence(self.queue).wait()
    }
}
