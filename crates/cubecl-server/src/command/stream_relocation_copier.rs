//! The streams a relocation copies on.

use super::{DeviceStream, Driver};
use crate::memory_management::drop_queue::StreamSyncer;
use crate::memory_management::relocation::{RelocationCopier, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::StorageAllocator;
use crate::stream::ResolvedStreams;
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

/// A command's streams, as the queue a relocation copies on: the current one
/// carries the copies, and the wait covers every stream the pool has created
/// plus whatever the driver runs outside them.
pub struct StreamRelocationCopier<'a, D: Driver> {
    /// Every stream the pool has created, to wait on before a byte moves.
    signals: Vec<<D::Stream as DeviceStream>::Signal>,
    /// The stream the copies are enqueued on.
    queue: <D::Stream as DeviceStream>::Signal,
    ctx: &'a mut D::Context,
}

impl<'a, D: Driver> StreamRelocationCopier<'a, D> {
    /// The copies of a relocation queued on `queue`, waiting on a signal of
    /// every stream `streams` holds.
    pub fn new(
        streams: &mut ResolvedStreams<'_, D::Backend>,
        queue: StreamId,
        ctx: &'a mut D::Context,
    ) -> Self {
        // Rare enough to gather the signals as it goes: it waits on the whole
        // device anyway.
        let signals = streams.all().map(|stream| stream.signal()).collect();
        let queue = streams.get(&queue).signal();
        Self {
            signals,
            queue,
            ctx,
        }
    }
}

// SAFETY: `new` reads a signal of every stream the command's pool holds, so
// `wait_device` waits on a fence of every stream of the device, each recorded
// after the work it enqueued, and on what the driver runs outside them: every
// earlier work has finished before the first copy. `wait_copies` waits on a
// fence of the queue every copy went on, recorded after the last of them.
unsafe impl<D: Driver> RelocationCopier<<D::Stream as DeviceStream>::DeviceStorage>
    for StreamRelocationCopier<'_, D>
{
    fn wait_device(&mut self) -> Result<(), ServerError> {
        for signal in self.signals.iter() {
            D::Stream::fence(*signal).sync_host()?;
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
        // SAFETY: the caller upholds `RelocationCopier::copy`'s contract (the
        // two storages are of the same size on distinct live allocations), and
        // this copier's waits order the copy after every earlier work and
        // before every later one.
        unsafe { D::copy_on_device(&source, &target, self.queue) }
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        D::Stream::fence(self.queue).sync_host()
    }
}
