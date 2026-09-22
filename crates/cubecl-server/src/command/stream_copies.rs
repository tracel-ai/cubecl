//! The streams a relocation copies on.

use super::{DeviceStream, Driver};
use crate::memory_management::drop_queue::Fence;
use crate::memory_management::relocation::{CopyQueue, StorageCopy};
use crate::server::{IoError, ServerError};
use crate::storage::ComputeStorage;
use crate::stream::ResolvedStreams;
use alloc::vec::Vec;

/// A command's streams, as the queue a relocation copies on: the current one
/// carries the copies, and the wait covers every stream the command resolved
/// plus whatever the driver runs outside them.
pub(crate) struct StreamCopies<'a, 'b, D: Driver> {
    streams: &'a mut ResolvedStreams<'b, D::Backend>,
    ctx: &'a mut D::Context,
}

impl<'a, 'b, D: Driver> StreamCopies<'a, 'b, D> {
    pub(crate) fn new(
        streams: &'a mut ResolvedStreams<'b, D::Backend>,
        ctx: &'a mut D::Context,
    ) -> Self {
        Self { streams, ctx }
    }
}

impl<D: Driver> CopyQueue for StreamCopies<'_, '_, D> {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        let fences: Vec<_> = self
            .streams
            .all()
            .map(|stream| D::Stream::fence(stream.signal()))
            .collect();
        for fence in fences {
            fence.wait()?;
        }
        D::wait_outside_streams(self.ctx)
    }

    fn copy(&mut self, copy: &StorageCopy) -> Result<(), IoError> {
        let stream = self.streams.current();
        let storage = stream.device_memory().storage();
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;
        // SAFETY: a relocation's copy is between two live slices of the same
        // size on distinct pages, and nothing reads the target or writes the
        // source until `wait_copies` returns.
        unsafe { D::copy_on_device(&source, &target, stream) }
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        let stream = self.streams.current();
        D::Stream::fence(stream.signal()).wait()
    }
}
