//! Moving a relocation's bytes on a Metal device.

use crate::memory::MetalStorage;
use cubecl_server::memory_management::relocation::{CopyQueue, StorageCopy};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::ComputeStorage;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

/// Metal's device-to-device copy: a blit of its own, committed and waited on
/// outside the stream's dispatch batch.
///
/// The stream ends its batch before a relocation starts, so the blits here
/// follow every dispatch that could still read what moves.
pub(crate) struct MetalCopies {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The last blit committed, which [`wait_copies`](CopyQueue::wait_copies)
    /// waits for.
    committed: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl MetalCopies {
    pub(crate) fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Self {
            queue,
            committed: None,
        }
    }
}

impl CopyQueue<MetalStorage> for MetalCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        // The stream committed and waited for its own work before handing the
        // relocation over.
        Ok(())
    }

    fn copy(&mut self, storage: &mut MetalStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;

        let command_buffer = self.queue.commandBuffer().ok_or_else(|| IoError::Unknown {
            description: "Metal refused a command buffer for a relocation".into(),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })?;
        let blit = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| IoError::Unknown {
                description: "Metal refused a blit encoder for a relocation".into(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })?;

        // SAFETY: both storages are live buffers of this device, and the
        // ranges are the slices the relocation reserved: same size, and on
        // buffers nothing else reads or writes until the blit has completed.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                source.inner(),
                copy.source.offset() as usize,
                target.inner(),
                copy.target.offset() as usize,
                copy.source.size() as usize,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        self.committed = Some(command_buffer);
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        if let Some(command_buffer) = self.committed.take() {
            command_buffer.waitUntilCompleted();
        }
        Ok(())
    }
}
