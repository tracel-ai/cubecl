//! Moving a relocation's bytes on a Metal device.

use crate::memory::MetalStorage;
use cubecl_server::memory_management::relocation::{CopyQueue, StorageCopy};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::ComputeStorage;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

/// Metal's device-to-device copy: blits encoded on a command buffer of their
/// own, committed together and waited on outside the stream's dispatch batch.
///
/// The stream ends its batch before a relocation starts, so the blits here
/// follow every dispatch that could still read what moves.
pub(crate) struct MetalCopies {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The blits encoded since the last commit, and the command buffer they
    /// are encoded on.
    pending: Option<Blits>,
}

struct Blits {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
}

impl MetalCopies {
    pub(crate) fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Self {
            queue,
            pending: None,
        }
    }

    /// The blits being encoded, opened on first use.
    fn blits(&mut self) -> Result<&mut Blits, IoError> {
        if self.pending.is_none() {
            let command_buffer = self.queue.commandBuffer().ok_or_else(|| IoError::Unknown {
                description: "Metal refused a command buffer for a relocation".into(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })?;
            let encoder = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| IoError::Unknown {
                    description: "Metal refused a blit encoder for a relocation".into(),
                    backtrace: cubecl_environment::backtrace::BackTrace::capture(),
                })?;
            self.pending = Some(Blits {
                command_buffer,
                encoder,
            });
        }
        Ok(self.pending.as_mut().expect("opened above"))
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
        let blits = self.blits()?;

        // SAFETY: both storages are live buffers of this device, and the
        // ranges are the slices the relocation reserved: same size, and on
        // buffers nothing else reads or writes until the blit has completed.
        unsafe {
            blits
                .encoder
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source.inner(),
                    copy.source.offset() as usize,
                    target.inner(),
                    copy.target.offset() as usize,
                    copy.source.size() as usize,
                );
        }
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        if let Some(blits) = self.pending.take() {
            blits.encoder.endEncoding();
            blits.command_buffer.commit();
            blits.command_buffer.waitUntilCompleted();
        }
        Ok(())
    }
}
