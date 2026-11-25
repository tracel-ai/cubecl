//! Moving a relocation's bytes on a wgpu device.

use crate::compute::device_poison::PoisonWatch;
use crate::compute::storage::WgpuStorage;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::memory_management::relocation::{
    RelocationCopier, StorageCopy, StreamsSubmitted,
};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::StorageAllocator;

/// wgpu's device-to-device copy: buffer-to-buffer copies recorded on an
/// encoder of their own, submitted together and waited on outside the stream's
/// encoder.
///
/// Every stream submitted what it queued before a relocation starts, as the
/// [`StreamsSubmitted`] it is built with proves, so the copies here follow
/// every launch that could still read what moves.
pub(crate) struct WgpuCopies {
    device: wgpu::Device,
    queue: wgpu::Queue,
    poison: PoisonWatch,
    /// The copies recorded since the last submission.
    batch: Option<Batch>,
}

/// Copies recorded together, and the error scope that catches any the device
/// refuses: a copy that did not land must fail the relocation, never be
/// committed.
struct Batch {
    encoder: wgpu::CommandEncoder,
    errors: wgpu::ErrorScopeGuard,
}

impl WgpuCopies {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        poison: PoisonWatch,
        _submitted: &StreamsSubmitted,
    ) -> Self {
        Self {
            device,
            queue,
            poison,
            batch: None,
        }
    }

    /// Wait for the device, up to `submission` when one is named.
    fn wait(&self, submission: Option<wgpu::SubmissionIndex>) -> Result<(), ServerError> {
        self.poison
            .wait_unless_lost(&self.device, &self.queue, submission)
    }
}

// SAFETY: every stream submitted its queued work before the copier was built,
// as the `StreamsSubmitted` `new` takes proves, and the copies go on the same
// queue after it. Natively, `wait_device` polls the device until every
// submission so far completed, and `wait_copies` submits the batch and polls
// until that submission completed. On wasm the polls return at once, but
// WebGPU runs its one queue in order: the copies run after every earlier
// submission, and every later one runs after the copies.
unsafe impl RelocationCopier<WgpuStorage> for WgpuCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        // Every stream's work is submitted already; this waits for the device
        // to finish it.
        self.wait(None)
    }

    unsafe fn copy(
        &mut self,
        storage: &mut WgpuStorage,
        copy: &StorageCopy,
    ) -> Result<(), IoError> {
        let source = storage.get(&copy.source);
        let target = storage.get(&copy.target);

        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            errors: device.push_error_scope(wgpu::ErrorFilter::Validation),
            encoder: device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL Relocation Encoder"),
            }),
        });
        // wgpu copies whole words. The slices are padded to the device
        // alignment, which is a multiple of that, so the rounded size stays
        // inside both.
        batch.encoder.copy_buffer_to_buffer(
            &source.buffer,
            source.offset,
            &target.buffer,
            target.offset,
            source.size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
        );
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        let Some(Batch { encoder, errors }) = self.batch.take() else {
            return Ok(());
        };
        let submission = self.queue.submit([encoder.finish()]);
        self.wait(Some(submission))?;
        match cubecl_environment::future::block_on(errors.pop()) {
            Some(error) => Err(ServerError::Generic {
                reason: format!("wgpu: a relocation copy was refused ({error})"),
                backtrace: BackTrace::capture(),
            }),
            None => Ok(()),
        }
    }
}
