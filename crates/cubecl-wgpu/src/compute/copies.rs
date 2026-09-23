//! Moving a relocation's bytes on a wgpu device.

use crate::compute::storage::WgpuStorage;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::memory_management::relocation::{CopyQueue, StorageCopy};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::ComputeStorage;

/// wgpu's device-to-device copy: buffer-to-buffer copies recorded on an
/// encoder of their own, submitted together and waited on outside the stream's
/// encoder.
///
/// The stream submits what it has queued before a relocation starts, so the
/// copies here follow every launch that could still read what moves.
pub(crate) struct WgpuCopies {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            batch: None,
        }
    }

    /// Wait for the device, up to `submission` when one is named.
    fn wait(&self, submission: Option<wgpu::SubmissionIndex>) -> Result<(), ServerError> {
        #[cfg(not(target_family = "wasm"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: submission,
                timeout: None,
            })
            .map_err(|err| ServerError::Generic {
                reason: format!("wgpu: waiting on a relocation failed ({err})"),
                backtrace: BackTrace::capture(),
            })?;
        #[cfg(target_family = "wasm")]
        let _ = submission;
        Ok(())
    }
}

impl CopyQueue<WgpuStorage> for WgpuCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        // The stream's own work is submitted by the caller; this waits for the
        // device to finish it.
        self.wait(None)
    }

    fn copy(&mut self, storage: &mut WgpuStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;

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
