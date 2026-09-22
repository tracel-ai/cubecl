//! Moving a relocation's bytes on a wgpu device.

use crate::compute::storage::WgpuStorage;
use cubecl_server::memory_management::relocation::{CopyQueue, StorageCopy};
use cubecl_server::server::{IoError, ServerError};
use cubecl_server::storage::ComputeStorage;

/// wgpu's device-to-device copy: a buffer-to-buffer copy of its own, submitted
/// and waited on outside the stream's encoder.
///
/// The stream submits what it has queued before a relocation starts, so the
/// copies here follow every launch that could still read what moves.
pub(crate) struct WgpuCopies {
    device: wgpu::Device,
    queue: wgpu::Queue,
    submission: Option<wgpu::SubmissionIndex>,
}

impl WgpuCopies {
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            submission: None,
        }
    }
}

impl CopyQueue<WgpuStorage> for WgpuCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        // The stream's own work is submitted by the caller; this waits for the
        // device to finish it.
        self.wait(None);
        Ok(())
    }

    fn copy(&mut self, storage: &mut WgpuStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL Relocation Encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &source.buffer,
            source.offset,
            &target.buffer,
            target.offset,
            source.size,
        );
        self.submission = Some(self.queue.submit([encoder.finish()]));
        Ok(())
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        let submission = self.submission.take();
        self.wait(submission);
        Ok(())
    }
}

impl WgpuCopies {
    /// Wait for the device, up to `submission` when one is named.
    fn wait(&self, submission: Option<wgpu::SubmissionIndex>) {
        #[cfg(not(target_family = "wasm"))]
        if let Err(err) = self.device.poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: None,
        }) {
            log::warn!("wgpu: relocation poll failed ({err})");
        }
        #[cfg(target_family = "wasm")]
        let _ = submission;
    }
}
