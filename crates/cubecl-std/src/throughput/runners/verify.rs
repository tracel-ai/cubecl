use cubecl_common::profile::Duration;
use cubecl_runtime::{client::Client, server::Handle, throughput::ThroughputError};

/// Runs one pass and confirms it wrote `written`, before any pass is timed.
///
/// A launch that fails leaves its failure on the buffers it never wrote, and
/// `sync` answers `Ok` regardless. A sample has no way to say so, and would
/// time the launch overhead and report it as bandwidth.
///
/// # Errors
///
/// [`ThroughputError::Launch`] when the pass did not run. The cause is logged
/// by the device where it happened.
pub fn verify(
    client: &Client,
    sample: impl Fn(usize) -> Duration,
    written: &Handle,
) -> Result<(), ThroughputError> {
    sample(1);

    cubecl_core::future::block_on(client.sync_buffers([written]))
        .map_err(|_| ThroughputError::Launch)
}
