use alloc::boxed::Box;
use cubecl_common::profile::Instant;
use cubecl_core::{future::block_on, prelude::*};
use cubecl_runtime::throughput::{KernelConfig, ThroughputError};

use crate::throughput::memory_probe;

/// Builds `Client::to_client` from `source` to `destination`, `bytes` per pass, over whichever
/// route the runtime takes. `ops_count` is the bytes moved.
///
/// # Errors
///
/// [`ThroughputError::Allocation`] where `source` has no room for the buffer, and
/// [`ThroughputError::Launch`] where the transfer did not arrive.
pub fn build_kernel(
    source: &Client,
    destination: &Client,
    bytes: u64,
) -> Result<KernelConfig, ThroughputError> {
    let [buffer] = memory_probe::reserve(source, [bytes as usize])?;

    // A failed receive leaves its failure on the destination it never wrote, which a sync does
    // not report.
    let received = source.clone().to_client(buffer.clone(), destination);
    let _ = block_on(source.sync());
    block_on(destination.sync_buffers([&received])).map_err(|_| ThroughputError::Launch)?;

    let (sender, receiver) = (source.clone(), destination.clone());
    let sample = Box::new(move |iterations: usize| {
        let start = Instant::now();
        for _ in 0..iterations {
            sender.clone().to_client(buffer.clone(), &receiver);
        }
        let _ = block_on(sender.sync());
        let _ = block_on(receiver.sync());
        start.elapsed()
    });

    Ok(KernelConfig {
        sample,
        ops_count: bytes as usize,
        min_iterations: 1,
    })
}
