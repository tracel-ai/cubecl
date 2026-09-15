pub mod compute_cmma;
pub mod compute_direct;
pub mod launch_overhead;
pub mod memory_direct;
pub mod memory_probe;
pub mod memory_read;
pub mod memory_write;

use cubecl_common::profile::{Duration, Instant};
use cubecl_environment::future::DynFut;
use cubecl_runtime::client::Client;

/// The sample every kernel probe builds: launch `iterations` of the kernel,
/// wait for the device, report the wall-clock time that took.
///
/// The launch is synchronous; only the wait for the device is awaited, which
/// is what lets a probe run on the browser.
pub(super) fn timed(
    client: Client,
    launch: impl Fn(usize) + Send + Sync + 'static,
) -> Box<dyn Fn(usize) -> DynFut<Duration>> {
    Box::new(move |iterations| {
        let client = client.clone();
        let start = Instant::now();
        launch(iterations);
        Box::pin(async move {
            let _ = client.sync().await;
            start.elapsed()
        })
    })
}
