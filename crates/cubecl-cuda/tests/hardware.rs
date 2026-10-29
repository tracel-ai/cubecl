//! What the runtime reads off the device it opens.

use cubecl_cuda::CudaRuntime;
use cubecl_server::runtime::Runtime;

/// The L2 is reported as the last-level cache: every NVIDIA GPU has one, and a
/// consumer that sizes a read past the cache reads past an assumed size
/// without it, which a large L2 outgrows.
#[test]
fn the_last_level_cache_is_the_l2() {
    let client = CudaRuntime::client(&Default::default());
    let bytes = client
        .properties()
        .hardware
        .last_level_cache_size
        .expect("the driver reports the L2");
    assert!(bytes >= 1 << 20, "an L2 of {bytes} bytes");
}
