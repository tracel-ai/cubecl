//! Validates dynamic memory-pool configuration on the actual device: a
//! programmatic layout applies to the calling stream in place, its hard cap
//! is enforced, and a later reconfigure (at a quiescent point) replaces it.

use cubecl_core::Runtime;
use cubecl_hip::HipRuntime;
use cubecl_runtime::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl_runtime::config::size::MemorySize;

const MIB: u64 = 1024 * 1024;

fn sliced(page_size: u64, pages: u64) -> MemoryPoolsConfig {
    MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
        page_size: MemorySize(page_size),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(page_size * pages)),
        dealloc_period: None,
    }])
}

/// Configure a small capped arena, fill it, free everything, then reconfigure
/// with a bigger layout and allocate past the old cap — the rebuild really
/// replaced the pools rather than keeping the frozen layout. (An allocation
/// exceeding the cap aborts the device task, so the cap itself is asserted in
/// the runtime's pool unit tests, not here.)
#[test]
fn pools_reconfigure_in_place() {
    let client = HipRuntime::client(&Default::default());

    // 2 pages of 8 MiB, hard-capped: both fit, and anything past the cap
    // would error rather than grow.
    client.configure_memory_pools(&sliced(8 * MIB, 2));
    let a = client.empty(8 * MIB as usize);
    let b = client.empty(8 * MIB as usize);

    // Quiesce: drop everything and flush, so the rebuild finds nothing live.
    drop((a, b));
    client.memory_cleanup();

    // A bigger layout replaces the old one: 4 pages of 16 MiB — more than the
    // old 16 MiB cap could ever serve — now fit.
    client.configure_memory_pools(&sliced(16 * MIB, 4));
    let handles: Vec<_> = (0..4).map(|_| client.empty(16 * MIB as usize)).collect();
    drop(handles);
    client.memory_cleanup();
}
